"""
ppo_actor_critic.py
===================

Stage C — Top-level PPO actor-critic orchestrator for the v2 CPO scheduler.

Combines :class:`HeteroEncoder`, :class:`CrossAttentionActor`, and
:class:`DualCritic` into a single ``nn.Module`` that:

1. encodes a (batched) heterograph,
2. gathers the per-graph "current task" query,
3. runs the placement and delay actor heads,
4. runs the dual critic,
5. exposes ``act()``, ``evaluate_actions()``, and ``get_value()`` —
   the three methods that the PPO trainer in Stage D will call.

Action mode handling
--------------------
``action_mode`` is set at construction time and dictates:

* ``auto_only``  → action = placement only.  ``act()`` returns
                   ``(action_proc, log_prob_proc, entropy_proc)``.
                   Delay head is built but produces gradient-detached
                   uniform output (so its parameters can warm-start
                   when the model is later loaded with mode=hybrid).

* ``agent_only`` /
  ``hybrid``     → action = (placement, delay).  ``act()`` returns
                   ``(action, log_prob, entropy)`` where ``log_prob =
                   log_prob_proc + log_prob_delay`` and ``entropy =
                   entropy_proc + entropy_delay``.  The trainer is
                   responsible for using the per-channel rewards
                   (``info['reward_channels']``) to compute two
                   separate advantages — this module just exposes
                   ``V_placement`` and ``V_delay`` separately.

Warm-start path (Q3 = "两阶段")
------------------------------
1. Train with ``action_mode='auto_only'`` until convergence; checkpoint
   ``state_dict()`` of this module.
2. Construct a new module with ``action_mode='hybrid'`` (or
   ``'agent_only'`` for the ablation), then call
   ``load_state_dict(ckpt, strict=False)``: encoder + actor.q_proj +
   actor.k_proj + critic.{trunk, value_placement} all transfer
   verbatim.  Only the delay head and ``value_delay`` start fresh.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch, HeteroData

try:
    from .hetero_encoder         import HeteroEncoder, graph_obs_to_hetero_data
    from .cross_attention_actor  import CrossAttentionActor
    from .value_critic           import DualCritic
except ImportError:  # pragma: no cover — flat dev layout
    from hetero_encoder         import HeteroEncoder, graph_obs_to_hetero_data  # type: ignore
    from cross_attention_actor  import CrossAttentionActor                      # type: ignore
    from value_critic           import DualCritic                               # type: ignore


# =====================================================================
# The orchestrator
# =====================================================================
class PPOActorCritic(nn.Module):
    """End-to-end PPO actor-critic for the v2 CPO scheduler."""

    def __init__(
        self,
        action_mode:  str = "auto_only",
        K_delay:      int = 5,
        # Encoder hyperparams
        task_in_dim:  int = 8,
        proc_in_dim:  int = 7,
        edge_dim_t2p: int = 2,    # task→proc edge dim (2=full, 1=fair-Decima)
        hidden:       int = 128,
        num_layers:   int = 2,
        num_heads:    int = 4,
        dropout:      float = 0.1,
        # Critic hyperparam
        critic_hidden: int = 256,
    ):
        super().__init__()
        if action_mode not in ("auto_only", "agent_only", "hybrid"):
            raise ValueError(f"unknown action_mode: {action_mode!r}")
        self.action_mode = action_mode
        self.K_delay     = K_delay

        self.encoder = HeteroEncoder(
            task_in_dim=task_in_dim, proc_in_dim=proc_in_dim,
            edge_dim_t2t=1, edge_dim_p2p=1, edge_dim_t2p=edge_dim_t2p,
            hidden=hidden, num_layers=num_layers,
            heads=num_heads, dropout=dropout,
        )
        self.actor = CrossAttentionActor(
            hidden=hidden, K_delay=K_delay, num_heads=num_heads,
        )
        self.critic = DualCritic(hidden=hidden, trunk_hidden=critic_hidden)

    # =================================================================
    # Internal: encoder + actor forward pass
    # =================================================================
    def _forward_features(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Run encoder + actor; return everything the public methods need."""
        x_task, x_proc, batch_task, batch_proc = self.encoder(batch)

        # Compute batch size B from the Batch object directly.  We CANNOT
        # use ``batch_task.max()+1`` because a graph with zero tasks (e.g.
        # the brief moment between DAG completion and DAG load in the env)
        # has no entries in batch_task, so the max would undercount.
        # ``batch.num_graphs`` is set by PyG's Batch.from_data_list and is
        # authoritative; for an unbatched single graph it falls back to 1.
        B = (int(batch.num_graphs)
             if hasattr(batch, "num_graphs") and batch.num_graphs is not None
             else 1)

        # Pull current_idx_within_graph from the batch.  PyG batching does
        # NOT auto-shift custom integer attributes, so each graph's
        # current_idx is already in its own local frame.  We still need to
        # collect them into a (B,)-shaped tensor in graph order.
        if hasattr(batch["task"], "current_idx") and batch["task"].current_idx is not None:
            current_idx = batch["task"].current_idx.view(-1)
            if current_idx.size(0) != B:
                # Reshape defensively in either direction:
                #  * fewer entries than B → pad with 0 (first task)
                #  * more entries than B → truncate
                # The "more" case can happen for an empty-task graph that
                # still set a current_idx, hence wasn't reflected in B.
                fix = current_idx.new_zeros(B)
                copy_n = min(current_idx.size(0), B)
                fix[:copy_n] = current_idx[:copy_n]
                current_idx = fix
        else:
            current_idx = torch.zeros(B, dtype=torch.long, device=x_task.device)

        # Build action mask from the env's bool mask (carried on
        # data['proc'].action_mask if the caller set it, otherwise from
        # x_proc directly).  At graph-construction time we set it on
        # ``data['proc'].action_mask`` as a (num_proc,) bool tensor; we
        # need to ragged-pad to (B, max_proc).
        if hasattr(batch["proc"], "action_mask") \
                and batch["proc"].action_mask is not None:
            proc_mask_flat = batch["proc"].action_mask.bool()
        else:
            # No mask -> all valid
            proc_mask_flat = torch.ones(
                x_proc.size(0), dtype=torch.bool, device=x_proc.device,
            )
        # Pad to (B, max_proc), padding with False (invalid).
        counts = torch.bincount(batch_proc, minlength=B)
        max_proc = int(counts.max().item()) if counts.numel() > 0 else 0
        action_mask = torch.zeros(B, max_proc, dtype=torch.bool,
                                  device=x_proc.device)
        if x_proc.size(0) > 0:
            arange_within = torch.arange(x_proc.size(0), device=x_proc.device) - \
                torch.cat([
                    x_proc.new_zeros(1, dtype=torch.long),
                    counts.cumsum(0)[:-1],
                ])[batch_proc]
            action_mask[batch_proc, arange_within] = proc_mask_flat
        # Defence: ensure at least one valid action per graph (else
        # log-prob is -inf and NaNs cascade).  Pick the first slot.
        no_valid = ~action_mask.any(dim=-1)        # (B,)
        if no_valid.any():
            action_mask[no_valid, 0] = True

        actor_out = self.actor(
            x_task, x_proc, batch_task, batch_proc,
            current_idx_within_graph=current_idx,
            action_mask=action_mask,
        )
        v_p, v_d = self.critic(x_task, x_proc, batch_task, batch_proc, B)
        return {
            "placement_logits": actor_out["placement_logits"],
            "delay_logits":     actor_out["delay_logits"],
            "placement_mask":   actor_out["placement_mask"],
            "v_placement":      v_p,
            "v_delay":          v_d,
            "batch_size":       B,
        }

    # =================================================================
    # Public: act
    # =================================================================
    @torch.no_grad()
    def act(
        self,
        batch:        Batch,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample actions for the rollout.

        Returns a dict with ``action`` (shape (B,) for auto_only or
        (B, 2) for agent_only/hybrid), ``log_prob`` (B,), ``entropy``
        (B,), ``v_placement`` (B,), and ``v_delay`` (B,).

        IMPORTANT: this method enters eval mode internally to disable
        dropout, so the same (state, action) pair gives the SAME log_prob
        when later replayed by ``evaluate_actions()``.  This is required
        for PPO — the importance ratio ``exp(log_prob_new - log_prob_old)``
        must equal 1.0 at the first PPO epoch when no parameters have
        changed.  See the corresponding ``evaluate_actions()`` for the
        matching eval-mode call.
        """
        was_training = self.training
        self.eval()
        try:
            out = self._forward_features(batch)
            place_dist = Categorical(logits=out["placement_logits"])

            if deterministic:
                action_p = out["placement_logits"].argmax(dim=-1)
            else:
                action_p = place_dist.sample()
            log_prob_p = place_dist.log_prob(action_p)
            entropy_p  = place_dist.entropy()

            if self.action_mode == "auto_only":
                result = {
                    "action":      action_p,                 # (B,)
                    "log_prob":    log_prob_p,
                    "entropy":     entropy_p,
                    "log_prob_p":  log_prob_p,
                    "log_prob_d":  torch.zeros_like(log_prob_p),
                    "entropy_p":   entropy_p,
                    "entropy_d":   torch.zeros_like(entropy_p),
                    "v_placement": out["v_placement"],
                    "v_delay":     out["v_delay"],
                }
            else:
                # agent_only / hybrid
                delay_dist = Categorical(logits=out["delay_logits"])
                if deterministic:
                    action_d = out["delay_logits"].argmax(dim=-1)
                else:
                    action_d = delay_dist.sample()
                log_prob_d = delay_dist.log_prob(action_d)
                entropy_d  = delay_dist.entropy()

                action = torch.stack([action_p, action_d], dim=-1)  # (B, 2)
                result = {
                    "action":      action,
                    "log_prob":    log_prob_p + log_prob_d,
                    "entropy":     entropy_p + entropy_d,
                    "log_prob_p":  log_prob_p,
                    "log_prob_d":  log_prob_d,
                    "entropy_p":   entropy_p,
                    "entropy_d":   entropy_d,
                    "v_placement": out["v_placement"],
                    "v_delay":     out["v_delay"],
                }
        finally:
            if was_training:
                self.train()
        return result

    # =================================================================
    # Public: evaluate_actions  (used in PPO update)
    # =================================================================
    def evaluate_actions(
        self,
        batch:    Batch,
        actions:  torch.Tensor,    # (B,) or (B, 2)
    ) -> Dict[str, torch.Tensor]:
        """Compute log-probs / entropy / values for given (s, a).

        Like ``act()``, this method runs in eval mode so dropout is
        disabled.  PPO's clip-ratio formulation requires that
        ``log_prob_old`` (computed by ``act()`` during rollout) and
        ``log_prob_new`` (computed by this method during update) be
        deterministic functions of (state, action, params).  Dropout
        breaks that determinism even within the same forward pass, so
        we disable it everywhere in the policy/value pipeline.

        This matches the convention used by cleanrl, SB3, and most
        public PPO implementations.  The model is still in train mode
        from the optimiser's perspective (parameters get gradients,
        BatchNorm running stats update if any modules use it), just
        with dropout off.
        """
        was_training = self.training
        self.eval()
        try:
            out = self._forward_features(batch)
            place_dist = Categorical(logits=out["placement_logits"])

            if self.action_mode == "auto_only":
                action_p = actions.long().view(-1)
                log_prob_p = place_dist.log_prob(action_p)
                entropy_p  = place_dist.entropy()
                result = {
                    "log_prob":    log_prob_p,
                    "entropy":     entropy_p,
                    "log_prob_p":  log_prob_p,
                    "log_prob_d":  torch.zeros_like(log_prob_p),
                    "entropy_p":   entropy_p,
                    "entropy_d":   torch.zeros_like(entropy_p),
                    "v_placement": out["v_placement"],
                    "v_delay":     out["v_delay"],
                }
            else:
                # agent_only / hybrid
                if actions.dim() == 1:
                    raise ValueError(
                        "Expected actions of shape (B, 2) for agent_only/hybrid, "
                        f"got {tuple(actions.shape)}"
                    )
                action_p = actions[:, 0].long()
                action_d = actions[:, 1].long()
                log_prob_p = place_dist.log_prob(action_p)
                entropy_p  = place_dist.entropy()
                delay_dist = Categorical(logits=out["delay_logits"])
                log_prob_d = delay_dist.log_prob(action_d)
                entropy_d  = delay_dist.entropy()
                result = {
                    "log_prob":    log_prob_p + log_prob_d,
                    "entropy":     entropy_p + entropy_d,
                    "log_prob_p":  log_prob_p,
                    "log_prob_d":  log_prob_d,
                    "entropy_p":   entropy_p,
                    "entropy_d":   entropy_d,
                    "v_placement": out["v_placement"],
                    "v_delay":     out["v_delay"],
                }
        finally:
            if was_training:
                self.train()
        return result

    # =================================================================
    # Public: get_value (used to bootstrap the GAE return at episode tail)
    # =================================================================
    @torch.no_grad()
    def get_value(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        try:
            out = self._forward_features(batch)
            v_p, v_d = out["v_placement"], out["v_delay"]
        finally:
            if was_training:
                self.train()
        return v_p, v_d


# =====================================================================
# Helpers: build a Batch from a list of graph_obs dicts + masks
# =====================================================================
def build_batch(
    graph_obs_list: list,
    action_masks:   list,                    # list of np.ndarray bool, len=B
    device: torch.device | str = "cpu",
    thermal_blind:  bool = False,
) -> Batch:
    """Convert ``B`` env-emitted graph_obs dicts (+ their action masks) into
    a single PyG :class:`Batch` ready for ``act`` / ``evaluate_actions``.

    Parameters
    ----------
    thermal_blind : bool
        If True, strip thermal information from each graph_obs before
        constructing the hetero data:
          - proc_x:          7 cols → 3 cols (drop T_norm, dT/dt,
                             leakage, headroom — keep busy/remaining/is_ASIC)
          - edges_t2p_attr:  2 cols → 1 col  (drop est_temp_rise — keep
                             est_exec_time)
        This is used for fair-Decima training/inference — the model is
        constructed with proc_in_dim=3 + edge_dim_t2p=1, and this flag
        ensures the obs dict matches that input shape regardless of what
        the env emits.  More robust than env-side wrapping because it
        runs in the main training process (no AsyncVectorEnv subprocess
        pickle issues).
    """
    import numpy as np

    edge_dim_t2p = 1 if thermal_blind else 2

    data_list = []
    for g, m in zip(graph_obs_list, action_masks):
        if thermal_blind:
            g = _strip_thermal_from_graph_obs(g)
        d = graph_obs_to_hetero_data(g, device=device,
                                      edge_dim_t2p=edge_dim_t2p)
        m_arr = np.asarray(m, dtype=bool).reshape(-1)
        d["proc"].action_mask = torch.tensor(m_arr, dtype=torch.bool, device=device)
        data_list.append(d)
    return Batch.from_data_list(data_list)


def _strip_thermal_from_graph_obs(g: dict) -> dict:
    """Return a SHALLOW COPY of ``g`` with thermal info stripped out.

    Strips:
      - proc_x:          (N, 7) → (N, 3)  (drop cols 0:4)
      - edges_t2p_attr:  (E, 2) → (E, 1)  (drop col 1)

    All other keys are passed through by reference (cheap shallow copy).
    Idempotent: if shapes are already (N, 3) / (E, 1), returns g unchanged.
    """
    import numpy as np

    new_g = dict(g)   # shallow copy — only the two keys we modify diverge

    # ---- proc_x ----
    proc_x = g.get("proc_x", None)
    if proc_x is not None:
        arr = np.asarray(proc_x, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 7:
            new_g["proc_x"] = arr[:, 4:7].tolist()
        # else: already 3-col, or empty, or unexpected — leave alone

    # ---- edges_t2p_attr ----
    edges = g.get("edges_t2p_attr", None)
    if edges is not None:
        arr = np.asarray(edges, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            new_g["edges_t2p_attr"] = arr[:, 0:1].tolist()

    return new_g
