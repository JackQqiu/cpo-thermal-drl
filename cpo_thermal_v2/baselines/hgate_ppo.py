"""
baselines/hgate_ppo.py — HGATE-PPO (Wu 2025 IEEE IoT Journal) reproduction
==========================================================================

Implementation: BLOCKED — scaffold only.  See
``paper_drafts/hgate_ppo_checklist.md`` for the 8-step implementation
plan and the wiring decisions to make before Step 1.

Delta vs Ours (CLAUDE.md §3 Stage 3):
  - Hetero GATv2 encoder (task / proc node types) but NO RC coupling
    edge attribute and NO proc<->proc edges (Wu 2025's graph is
    task-task DAG + task-proc affinity only).
  - Task-proc edges carry est_time only (no est_temp_rise).
  - Standard placement actor: mean-pool proc embeddings then project
    to (N_proc,) logits via an MLP — NOT cross-attention.
  - Standard single-value critic — NOT dual placement/delay.
  - PPO with clipping + GAE; standard hyper-params from Wu 2025 §4.

§5 ablation chain isolation point: Ours-NoThermal -> HGATE-PPO removes
the CPO-specific RC edge attribute AND the cross-attention placement
actor.  HGATE-PPO -> decima_fair further removes hetero edge typing.

HANDOFF discipline: do NOT import HeteroEncoder / CrossAttentionActor /
DualCritic / PPOActorCritic from cpo_thermal_v2.models.  This is a
faithful reimplementation of Wu 2025; sharing code with Ours would
defeat the controlled comparison §5 depends on.  Build the encoder +
actor + critic from scratch using torch_geometric.nn.GATv2Conv.

Reference:
  Wu et al., "Dependency-Aware Task Offloading Strategy via
  Heterogeneous Graph Neural Network and Deep Reinforcement Learning",
  IEEE IoT Journal 2025, vol 12 no 13, pp 22915-22933,
  doi 10.1109/JIOT.2025.3549441
  https://github.com/JM-Wu-BIT/HGATE-PPO
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .base import Action, BaseScheduler


# =====================================================================
# Step 1 — Heterogeneous GATv2 encoder
# =====================================================================
class HGATEEncoder(nn.Module):
    """Hetero GATv2 stack over (task, proc) nodes.

    Edges (Wu 2025 §3.3):
      ('task', 'prec',  'task') — DAG precedence
      ('task', 'sched', 'proc') — task-proc affinity (scalar weight)

    NO proc<->proc edges (HGATE has no thermal physics).  NO RC edge
    attribute on task-proc edges (the controlled comparison point with
    Ours).
    """

    def __init__(
        self,
        *,
        task_in_dim:    int = 8,
        proc_in_dim:    int = 7,
        hidden_dim:     int = 128,
        num_layers:     int = 2,
        num_heads:      int = 4,
    ):
        super().__init__()
        self.task_in_dim = int(task_in_dim)
        self.proc_in_dim = int(proc_in_dim)
        self.hidden_dim  = int(hidden_dim)
        self.num_layers  = int(num_layers)
        self.num_heads   = int(num_heads)

        # Per-type input projections.  Task and proc features have
        # different dimensionality (8 vs 7 in the env's default
        # schema) so they're projected separately to a common
        # hidden_dim before any cross-type message passing.
        self.task_proj = nn.Linear(task_in_dim, hidden_dim)
        self.proc_proj = nn.Linear(proc_in_dim, hidden_dim)

        # Hetero GATv2 stack — two parallel per-layer convs, one per
        # edge type.  Implemented as explicit ModuleLists rather than
        # via HeteroConv so edge_attr handling on the bipartite
        # ('task','sched','proc') key is unambiguous across PyG
        # versions.  ``add_self_loops=True`` on the task-task DAG
        # conv ensures isolated tasks still update their own
        # embedding even with empty DAG edges.
        self.t2t_convs = nn.ModuleList()   # task -> task (DAG precedence)
        self.t2p_convs = nn.ModuleList()   # task -> proc (sched affinity)
        self.task_norms = nn.ModuleList()
        self.proc_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.t2t_convs.append(GATv2Conv(
                hidden_dim, hidden_dim,
                heads=num_heads,
                concat=False,             # average heads -> output dim = hidden_dim
                add_self_loops=True,
            ))
            self.t2p_convs.append(GATv2Conv(
                (hidden_dim, hidden_dim), hidden_dim,
                heads=num_heads,
                concat=False,
                edge_dim=1,               # est_time only — NO est_temp_rise
                add_self_loops=False,     # bipartite, no self-loops applicable
            ))
            self.task_norms.append(nn.LayerNorm(hidden_dim))
            self.proc_norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        graph_obs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(task_embs, proc_embs)`` of shape (T, H), (P, H).

        Implementation notes
        --------------------
        * Every locally-built tensor lives on the same device as the
          module parameters (HK-3.1.1 discipline).
        * ``_graph_obs_to_hetero`` drops the RC coupling p2p edges
          entirely and keeps only the ``est_time`` column of
          ``edges_t2p_attr`` — this is the §5 ablation isolation
          point with Ours.
        """
        device = next(self.parameters()).device
        task_x, proc_x, edge_t2t, edge_t2p, edge_t2p_attr = \
            self._graph_obs_to_hetero(graph_obs, device)

        h_task = self.task_proj(task_x)        # (T, H)
        h_proc = self.proc_proj(proc_x)        # (P, H)

        for i in range(self.num_layers):
            # task <- task (DAG precedence); self-loops cover isolated nodes
            h_task_new = self.t2t_convs[i](h_task, edge_t2t)
            # proc <- task (sched affinity) with est_time edge_attr
            h_proc_new = self.t2p_convs[i](
                (h_task, h_proc), edge_t2p,
                edge_attr=edge_t2p_attr,
            )
            h_task = F.relu(self.task_norms[i](h_task_new))
            h_proc = F.relu(self.proc_norms[i](h_proc_new))

        return h_task, h_proc

    # -----------------------------------------------------------------
    # Adapter (a Step-4 helper, placed here because forward() needs it
    # and forward() is part of Step 1's acceptance criteria).
    # -----------------------------------------------------------------
    def _graph_obs_to_hetero(
        self,
        graph_obs: Dict[str, Any],
        device:    torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert env graph_obs dict to per-type tensors.

        Returns
        -------
        ``(task_x, proc_x, edge_t2t, edge_t2p, edge_t2p_attr)`` — five
        tensors, all on ``device``.  Drops ``edges_p2p`` and
        ``edges_p2p_attr`` entirely (HGATE has no thermal physics);
        keeps only column 0 of ``edges_t2p_attr`` (est_time) and
        discards column 1 (est_temp_rise) — this is the §5 controlled
        comparison anchor.
        """
        task_x_raw = np.asarray(graph_obs.get("task_x", []),
                                dtype=np.float32)
        proc_x_raw = np.asarray(graph_obs.get("proc_x", []),
                                dtype=np.float32)

        if task_x_raw.size == 0 or task_x_raw.ndim != 2:
            task_x = torch.zeros((0, self.task_in_dim),
                                  dtype=torch.float32, device=device)
        else:
            task_x = torch.from_numpy(task_x_raw).to(device)

        if proc_x_raw.size == 0 or proc_x_raw.ndim != 2:
            proc_x = torch.zeros((0, self.proc_in_dim),
                                  dtype=torch.float32, device=device)
        else:
            proc_x = torch.from_numpy(proc_x_raw).to(device)

        # task -> task DAG precedence edges
        t2t_raw = graph_obs.get("edges_t2t", []) or []
        if t2t_raw:
            edge_t2t = torch.tensor(t2t_raw, dtype=torch.long,
                                     device=device).t()
        else:
            edge_t2t = torch.zeros((2, 0), dtype=torch.long, device=device)

        # task -> proc affinity edges + est_time-only edge_attr
        # (RC ablation anchor — column 1 = est_temp_rise is DROPPED)
        t2p_raw = graph_obs.get("edges_t2p", []) or []
        t2p_attr_raw = graph_obs.get("edges_t2p_attr", []) or []
        if t2p_raw:
            edge_t2p = torch.tensor(t2p_raw, dtype=torch.long,
                                     device=device).t()
            attr_np = np.asarray(t2p_attr_raw, dtype=np.float32)
            if attr_np.ndim == 2 and attr_np.shape[1] >= 1:
                # Keep est_time only.  Slicing column 0:1 preserves a
                # 2-D shape (E, 1) which GATv2Conv(edge_dim=1) expects.
                edge_t2p_attr = torch.from_numpy(
                    attr_np[:, 0:1].copy()).to(device)
            else:
                edge_t2p_attr = torch.zeros(
                    (edge_t2p.shape[1], 1),
                    dtype=torch.float32, device=device)
        else:
            edge_t2p = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_t2p_attr = torch.zeros((0, 1), dtype=torch.float32,
                                         device=device)

        return task_x, proc_x, edge_t2t, edge_t2p, edge_t2p_attr


# =====================================================================
# Step 2 + Step 3 — PPO actor + critic on top of HGATEEncoder
# =====================================================================
class HGATEActorCritic(nn.Module):
    """Standard PPO actor-critic.

    Actor (Wu 2025 §3.4):  mean-pool proc embeddings -> MLP ->
                           Discrete(N_proc) logits.
    Critic:                mean-pool ALL node embeddings -> MLP ->
                           scalar value.

    No cross-attention, no factored heads, no dual critic.
    """

    def __init__(
        self,
        *,
        hidden_dim:     int = 128,
        num_procs:      int = 17,
        num_heads:      int = 4,
        num_gat_layers: int = 2,
        task_in_dim:    int = 8,
        proc_in_dim:    int = 7,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # ``num_procs`` is informational — the actor's per-pair scorer is
        # N-agnostic by construction (Decision 1, Path B) so no learnable
        # layer binds to N.  Stored for diagnostic consistency with the
        # eval-time scheduler's ctor.
        self.num_procs  = int(num_procs)

        self.encoder = HGATEEncoder(
            task_in_dim    = task_in_dim,
            proc_in_dim    = proc_in_dim,
            hidden_dim     = hidden_dim,
            num_layers     = num_gat_layers,
            num_heads      = num_heads,
        )

        # Per-pair scorer (checklist Decision 1, Path B).  For each
        # processor i, compute score_i = MLP([global_ctx, proc_emb_i]).
        # Input dim = 2 * hidden_dim (concat of global context and the
        # proc's own embedding).  N-agnostic — the same MLP applies at
        # any N.
        self.actor_score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Single scalar critic (Wu 2025 §3.4).  Operates on the pooled
        # global context only; NOT dual placement/delay (that's Ours,
        # which uses a factored cross-attention actor with two heads).
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        graph_obs:   Dict[str, Any],
        action_mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)``.

        Shapes
        ------
        logits : (N_proc,)   — per-processor scores, **unmasked**.  The
                                caller (`act()` in Step 3) applies the
                                action mask before softmax.
        value  : ()          — scalar critic V(s).

        Implementation
        --------------
        1. encode graph_obs -> (task_embs[T,H], proc_embs[P,H])
        2. global context = mean-pool over all node embeddings
        3. per-proc logits = actor_score([global_ctx, proc_emb_i])
        4. value = critic(global_ctx)
        """
        device = next(self.parameters()).device
        task_embs, proc_embs = self.encoder(graph_obs)

        # Global context = mean-pool over all node embeddings.  Guard
        # against the degenerate empty-graph case (env always has N
        # procs ≥ 1 so this rarely triggers, but the guard keeps the
        # forward total-function).
        all_embs = torch.cat([task_embs, proc_embs], dim=0)
        if all_embs.shape[0] == 0:
            global_ctx = torch.zeros(self.hidden_dim, device=device)
        else:
            global_ctx = all_embs.mean(dim=0)         # (H,)

        # Per-pair scoring: broadcast global_ctx across procs, concat
        # with each proc's own embedding, run the shared scorer.
        P = proc_embs.shape[0]
        if P == 0:
            # Defensive: emit zeros so the caller's masking still works.
            logits = torch.zeros(0, device=device)
        else:
            ctx_repeat = global_ctx.unsqueeze(0).expand(P, -1)   # (P, H)
            pair_input = torch.cat([ctx_repeat, proc_embs], dim=1)  # (P, 2H)
            logits = self.actor_score(pair_input).squeeze(-1)    # (P,)

        # Critic — collapse (1,) -> () for the scalar contract
        value = self.critic(global_ctx).squeeze(-1)              # ()

        return logits, value

    def act(
        self,
        graph_obs:    Dict[str, Any],
        action_mask:  np.ndarray,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample or argmax a processor index, return PPO-compatible dict.

        Returns
        -------
        dict::
            {
                'action':   int   — chosen processor index
                'log_prob': Tensor (scalar) — log π(action | state) over the
                                              MASKED distribution
                'entropy':  Tensor (scalar) — H[π( · | state)] over the
                                              MASKED distribution
                'value':    Tensor (scalar) — V(s) from the critic
            }

        Implementation
        --------------
        - Run forward to obtain (logits, value).  Critic V(s) is reused
          for the GAE bootstrap so the caller does not need a separate
          ``get_value`` call.
        - Apply ``action_mask`` with ``masked_fill(-inf)`` BEFORE the
          softmax so the resulting distribution is properly renormalised
          over only valid procs.
        - Sample via Categorical or argmax depending on ``deterministic``.
        """
        action_mask_np = np.asarray(action_mask, dtype=bool)
        if not action_mask_np.any():
            raise RuntimeError(
                "HGATEActorCritic.act: action_mask is all False — "
                "no valid procs"
            )

        logits, value = self.forward(graph_obs, action_mask_np)
        device = logits.device

        # Mask invalid procs with -inf so they have zero softmax prob.
        mask_t = torch.tensor(action_mask_np, dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask_t, float("-inf"))

        if deterministic:
            action_t = torch.argmax(masked_logits)
        else:
            dist = torch.distributions.Categorical(logits=masked_logits)
            action_t = dist.sample()
        action_int = int(action_t.item())

        # Always recompute log_prob / entropy from a fresh softmax over
        # the masked logits — this keeps the values defined even in the
        # deterministic branch (where dist was never built).  Entropy
        # is computed only over the VALID positions; invalid positions
        # contribute 0 (their prob is exactly 0).
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs_all     = log_probs_all.exp()
        log_prob      = log_probs_all[action_int]
        entropy = -(probs_all[mask_t] * log_probs_all[mask_t]).sum()

        return {
            "action":   action_int,
            "log_prob": log_prob,
            "entropy":  entropy,
            "value":    value,
        }

    def get_value(self, graph_obs: Dict[str, Any]) -> torch.Tensor:
        """Return scalar V(s); used for GAE bootstrap by the PPO loop.

        Skips the actor scoring path (no per-pair MLP, no action_mask
        needed) — just encoder + global-pool + critic.  This is
        cheaper than calling ``forward`` when the actor's logits are
        not required.
        """
        device = next(self.parameters()).device
        task_embs, proc_embs = self.encoder(graph_obs)
        all_embs = torch.cat([task_embs, proc_embs], dim=0)
        if all_embs.shape[0] == 0:
            global_ctx = torch.zeros(self.hidden_dim, device=device)
        else:
            global_ctx = all_embs.mean(dim=0)         # (H,)
        value = self.critic(global_ctx).squeeze(-1)   # ()
        return value


# =====================================================================
# Step 8 — Eval-time scheduler
# =====================================================================
class HGATEPPOScheduler(BaseScheduler):
    """Inference-only wrapper around a trained ``HGATEActorCritic``.

    Mirrors DecimaTrueScheduler / DecimaFairScheduler shape so
    ``evaluation/runner.py`` can register it uniformly via
    ``--scheduler hgate_ppo``.
    """

    name = "HGATE-PPO"   # paper-facing name (figure / table label)

    def __init__(
        self,
        ckpt_path:    str,
        num_nodes:    int  = 17,
        deterministic: bool = True,
        device:       str  = "cpu",
        # Model hyper-params must match training-time values
        hidden_dim:     int = 128,
        num_gat_layers: int = 2,
        num_heads:      int = 4,
    ):
        super().__init__(action_mode="auto_only", K_delay=5)
        self.num_nodes     = int(num_nodes)
        self.deterministic = bool(deterministic)
        self.device        = device
        # TODO checklist Step 8:
        #   - self._model = HGATEActorCritic(...)
        #   - load_state_dict(strict=True)
        #   - self._model.eval()
        raise NotImplementedError(
            "see hgate_ppo_checklist.md Step 8 — eval-time wrapper"
        )

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        pass    # stateless

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        # TODO checklist Step 8:
        #   - extract graph_obs + action_mask from info
        #   - run self._model.act(deterministic=self.deterministic)
        #   - return self._wrap_action(int(out['action']))
        raise NotImplementedError(
            "see hgate_ppo_checklist.md Step 8 — schedule()"
        )
