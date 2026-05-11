"""
baselines/decima_true.py — Decima (Mao 2019 SIGCOMM) faithful reproduction
==========================================================================

Step 1-7 implementation per
``paper_drafts/decima_true_implementation_checklist.md``.  See that file
for per-step acceptance criteria.

Two variants supported via yaml config (model class identical for both;
the only difference is the reward signal seen during training):

  - vanilla:  env.reward_mode=makespan_only  (Mao 2019 faithful)
              configs/train_decima_true_vanilla.yaml
              checkpoints/decima_true_vanilla_N17/

  - thermal:  env.reward_mode=thermal_aware  (controlled comparison vs Ours)
              configs/train_decima_true_thermal.yaml
              checkpoints/decima_true_thermal_N17/

Architectural choices vs baselines/decima_fair.py
-------------------------------------------------
* Homogeneous GCN (``torch_geometric.nn.GCNConv`` × ``num_gcn_layers``)
  with mean aggregation — replaces hetero GAT
* Single node type: task and processor features are projected to the
  same hidden dim and concatenated with a 2-d one-hot type tag, then
  fed through a single GCN stack
* Edges: DAG precedence (task↔task) + processor self-loops only; no
  task-proc affinity edges, no thermal coupling edges
* Score head: a Mao-style "single-stage softmax" over processors,
  conditioned on the embedding of the currently-dispatched task.
  This is a faithful adaptation of Mao §3-4 to our env's
  ``Discrete(N_proc)`` action space — Mao's Stage A (DAG selection) is
  performed implicitly by the env's ready-task queue ordering, leaving
  Stage B (the policy's only learned decision) as a single processor
  choice per dispatched task.
* REINFORCE with moving-average baseline (NOT PPO, NOT GAE, NOT
  clipping); one gradient step per episode

Bias check
----------
HANDOFF §4 explicitly forbids reusing HeteroEncoder + thermal-mask
hack here. This module imports ``torch_geometric.nn.GCNConv`` directly
and does not touch ``cpo_thermal_v2.models.HeteroEncoder``.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .base import Action, BaseScheduler


# Default feature dimensions emitted by cpo_thermal_env._build_graph_obs.
# Used as fallback when the env hasn't reported them.
_DEFAULT_TASK_IN_DIM = 8
_DEFAULT_PROC_IN_DIM = 7


# =====================================================================
# Step 1 + Step 2 — Homogeneous GCN policy + two-stage Mao score
# =====================================================================
class DecimaTruePolicy(nn.Module):
    """Homogeneous GCN policy (Mao 2019 §3-4) adapted to our env.

    Forward signature
    -----------------
    ``forward(graph_obs, action_mask) -> proc_logits``

    ``graph_obs`` is the per-env dict produced by
    ``cpo_thermal_env._build_graph_obs`` (or a thermal-stripped copy).
    Only ``task_x`` and ``edges_t2t`` are consumed; ``proc_x`` is
    consumed for the processor projection (the homogeneous trunk needs
    a feature vector for every node, regardless of which keys the
    *gradient signal* sees) but the env's ``observation_keys`` filter
    elects whether thermal columns survive.

    Returns logits of shape ``(N_proc,)``.  Caller is responsible for
    masking (Step 3 handles this).
    """

    def __init__(
        self,
        *,
        task_in_dim:    int   = _DEFAULT_TASK_IN_DIM,
        proc_in_dim:    int   = _DEFAULT_PROC_IN_DIM,
        hidden_dim:     int   = 256,
        num_gcn_layers: int   = 4,
    ):
        super().__init__()
        self.hidden_dim     = int(hidden_dim)
        self.num_gcn_layers = int(num_gcn_layers)

        # Per-type input projections.  Task and proc features have
        # different dimensionality (8 vs 7 in the default schema) so
        # they're projected separately to ``hidden_dim`` before being
        # concatenated with a one-hot type tag.  The concat dim is
        # therefore ``hidden_dim + 2``; the first GCNConv consumes that.
        self.task_proj = nn.Linear(task_in_dim, hidden_dim)
        self.proc_proj = nn.Linear(proc_in_dim, hidden_dim)

        # GCN stack with mean aggregation (Mao §3 prescribes mean agg;
        # default GCNConv uses symmetric normalisation which approximates
        # mean for regular graphs.  We explicitly set ``aggr='mean'`` via
        # the parent ``MessagePassing.__init__`` override below).
        layers: List[GCNConv] = []
        in_dim = hidden_dim + 2   # + 2 one-hot type bits
        for _ in range(num_gcn_layers):
            conv = GCNConv(in_dim, hidden_dim, aggr="mean")
            layers.append(conv)
            in_dim = hidden_dim
        self.gcn_layers = nn.ModuleList(layers)

        # Stage-B score head: conditions on the current task's embedding
        # and a processor embedding to produce a scalar score per proc.
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # -----------------------------------------------------------------
    # Step 2 — forward pass through the homogeneous GCN
    # -----------------------------------------------------------------
    def forward(
        self,
        graph_obs:   Dict[str, Any],
        action_mask: np.ndarray,
    ) -> torch.Tensor:
        """Return masked logits of shape ``(N_proc,)`` for one env state.

        Implementation notes
        --------------------
        * If ``task_x`` is empty (no uncompleted tasks), the env should
          have ended the episode already.  As a defensive fallback we
          return uniform logits over unmasked procs.
        * The current-task embedding is read via
          ``graph_obs['current_task_idx']``; downstream sampling code
          attributes the gradient to whichever processor the policy
          selects for *that* task.
        * Every locally-built tensor lives on the same device as the
          module parameters (HK-3.1.1 fix — previously hard-coded CPU
          regardless of where the model was, causing V100 idle).
        """
        device = next(self.parameters()).device
        N_proc = int(np.asarray(action_mask).shape[0])

        task_x_raw = np.asarray(graph_obs.get("task_x", []),
                                dtype=np.float32)
        proc_x_raw = np.asarray(graph_obs.get("proc_x", []),
                                dtype=np.float32)

        # Defensive: empty task set => uniform over unmasked procs
        if task_x_raw.size == 0 or task_x_raw.ndim != 2:
            # uniform; mask handled by caller
            return torch.zeros(N_proc, device=device)

        # If proc_x is missing (filtered) we need *some* tensor so the
        # GCN can include processor nodes — fall back to zeros.  This
        # is exercised under env.observation_keys=[task_x, edges_t2t]
        # where the policy must still emit per-proc logits.
        if proc_x_raw.size == 0 or proc_x_raw.ndim != 2:
            proc_x_raw = np.zeros((N_proc, self.proc_proj.in_features),
                                  dtype=np.float32)

        task_x = torch.from_numpy(task_x_raw).to(device)
        proc_x = torch.from_numpy(proc_x_raw).to(device)

        # --- per-type projection to hidden_dim ---
        task_h = self.task_proj(task_x)        # (T, hidden_dim)
        proc_h = self.proc_proj(proc_x)        # (P, hidden_dim)

        # --- one-hot type tag ---
        T = task_h.shape[0]
        P = proc_h.shape[0]
        task_tag = torch.cat(
            [torch.ones(T, 1, device=device),
             torch.zeros(T, 1, device=device)], dim=1)
        proc_tag = torch.cat(
            [torch.zeros(P, 1, device=device),
             torch.ones(P, 1, device=device)], dim=1)
        task_h = torch.cat([task_h, task_tag], dim=1)   # (T, hidden+2)
        proc_h = torch.cat([proc_h, proc_tag], dim=1)   # (P, hidden+2)

        # --- combine into single homogeneous node tensor ---
        # Node indices: tasks first (0..T-1), procs next (T..T+P-1)
        x = torch.cat([task_h, proc_h], dim=0)   # (T+P, hidden+2)

        # --- build homogeneous edge_index: DAG precedence + proc self-loops ---
        edges_t2t = graph_obs.get("edges_t2t", []) or []
        edge_list: List[Tuple[int, int]] = []
        for u, v in edges_t2t:
            edge_list.append((int(u), int(v)))
            edge_list.append((int(v), int(u)))   # undirected DAG message-passing
        # processor self-loops only (no proc-proc coupling per Mao
        # — Mao's original is task-task DAG edges only; we add proc
        # self-loops so processors get a feature update from their own
        # node embedding under GCNConv's normalisation)
        for p in range(P):
            edge_list.append((T + p, T + p))

        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long,
                                       device=device).t()

        # --- GCN stack ---
        h = x
        for conv in self.gcn_layers:
            h = F.relu(conv(h, edge_index))

        # --- Stage-B score: condition on current task's embedding ---
        # Mao §3 Stage B picks a (task, proc) pair within the chosen
        # DAG.  In our env the task is fixed (env's ready-task queue);
        # we score processors conditioned on the current task embedding.
        cur_idx = int(graph_obs.get("current_task_idx", 0))
        cur_idx = max(0, min(cur_idx, T - 1))
        cur_task_emb = h[cur_idx]                          # (hidden,)
        proc_embs    = h[T:T + N_proc]                     # (P, hidden)

        # Broadcast cur_task_emb across all procs and concat
        cur_repeat = cur_task_emb.unsqueeze(0).expand(N_proc, -1)
        pair_feat  = torch.cat([cur_repeat, proc_embs], dim=1)  # (P, 2*h)
        logits     = self.score_head(pair_feat).squeeze(-1)     # (P,)

        return logits

    # -----------------------------------------------------------------
    # Step 3 — action sampling + masking
    # -----------------------------------------------------------------
    def select_action(
        self,
        graph_obs:   Dict[str, Any],
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) a processor index.

        Returns
        -------
        action      : int                   chosen processor index
        log_prob    : Tensor (scalar)       log π(action | state)
        entropy     : Tensor (scalar)       H[π( · | state)]

        The mask is applied **before** the softmax so the resulting
        distribution is renormalised over only valid procs.
        """
        action_mask_np = np.asarray(action_mask, dtype=bool)
        valid_idx = np.where(action_mask_np)[0]
        if valid_idx.size == 0:
            raise RuntimeError(
                "DecimaTruePolicy.select_action: no valid procs "
                "(action_mask is all False)."
            )

        logits = self.forward(graph_obs, action_mask_np)
        device = logits.device

        # Mask invalid procs with -inf (mask tensor must live on the
        # same device as logits — HK-3.1.1 fix).
        mask_t = torch.tensor(action_mask_np, dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask_t, float("-inf"))

        if deterministic:
            action = int(torch.argmax(masked_logits).item())
        else:
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = int(dist.sample().item())

        # Probabilities for log_prob / entropy (recompute clean)
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs_all     = log_probs_all.exp()
        log_prob      = log_probs_all[action]
        # entropy: only over valid procs (others have prob 0)
        valid_mask = mask_t
        entropy = -(probs_all[valid_mask] *
                    log_probs_all[valid_mask]).sum()
        return action, log_prob, entropy


# =====================================================================
# Step 4 — REINFORCE agent with moving-average baseline
# =====================================================================
class DecimaTrueAgent:
    """REINFORCE + moving-average baseline (Mao 2019 §5.1).

    One gradient step per episode.  No GAE, no clipping, no minibatching.
    Optional small entropy bonus to slow collapse (off by default;
    enable via ``entropy_coef > 0`` if pilot shows premature
    entropy collapse).
    """

    def __init__(
        self,
        policy:          DecimaTruePolicy,
        *,
        lr:              float = 5.0e-4,
        gamma:           float = 0.99,
        baseline_window: int   = 50,
        entropy_coef:    float = 0.0,
    ):
        self.policy       = policy
        self.gamma        = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.optimizer    = torch.optim.Adam(policy.parameters(), lr=lr)
        # Ring buffer of recent episode-mean discounted-returns; updated
        # at the end of each episode via ``record_episode_return``.
        self._baseline_buf: deque = deque(maxlen=int(baseline_window))

    # -----------------------------------------------------------------
    @property
    def baseline(self) -> float:
        if not self._baseline_buf:
            return 0.0
        return float(np.mean(self._baseline_buf))

    def record_episode_return(self, ep_return: float) -> None:
        self._baseline_buf.append(float(ep_return))

    # -----------------------------------------------------------------
    def discounted_returns(
        self,
        rewards: List[float],
    ) -> torch.Tensor:
        """Compute G_t = Σ_{k≥0} gamma^k * r_{t+k} for one episode.

        Output tensor lives on the policy's device so REINFORCE's
        ``log_probs * adv`` runs without a device mismatch on V100
        (HK-3.1.1 fix).
        """
        G = 0.0
        out: List[float] = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            out[t] = G
        device = next(self.policy.parameters()).device
        return torch.tensor(out, dtype=torch.float32, device=device)

    # -----------------------------------------------------------------
    def update(
        self,
        log_probs: List[torch.Tensor],
        rewards:   List[float],
        entropies: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """One REINFORCE step from a single episode's trajectory.

        Loss = - mean_t [ log π(a_t|s_t) * (G_t - baseline) ]
               - entropy_coef * mean_t H[π(·|s_t)]

        After the step the moving-average baseline is updated with the
        episode's mean discounted return.
        """
        if not log_probs:
            return {"loss": 0.0, "mean_return": 0.0,
                    "baseline": self.baseline, "n_steps": 0}

        G_t      = self.discounted_returns(rewards)
        baseline = self.baseline
        adv      = G_t - baseline

        # REINFORCE loss (negative because we maximise return)
        log_probs_t = torch.stack(log_probs)
        loss_pg = -(log_probs_t * adv.detach()).mean()

        loss = loss_pg
        loss_entropy_val = 0.0
        if entropies is not None and self.entropy_coef > 0:
            ent_t = torch.stack(entropies)
            loss_entropy = -self.entropy_coef * ent_t.mean()
            loss = loss + loss_entropy
            loss_entropy_val = float(loss_entropy.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mean_return = float(G_t.mean().item())
        self.record_episode_return(mean_return)

        return {
            "loss":         float(loss.item()),
            "loss_pg":      float(loss_pg.item()),
            "loss_entropy": loss_entropy_val,
            "mean_return":  mean_return,
            "baseline":     baseline,
            "n_steps":      int(len(rewards)),
        }


# =====================================================================
# Step 7 — Eval-time scheduler (drop-in for evaluation/runner.py)
# =====================================================================
class DecimaTrueScheduler(BaseScheduler):
    """Inference-only wrapper around a trained DecimaTruePolicy.

    Auto_only action space (single int).  Loads a checkpoint produced
    by ``training/train_decima_true.py``.
    """

    name = "Decima"   # paper-facing name; aligns with decima_fair.py's name

    def __init__(
        self,
        ckpt_path:    str,
        num_nodes:    int  = 17,
        deterministic: bool = True,
        device:       str  = "cpu",
        # Model hyper-params; must match training-time values.
        hidden_dim:     int = 256,
        num_gcn_layers: int = 4,
        task_in_dim:    int = _DEFAULT_TASK_IN_DIM,
        proc_in_dim:    int = _DEFAULT_PROC_IN_DIM,
    ):
        super().__init__(action_mode="auto_only", K_delay=5)
        self.num_nodes     = int(num_nodes)
        self.deterministic = bool(deterministic)
        self.device        = device

        self._model = DecimaTruePolicy(
            task_in_dim    = task_in_dim,
            proc_in_dim    = proc_in_dim,
            hidden_dim     = hidden_dim,
            num_gcn_layers = num_gcn_layers,
        ).to(device)

        state = torch.load(ckpt_path, map_location=device,
                           weights_only=False)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "policy" in state:
                state = state["policy"]
        self._model.load_state_dict(state, strict=True)
        self._model.eval()

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        pass    # stateless (Mao policy is purely reactive)

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        graph_obs = info["graph_obs"][0] if isinstance(
            info["graph_obs"], (list, np.ndarray)
        ) else info["graph_obs"]
        action_mask = np.asarray(info["action_mask"], dtype=bool)

        with torch.no_grad():
            action, _logp, _ent = self._model.select_action(
                graph_obs, action_mask,
                deterministic=self.deterministic,
            )
        return self._wrap_action(int(action))
