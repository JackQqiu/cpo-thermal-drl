"""
baselines/decima_xattn.py — D2 baseline (Decima encoder + cross-attention actor)
==============================================================================

§5 ablation chain isolation point:

    Decima(homog GCN + MLP scorer)
        --(swap MLP scorer for cross-attention actor)-->
    D2 (homog GCN + cross-attention actor)            <-- THIS FILE
        --(swap homog GCN for hetero GAT)-->
    HGATE-PPO (Wu 2025 hetero GAT + MLP scorer)

Why D2
------
The §5 chain compares D2 against Decima (isolating the cross-attention
actor's contribution under a homogeneous GCN trunk) and against
HGATE-PPO (isolating homog GCN vs hetero GAT under a cross-attention
actor).  This is the "encoder architecture vs actor architecture"
contribution question the reviewer raised.

Architecture
------------
* Encoder: ``DecimaXAttnEncoder`` — Mao-style homogeneous GCN over
  (task ∪ proc) nodes.  Task / proc features are projected separately
  to ``hidden_dim``, concatenated with a 2-d one-hot type tag, then
  fed through ``num_gcn_layers`` GCN layers with mean aggregation.
  Edges are DAG precedence (task↔task, undirected) + processor
  self-loops only.  No thermal physics, no task-proc affinity edges
  in the GCN — same encoder constraints as ``decima_true.py``.
  ``forward`` returns ``(task_embs[T,H], proc_embs[P,H])`` split back
  from the joint GCN tensor — the cross-attention actor consumes them
  as separate ragged batches.

* Actor: ``cpo_thermal_v2.models.CrossAttentionActor`` reused as-is.
  Multi-head scaled dot-product attention from the current task's
  embedding (query) over the proc embeddings (keys), with K_delay=1
  for auto_only (delay head exists but is sampled trivially).  The
  cross-attention head is the §5 ablation contribution under test.

* Critic: single scalar critic (mean-pool over all node embeddings ->
  MLP -> scalar V(s)).  Same pattern as HGATE-PPO's critic, NOT the
  DualCritic used by Ours (which is for hybrid-mode dual-channel
  advantage).  D2 is auto_only-only.

* Trainer: PPO with clipping + GAE.  Same multi-env + minibatch
  recipe as ``train_hgate_ppo.py`` (HK-4.5 / HK-4.6 perf fixes apply).

HANDOFF discipline: reuses ``CrossAttentionActor`` from models/ per
user spec ("Actor: Cross-attention actor 复用").  Does NOT touch
``decima.py`` / ``hgate_ppo.py`` / ``cross_attention_actor.py``
themselves — D2 is a pure composition baseline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .base import Action, BaseScheduler
from ..models.cross_attention_actor import CrossAttentionActor


# Default feature dimensions emitted by cpo_thermal_env._build_graph_obs.
# Match decima_true.py / hgate_ppo.py.
_DEFAULT_TASK_IN_DIM = 8
_DEFAULT_PROC_IN_DIM = 7


# =====================================================================
# Encoder — Mao-style homogeneous GCN, returning task/proc splits
# =====================================================================
class DecimaXAttnEncoder(nn.Module):
    """Homogeneous GCN trunk that exposes task and proc embeddings.

    Differs from ``DecimaTruePolicy``:
      - no Mao score head (cross-attention actor does scoring downstream)
      - ``forward`` returns ``(task_embs, proc_embs)`` split back from
        the joint GCN tensor

    Same as ``DecimaTruePolicy``:
      - one-hot task/proc tag concatenated post-projection
      - DAG precedence (undirected) + proc self-loops; no thermal,
        no task-proc affinity edges in the trunk
      - mean-aggregation GCNConv stack

    Note on `hidden_dim`: the GCN trunk input dim is ``hidden_dim + 2``
    (after the type tag concat), and the GCN OUTPUT dim is ``hidden_dim``
    — matching the cross-attention actor's expected input dim without
    a projection layer (per the HK-5.0 design check).
    """

    def __init__(
        self,
        *,
        task_in_dim:    int = _DEFAULT_TASK_IN_DIM,
        proc_in_dim:    int = _DEFAULT_PROC_IN_DIM,
        hidden_dim:     int = 128,
        num_gcn_layers: int = 4,
    ):
        super().__init__()
        self.task_in_dim    = int(task_in_dim)
        self.proc_in_dim    = int(proc_in_dim)
        self.hidden_dim     = int(hidden_dim)
        self.num_gcn_layers = int(num_gcn_layers)

        self.task_proj = nn.Linear(task_in_dim, hidden_dim)
        self.proc_proj = nn.Linear(proc_in_dim, hidden_dim)

        layers: List[GCNConv] = []
        in_dim = hidden_dim + 2   # + 2 one-hot type bits
        for _ in range(num_gcn_layers):
            conv = GCNConv(in_dim, hidden_dim, aggr="mean")
            layers.append(conv)
            in_dim = hidden_dim
        self.gcn_layers = nn.ModuleList(layers)

    # -----------------------------------------------------------------
    # Single-graph forward
    # -----------------------------------------------------------------
    def forward(
        self,
        graph_obs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(task_embs[T, H], proc_embs[P, H])`` for one graph.

        Defensive fallbacks:
          - empty task_x: returns ``(zeros(0,H), proc_embs)``
          - missing proc_x: zero-fill at ``proc_in_dim``
        """
        device = next(self.parameters()).device
        task_x, proc_x, edge_index, T, P = self._graph_obs_to_tensors(
            graph_obs, device)
        return self._forward_from_tensors(task_x, proc_x, edge_index, T, P)

    def _forward_from_tensors(
        self,
        task_x:     torch.Tensor,
        proc_x:     torch.Tensor,
        edge_index: torch.Tensor,
        T:          int,
        P:          int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stateless GCN forward; shared by single-graph and batched paths.

        Inputs are PRE-projection raw features; this method does the
        projection + type-tag + GCN stack so the batched path can merge
        graph features and edge offsets BEFORE projection — saving one
        Linear call per env per step.
        """
        device = next(self.parameters()).device
        if T == 0 and P == 0:
            return (
                torch.zeros((0, self.hidden_dim), device=device),
                torch.zeros((0, self.hidden_dim), device=device),
            )

        # Per-type projection
        task_h = self.task_proj(task_x) if T > 0 else \
                  torch.zeros((0, self.hidden_dim), device=device)
        proc_h = self.proc_proj(proc_x) if P > 0 else \
                  torch.zeros((0, self.hidden_dim), device=device)

        # One-hot type tag (task=[1,0], proc=[0,1])
        task_tag = torch.cat(
            [torch.ones (T, 1, device=device),
             torch.zeros(T, 1, device=device)], dim=1) if T > 0 else \
            torch.zeros((0, 2), device=device)
        proc_tag = torch.cat(
            [torch.zeros(P, 1, device=device),
             torch.ones (P, 1, device=device)], dim=1) if P > 0 else \
            torch.zeros((0, 2), device=device)
        task_h = torch.cat([task_h, task_tag], dim=1)   # (T, H+2)
        proc_h = torch.cat([proc_h, proc_tag], dim=1)   # (P, H+2)

        # Joint node tensor: tasks then procs (matches index layout in
        # edge_index)
        x = torch.cat([task_h, proc_h], dim=0)          # (T+P, H+2)

        h = x
        for conv in self.gcn_layers:
            h = F.relu(conv(h, edge_index))

        task_embs = h[:T]                               # (T, H)
        proc_embs = h[T:T + P]                          # (P, H)
        return task_embs, proc_embs

    # -----------------------------------------------------------------
    # Adapter — graph_obs dict to tensors (+ T, P counts)
    # -----------------------------------------------------------------
    def _graph_obs_to_tensors(
        self,
        graph_obs: Dict[str, Any],
        device:    torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Convert one graph_obs dict to (task_x, proc_x, edge_index, T, P).

        Edge index covers: DAG precedence (undirected) + proc self-loops.
        Task indices are 0..T-1; proc indices are T..T+P-1 (Mao-style
        joint indexing).
        """
        task_x_raw = np.asarray(graph_obs.get("task_x", []), dtype=np.float32)
        proc_x_raw = np.asarray(graph_obs.get("proc_x", []), dtype=np.float32)

        if task_x_raw.size == 0 or task_x_raw.ndim != 2:
            task_x = torch.zeros((0, self.task_in_dim),
                                  dtype=torch.float32, device=device)
            T = 0
        else:
            task_x = torch.from_numpy(task_x_raw).to(device)
            T = int(task_x.shape[0])

        if proc_x_raw.size == 0 or proc_x_raw.ndim != 2:
            proc_x = torch.zeros((0, self.proc_in_dim),
                                  dtype=torch.float32, device=device)
            P = 0
        else:
            proc_x = torch.from_numpy(proc_x_raw).to(device)
            P = int(proc_x.shape[0])

        edges_t2t = graph_obs.get("edges_t2t", []) or []
        edge_list: List[Tuple[int, int]] = []
        for u, v in edges_t2t:
            u_i, v_i = int(u), int(v)
            if 0 <= u_i < T and 0 <= v_i < T:
                edge_list.append((u_i, v_i))
                edge_list.append((v_i, u_i))    # undirected DAG message-passing
        for p in range(P):
            edge_list.append((T + p, T + p))    # proc self-loops

        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long,
                                       device=device).t().contiguous()
        return task_x, proc_x, edge_index, T, P


# =====================================================================
# Actor-Critic — encoder + cross-attention actor + single critic
# =====================================================================
class DecimaXAttnActorCritic(nn.Module):
    """PPO actor-critic for the D2 baseline.

    Single-graph contract (act / forward / get_value) mirrors HGATE-PPO
    so the eval-time scheduler is interchangeable.  Batched contract
    (act_batched / forward_batched / evaluate_actions_batched /
    get_value_batched) mirrors HGATE-PPO so the trainer is a copy.
    """

    def __init__(
        self,
        *,
        hidden_dim:     int = 128,
        num_procs:      int = 17,
        num_heads:      int = 4,
        num_gcn_layers: int = 4,
        task_in_dim:    int = _DEFAULT_TASK_IN_DIM,
        proc_in_dim:    int = _DEFAULT_PROC_IN_DIM,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads}) — cross-attention head_dim contract."
            )
        self.hidden_dim     = int(hidden_dim)
        self.num_procs      = int(num_procs)
        self.num_heads      = int(num_heads)
        self.num_gcn_layers = int(num_gcn_layers)

        self.encoder = DecimaXAttnEncoder(
            task_in_dim    = task_in_dim,
            proc_in_dim    = proc_in_dim,
            hidden_dim     = hidden_dim,
            num_gcn_layers = num_gcn_layers,
        )

        # K_delay=1 — auto_only mode; the delay head exists for
        # architectural reuse but is sampled trivially and never
        # consulted by the trainer.
        self.actor = CrossAttentionActor(
            hidden    = hidden_dim,
            K_delay   = 1,
            num_heads = num_heads,
        )

        # Single scalar critic on the mean-pooled global context
        # (same shape as HGATE-PPO's critic — keeps the §5 critic
        # architecture variable held constant between D2 and HGATE).
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # -----------------------------------------------------------------
    # Single-graph forward + act + get_value
    # -----------------------------------------------------------------
    def forward(
        self,
        graph_obs:   Dict[str, Any],
        action_mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits[P], value[scalar])``."""
        device = next(self.parameters()).device
        task_embs, proc_embs = self.encoder(graph_obs)
        P = int(np.asarray(action_mask).shape[0])

        # Cross-attention actor expects batched form; wrap as B=1
        if proc_embs.shape[0] == 0:
            logits = torch.zeros(P, device=device)
        else:
            cur_idx = int(graph_obs.get("current_task_idx", 0))
            T = int(task_embs.shape[0])
            cur_idx = max(0, min(cur_idx, max(T - 1, 0)))

            batch_task = torch.zeros(T, dtype=torch.long, device=device)
            batch_proc = torch.zeros(proc_embs.shape[0],
                                      dtype=torch.long, device=device)
            mask_t = torch.tensor(np.asarray(action_mask, dtype=bool),
                                   dtype=torch.bool, device=device).unsqueeze(0)
            cur_t  = torch.tensor([cur_idx], dtype=torch.long, device=device)

            out = self.actor(
                x_task      = task_embs,
                x_proc      = proc_embs,
                batch_task  = batch_task,
                batch_proc  = batch_proc,
                current_idx_within_graph = cur_t,
                action_mask = mask_t,
            )
            placement = out["placement_logits"].squeeze(0)    # (P,)
            # The actor already masks invalid positions to -1e9; we
            # also clamp shape to len(action_mask) (P) in case the
            # actor padded.
            logits = placement[:P]

        # Critic on mean-pooled global context (task ∪ proc)
        all_embs = torch.cat([task_embs, proc_embs], dim=0)
        if all_embs.shape[0] == 0:
            global_ctx = torch.zeros(self.hidden_dim, device=device)
        else:
            global_ctx = all_embs.mean(dim=0)
        value = self.critic(global_ctx).squeeze(-1)
        return logits, value

    def act(
        self,
        graph_obs:    Dict[str, Any],
        action_mask:  np.ndarray,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample / argmax a processor index from the masked softmax."""
        action_mask_np = np.asarray(action_mask, dtype=bool)
        if not action_mask_np.any():
            raise RuntimeError(
                "DecimaXAttnActorCritic.act: action_mask is all False — "
                "no valid procs")
        logits, value = self.forward(graph_obs, action_mask_np)
        device = logits.device
        mask_t = torch.tensor(action_mask_np, dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask_t, float("-inf"))

        if deterministic:
            action_t = torch.argmax(masked_logits)
        else:
            action_t = torch.distributions.Categorical(
                logits=masked_logits).sample()
        action_int = int(action_t.item())

        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs_all     = log_probs_all.exp()
        log_prob = log_probs_all[action_int]
        entropy  = -(probs_all[mask_t] * log_probs_all[mask_t]).sum()

        return {
            "action":   action_int,
            "log_prob": log_prob,
            "entropy":  entropy,
            "value":    value,
        }

    def get_value(self, graph_obs: Dict[str, Any]) -> torch.Tensor:
        """Scalar V(s); used by the GAE bootstrap.  Skips actor scoring."""
        device = next(self.parameters()).device
        task_embs, proc_embs = self.encoder(graph_obs)
        all_embs = torch.cat([task_embs, proc_embs], dim=0)
        if all_embs.shape[0] == 0:
            global_ctx = torch.zeros(self.hidden_dim, device=device)
        else:
            global_ctx = all_embs.mean(dim=0)
        return self.critic(global_ctx).squeeze(-1)

    # =================================================================
    # Batched API — multi-env rollout + minibatched PPO update
    # =================================================================
    def _merge_graph_obs(
        self,
        graph_obs_list: List[Dict[str, Any]],
        device:         torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, int, int]:
        """Merge N graph_obs dicts into one big homogeneous graph.

        Returns
        -------
        task_x_b   : (sum_T_i, task_in_dim)
        proc_x_b   : (N*P,     proc_in_dim)
        edge_index : (2, sum_E) — joint task+proc indices, offset per graph
        batch_task : (sum_T_i,) — per-task graph index in [0, N)
        batch_proc : (N*P,)     — per-proc graph index in [0, N)
        cur_idx_within : (N,)   — clamped current_task_idx per graph
        P_ref      : int        — uniform proc count
        T_total    : int        — sum of T_i

        Edge offsets follow the same joint-indexing convention as the
        single-graph forward: per graph i, tasks are 0..T_i-1 and procs
        are T_i..T_i+P_ref-1 BEFORE offsetting.  After merging, all
        tasks of all graphs come first (offset by cumulative T), then
        all procs (offset by total_T + i * P_ref).
        """
        N = len(graph_obs_list)
        task_xs:        List[torch.Tensor] = []
        proc_xs:        List[torch.Tensor] = []
        edge_chunks:    List[torch.Tensor] = []
        batch_t_chunks: List[torch.Tensor] = []
        cur_idx_list:   List[int]          = []

        task_offset = 0
        P_ref:       int = -1
        T_offsets:   List[int] = []

        # First pass: collect per-graph raw tensors + figure out T_total / P_ref
        per_graph = []
        for i, go in enumerate(graph_obs_list):
            tx, px, _e_unused, T_i, P_i = self.encoder._graph_obs_to_tensors(
                go, device)
            if P_ref < 0:
                P_ref = P_i
            elif P_i != P_ref:
                raise RuntimeError(
                    f"_merge_graph_obs: graph {i} has {P_i} procs but graph 0 "
                    f"has {P_ref}.  Batched forward requires uniform proc count."
                )
            per_graph.append((tx, px, go, T_i))
            T_offsets.append(task_offset)
            task_offset += T_i
        T_total = task_offset
        if P_ref < 0:
            P_ref = 0

        # Second pass: build edge_index in joint task+proc index space.
        # In the MERGED graph, all task nodes come first (indices
        # 0..T_total-1) then all proc nodes (T_total..T_total+N*P_ref-1).
        for i, (tx, px, go, T_i) in enumerate(per_graph):
            t_off = T_offsets[i]
            p_off_in_merged = T_total + i * P_ref   # absolute proc-index base

            edges_t2t = go.get("edges_t2t", []) or []
            for u, v in edges_t2t:
                u_i, v_i = int(u), int(v)
                if 0 <= u_i < T_i and 0 <= v_i < T_i:
                    edge_chunks.append(torch.tensor(
                        [[u_i + t_off, v_i + t_off],
                         [v_i + t_off, u_i + t_off]],
                        dtype=torch.long, device=device).t().contiguous())
            # proc self-loops (in merged space)
            for p in range(P_ref):
                idx = p_off_in_merged + p
                edge_chunks.append(torch.tensor(
                    [[idx], [idx]], dtype=torch.long, device=device))

            task_xs.append(tx)
            proc_xs.append(px)
            batch_t_chunks.append(torch.full((T_i,), i, dtype=torch.long,
                                              device=device))
            # Clamp current_task_idx into [0, T_i-1] (defensive)
            cur = int(go.get("current_task_idx", 0))
            cur = max(0, min(cur, max(T_i - 1, 0)))
            cur_idx_list.append(cur)

        task_x_b = (torch.cat(task_xs, dim=0) if task_xs
                     else torch.zeros((0, self.encoder.task_in_dim),
                                       dtype=torch.float32, device=device))
        proc_x_b = (torch.cat(proc_xs, dim=0) if proc_xs
                     else torch.zeros((0, self.encoder.proc_in_dim),
                                       dtype=torch.float32, device=device))
        if edge_chunks:
            edge_index = torch.cat(edge_chunks, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        batch_task = (torch.cat(batch_t_chunks, dim=0) if batch_t_chunks
                       else torch.zeros((0,), dtype=torch.long, device=device))
        batch_proc = torch.arange(N, device=device).repeat_interleave(P_ref) \
                        if (N > 0 and P_ref > 0) else \
                        torch.zeros((0,), dtype=torch.long, device=device)
        cur_idx_within = torch.tensor(cur_idx_list, dtype=torch.long,
                                       device=device)
        return (task_x_b, proc_x_b, edge_index,
                batch_task, batch_proc, cur_idx_within,
                None, T_total, P_ref)  # placeholder slot kept for API symmetry

    def forward_batched(
        self,
        graph_obs_list: List[Dict[str, Any]],
        action_mask_list: List[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """Batched (logits, values) over N graphs in one forward pass.

        ``action_mask_list`` is optional — if None (e.g. for the
        get_value bootstrap path), an all-True mask is used.  The
        cross-attention actor needs the mask to compute placement
        logits anyway, so we always materialise one.
        """
        device = next(self.parameters()).device
        N = len(graph_obs_list)
        if N == 0:
            return {
                "logits": torch.zeros((0, self.num_procs), device=device),
                "values": torch.zeros((0,),                device=device),
            }
        H = self.hidden_dim

        (task_x_b, proc_x_b, edge_index,
         batch_task, batch_proc, cur_idx_within,
         _unused, T_total, P_ref) = self._merge_graph_obs(
            graph_obs_list, device)

        # Encoder forward over the merged graph
        task_embs, proc_embs = self.encoder._forward_from_tensors(
            task_x_b, proc_x_b, edge_index, T_total, N * P_ref)

        # Build per-graph action mask tensor (N, P_ref).  If caller
        # didn't supply one, use all-True (used by the get_value
        # bootstrap path where we discard logits).
        if action_mask_list is None:
            mask_t = torch.ones((N, P_ref), dtype=torch.bool, device=device)
        else:
            if P_ref == 0:
                mask_t = torch.zeros((N, 0), dtype=torch.bool, device=device)
            else:
                mask_np = np.stack(
                    [np.asarray(m, dtype=bool)[:P_ref] for m in action_mask_list],
                    axis=0)
                mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

        # Cross-attention placement logits
        if P_ref == 0:
            logits = torch.zeros((N, self.num_procs), device=device)
        else:
            actor_out = self.actor(
                x_task      = task_embs,
                x_proc      = proc_embs,
                batch_task  = batch_task,
                batch_proc  = batch_proc,
                current_idx_within_graph = cur_idx_within,
                action_mask = mask_t,
            )
            logits = actor_out["placement_logits"]    # (N, P_ref)

        # Per-graph mean-pooled global context for the critic.  Same
        # semantics as HGATE-PPO's batched path: scatter-mean over the
        # union of task and proc embeddings, grouped by per-node graph
        # membership.
        global_ctx = torch.zeros((N, H), device=device)
        counts     = torch.zeros((N,),  device=device)
        if task_embs.shape[0] > 0:
            global_ctx.index_add_(0, batch_task, task_embs)
            counts    .index_add_(0, batch_task,
                                   torch.ones_like(batch_task, dtype=torch.float32))
        if proc_embs.shape[0] > 0:
            global_ctx.index_add_(0, batch_proc, proc_embs)
            counts    .index_add_(0, batch_proc,
                                   torch.ones_like(batch_proc, dtype=torch.float32))
        counts = counts.clamp(min=1.0).unsqueeze(-1)
        global_ctx = global_ctx / counts

        values = self.critic(global_ctx).squeeze(-1)
        return {"logits": logits, "values": values}

    def act_batched(
        self,
        graph_obs_list:   List[Dict[str, Any]],
        action_mask_list: List[np.ndarray],
        deterministic:    bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Batched sampling — N graphs in, action / log_prob / entropy / value tensors out."""
        out = self.forward_batched(graph_obs_list, action_mask_list)
        logits = out["logits"]
        values = out["values"]
        device = logits.device
        N, P   = int(logits.shape[0]), int(logits.shape[1])

        mask_np = np.stack(
            [np.asarray(m, dtype=bool)[:P] for m in action_mask_list], axis=0)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        if not mask_t.any(dim=-1).all():
            bad = (~mask_t.any(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"act_batched: action_mask is all False for envs {bad} — "
                f"no valid procs")

        # The actor already applies -1e9 to masked positions; re-apply
        # -inf here for the log-softmax / Categorical contract.  This
        # double-masking is benign (-1e9 already gives ~0 softmax mass)
        # but the -inf form keeps the entropy reduction exact.
        masked_logits = logits.masked_fill(~mask_t, float("-inf"))
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs_all     = log_probs_all.exp()

        if deterministic:
            actions = masked_logits.argmax(dim=-1)
        else:
            actions = torch.distributions.Categorical(
                logits=masked_logits).sample()

        log_probs = log_probs_all.gather(
            1, actions.unsqueeze(-1)).squeeze(-1)
        zero = torch.zeros_like(log_probs_all)
        lp_v = torch.where(mask_t, log_probs_all, zero)
        p_v  = torch.where(mask_t, probs_all,     zero)
        entropies = -(p_v * lp_v).sum(dim=-1)

        return {
            "actions":   actions,
            "log_probs": log_probs,
            "entropies": entropies,
            "values":    values,
        }

    def evaluate_actions_batched(
        self,
        graph_obs_list:   List[Dict[str, Any]],
        action_mask_list: List[np.ndarray],
        actions:          Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under the current policy.  Used inside
        the PPO minibatch update."""
        out = self.forward_batched(graph_obs_list, action_mask_list)
        logits = out["logits"]
        values = out["values"]
        device = logits.device
        N, P   = int(logits.shape[0]), int(logits.shape[1])

        mask_np = np.stack(
            [np.asarray(m, dtype=bool)[:P] for m in action_mask_list], axis=0)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask_t, float("-inf"))
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs_all     = log_probs_all.exp()

        if isinstance(actions, torch.Tensor):
            actions_t = actions.to(device=device, dtype=torch.long)
        else:
            actions_t = torch.as_tensor(
                np.asarray(actions, dtype=np.int64),
                dtype=torch.long, device=device)

        new_log_probs = log_probs_all.gather(
            1, actions_t.unsqueeze(-1)).squeeze(-1)
        zero = torch.zeros_like(log_probs_all)
        lp_v = torch.where(mask_t, log_probs_all, zero)
        p_v  = torch.where(mask_t, probs_all,     zero)
        entropies = -(p_v * lp_v).sum(dim=-1)
        return new_log_probs, entropies, values

    def get_value_batched(
        self,
        graph_obs_list: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Batched V(s); same encoder + global-pool + critic path as the
        N-fold sequential ``get_value`` calls but in one forward."""
        return self.forward_batched(graph_obs_list)["values"]


# =====================================================================
# Eval-time scheduler
# =====================================================================
class DecimaXAttnScheduler(BaseScheduler):
    """Inference-only wrapper around a trained ``DecimaXAttnActorCritic``."""

    name = "D2"   # paper-facing name (figure / table label)

    def __init__(
        self,
        ckpt_path:    str,
        num_nodes:    int  = 17,
        deterministic: bool = True,
        device:       str  = "cpu",
        # Architecture hyper-params; must match training-time values
        hidden_dim:     int = 128,
        num_gcn_layers: int = 4,
        num_heads:      int = 4,
        task_in_dim:    int = _DEFAULT_TASK_IN_DIM,
        proc_in_dim:    int = _DEFAULT_PROC_IN_DIM,
    ):
        super().__init__(action_mode="auto_only", K_delay=5)
        self.num_nodes     = int(num_nodes)
        self.deterministic = bool(deterministic)
        self.device        = device

        self._model = DecimaXAttnActorCritic(
            hidden_dim     = hidden_dim,
            num_procs      = num_nodes,
            num_heads      = num_heads,
            num_gcn_layers = num_gcn_layers,
            task_in_dim    = task_in_dim,
            proc_in_dim    = proc_in_dim,
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
        pass    # stateless (the policy is purely reactive)

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        graph_obs = info["graph_obs"][0] if isinstance(
            info["graph_obs"], (list, np.ndarray)
        ) else info["graph_obs"]
        action_mask = np.asarray(info["action_mask"], dtype=bool)

        with torch.no_grad():
            out = self._model.act(
                graph_obs, action_mask,
                deterministic=self.deterministic,
            )
        return self._wrap_action(int(out["action"]))
