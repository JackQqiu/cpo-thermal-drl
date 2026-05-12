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
        return self._forward_from_tensors(
            task_x, proc_x, edge_t2t, edge_t2p, edge_t2p_attr)

    def _forward_from_tensors(
        self,
        task_x:        torch.Tensor,
        proc_x:        torch.Tensor,
        edge_t2t:      torch.Tensor,
        edge_t2p:      torch.Tensor,
        edge_t2p_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stateless encoder forward over pre-merged tensors.

        Used by both single-graph ``forward()`` and the batched path in
        ``HGATEActorCritic.forward_batched``.  Keeping the message-passing
        logic in one place guarantees the two callers stay semantically
        identical (single-graph ⇄ N-graph merged) so the bit-identical
        unit tests catch any future drift.
        """
        h_task = self.task_proj(task_x)        # (T, H)
        h_proc = self.proc_proj(proc_x)        # (P, H)

        for i in range(self.num_layers):
            # task <- task (DAG precedence); self-loops cover isolated nodes
            h_task_new = self.t2t_convs[i](h_task, edge_t2t)
            # proc <- task (sched affinity) with est_time edge_attr.
            # Source (h_task) is intentionally the PRE-update embedding —
            # update order matches the in-paper formulation; do not
            # swap to h_task_new without re-running bit-identical tests.
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

    # =================================================================
    # Step 5 (HK-4.5.2) — batched API
    # =================================================================
    # The single-graph forward / act / get_value above are kept as-is
    # for the eval-time scheduler (which sees one graph per env step)
    # and for the unit tests that established correctness in HK-4.1..3.
    #
    # The training loop uses the batched variants below: per env-step
    # they merge N graph_obs into a single big graph (concatenating
    # node features with task/proc offsets on edges), run ONE encoder
    # forward over the merged graph, then split outputs back per graph.
    # This is the standard PyG batching pattern; the bit-identical
    # unit tests in test_hgate_ppo.py Step 5 guard against drift.
    # =================================================================
    def _merge_graph_obs(
        self,
        graph_obs_list: List[Dict[str, Any]],
        device:         torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge N graph_obs into one big hetero-graph.

        Assumes ``proc_x`` has the same row count ``P`` across all N
        graphs (true for CPOThermalDAGEnvV2 — num_procs is a fixed env
        config).  Task counts ``T_i`` are allowed to vary; ``batch_t``
        tracks per-task graph membership for the per-graph mean pool.

        HK-4.5.4 note
        -------------
        HK-4.5.3 tried a CPU-concat + 6-H→D-transfer variant betting on
        host→device kernel-launch overhead being the bottleneck.
        Microbenchmark on Mac CPU (which has no H→D cost) showed that
        variant was actually 5-10% SLOWER at realistic graph sizes
        (T≈30, P=17), and the V100 pilot confirmed a regression — the
        H→D copies on V100 weren't 80× the per-call overhead I'd
        assumed (driver batches small transfers into shared PCIe
        transactions).  Reverted to the HK-4.5.2 path below.

        Returns
        -------
        task_x_b      : (sum_T_i, task_in_dim)
        proc_x_b      : (N*P,    proc_in_dim)
        edge_t2t_b    : (2, sum_E_t2t) — task indices already offset
        edge_t2p_b    : (2, sum_E_t2p) — row 0 task-offset, row 1 proc-offset
        edge_t2p_attr : (sum_E_t2p, 1)
        batch_t       : (sum_T_i,) — per-task graph index in [0, N)
        """
        task_xs:        List[torch.Tensor] = []
        proc_xs:        List[torch.Tensor] = []
        edge_t2ts:      List[torch.Tensor] = []
        edge_t2ps:      List[torch.Tensor] = []
        edge_t2p_attrs: List[torch.Tensor] = []
        batch_t_list:   List[torch.Tensor] = []

        task_offset = 0
        P_ref:       int = -1
        for i, go in enumerate(graph_obs_list):
            tx, px, et2t, et2p, et2p_attr = \
                self.encoder._graph_obs_to_hetero(go, device)
            T_i = int(tx.shape[0])
            P_i = int(px.shape[0])
            if P_ref < 0:
                P_ref = P_i
            elif P_i != P_ref:
                raise RuntimeError(
                    f"_merge_graph_obs: graph {i} has {P_i} procs but graph 0 "
                    f"has {P_ref}.  Batched forward requires uniform proc count."
                )
            proc_offset = i * P_ref

            task_xs.append(tx)
            proc_xs.append(px)
            if et2t.numel() > 0:
                edge_t2ts.append(et2t + task_offset)
            if et2p.numel() > 0:
                # row 0 = task source (offset by cumulative task count);
                # row 1 = proc dest  (offset by i * P)
                offset = torch.stack([
                    torch.full((et2p.shape[1],), task_offset,
                                dtype=torch.long, device=device),
                    torch.full((et2p.shape[1],), proc_offset,
                                dtype=torch.long, device=device),
                ])
                edge_t2ps.append(et2p + offset)
                edge_t2p_attrs.append(et2p_attr)
            batch_t_list.append(torch.full(
                (T_i,), i, dtype=torch.long, device=device))
            task_offset += T_i

        task_x_b = (torch.cat(task_xs, dim=0) if task_xs
                     else torch.zeros((0, self.encoder.task_in_dim),
                                       dtype=torch.float32, device=device))
        proc_x_b = (torch.cat(proc_xs, dim=0) if proc_xs
                     else torch.zeros((0, self.encoder.proc_in_dim),
                                       dtype=torch.float32, device=device))
        edge_t2t_b = (torch.cat(edge_t2ts, dim=1) if edge_t2ts
                       else torch.zeros((2, 0), dtype=torch.long, device=device))
        edge_t2p_b = (torch.cat(edge_t2ps, dim=1) if edge_t2ps
                       else torch.zeros((2, 0), dtype=torch.long, device=device))
        edge_t2p_attr_b = (torch.cat(edge_t2p_attrs, dim=0) if edge_t2p_attrs
                            else torch.zeros((0, 1), dtype=torch.float32,
                                              device=device))
        batch_t = (torch.cat(batch_t_list, dim=0) if batch_t_list
                    else torch.zeros((0,), dtype=torch.long, device=device))
        return (task_x_b, proc_x_b, edge_t2t_b, edge_t2p_b,
                edge_t2p_attr_b, batch_t)

    def forward_batched(
        self,
        graph_obs_list: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Run the encoder + actor scorer + critic over N graphs in one pass.

        Returns
        -------
        dict::
            {
                'logits':  (N, P) — per-proc unmasked scores per graph,
                'values':  (N,)   — scalar V(s) per graph.
            }

        Bit-identical to N sequential ``forward()`` calls (within float32
        reduction-order noise ≤ 1e-5).  Guarded by Step-5 unit tests.
        """
        device = next(self.parameters()).device
        N = len(graph_obs_list)
        if N == 0:
            return {
                "logits": torch.zeros((0, self.num_procs), device=device),
                "values": torch.zeros((0,),                device=device),
            }
        H = self.hidden_dim

        task_x_b, proc_x_b, edge_t2t_b, edge_t2p_b, edge_t2p_attr_b, batch_t = \
            self._merge_graph_obs(graph_obs_list, device)

        h_task, h_proc = self.encoder._forward_from_tensors(
            task_x_b, proc_x_b, edge_t2t_b, edge_t2p_b, edge_t2p_attr_b,
        )

        # P is uniform across graphs (validated in _merge_graph_obs).
        # If the env emitted zero procs (shouldn't happen but be safe)
        # we degrade to per-graph zero-logits.
        total_P = int(proc_x_b.shape[0])
        P = total_P // N if N > 0 else 0

        # ----- per-graph global context = mean of (task ∪ proc) embeddings
        # of that graph.  Implemented via two index_add_ reductions to
        # match the single-graph ``forward()`` mean-pool semantics.
        global_ctx = torch.zeros((N, H), device=device)
        counts     = torch.zeros((N,),  device=device)
        if h_task.shape[0] > 0:
            global_ctx.index_add_(0, batch_t, h_task)
            counts    .index_add_(0, batch_t,
                                   torch.ones_like(batch_t, dtype=torch.float32))
        if total_P > 0:
            # batch_p is sequential: 0,0,..,0 (P times), 1,1,..,1, ..., N-1
            batch_p = torch.arange(N, device=device).repeat_interleave(P)
            global_ctx.index_add_(0, batch_p, h_proc)
            counts    .index_add_(0, batch_p,
                                   torch.ones_like(batch_p, dtype=torch.float32))
        counts = counts.clamp(min=1.0).unsqueeze(-1)   # (N, 1)
        global_ctx = global_ctx / counts                # (N, H)

        # ----- per-pair scoring: (N, P, 2H) -> (N, P) via one Linear stack
        if P == 0:
            logits = torch.zeros((N, self.num_procs), device=device)
        else:
            ctx_expanded = global_ctx.unsqueeze(1).expand(-1, P, -1)   # (N, P, H)
            h_proc_grouped = h_proc.view(N, P, H)
            pair_input = torch.cat([ctx_expanded, h_proc_grouped], dim=-1)
            logits = self.actor_score(
                pair_input.view(N * P, 2 * H)).view(N, P)

        # ----- critic
        values = self.critic(global_ctx).squeeze(-1)                  # (N,)

        return {"logits": logits, "values": values}

    def act_batched(
        self,
        graph_obs_list:   List[Dict[str, Any]],
        action_mask_list: List[np.ndarray],
        deterministic:    bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Batched sampling: takes N graphs + N masks, returns stacked
        action / log_prob / entropy / value tensors.

        For ``deterministic=True`` the output is bit-identical to N
        sequential ``act()`` calls.  For ``deterministic=False`` only
        the masking contract is guaranteed (no semantically-equivalent
        single-RNG path exists).
        """
        out = self.forward_batched(graph_obs_list)
        logits = out["logits"]                       # (N, P)
        values = out["values"]                       # (N,)
        device = logits.device
        N, P   = int(logits.shape[0]), int(logits.shape[1])

        mask_np = np.stack(
            [np.asarray(m, dtype=bool) for m in action_mask_list], axis=0,
        )                                            # (N, P)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        if not mask_t.any(dim=-1).all():
            bad = (~mask_t.any(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"act_batched: action_mask is all False for envs {bad} — "
                f"no valid procs")

        masked_logits = logits.masked_fill(~mask_t, float("-inf"))
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        probs_all     = log_probs_all.exp()

        if deterministic:
            actions = masked_logits.argmax(dim=-1)              # (N,)
        else:
            actions = torch.distributions.Categorical(
                logits=masked_logits).sample()                  # (N,)

        log_probs = log_probs_all.gather(
            1, actions.unsqueeze(-1)).squeeze(-1)               # (N,)
        # Entropy: -sum_p (p * log_p) over VALID positions only.
        # Masked-out positions have prob 0 + log_prob -inf, which yields
        # 0 * -inf = NaN; suppress via where().
        zero  = torch.zeros_like(log_probs_all)
        lp_v  = torch.where(mask_t, log_probs_all, zero)
        p_v   = torch.where(mask_t, probs_all,     zero)
        entropies = -(p_v * lp_v).sum(dim=-1)                   # (N,)

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
        """Re-evaluate stored actions under the current policy.

        This is the inner loop of the PPO minibatch update — called once
        per minibatch instead of once per transition.  Returns
        ``(new_log_probs, entropies, values)`` each of shape ``(B,)``.
        """
        out = self.forward_batched(graph_obs_list)
        logits = out["logits"]
        values = out["values"]
        device = logits.device
        N, P   = int(logits.shape[0]), int(logits.shape[1])

        mask_np = np.stack(
            [np.asarray(m, dtype=bool) for m in action_mask_list], axis=0,
        )
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
            1, actions_t.unsqueeze(-1)).squeeze(-1)             # (N,)
        zero = torch.zeros_like(log_probs_all)
        lp_v = torch.where(mask_t, log_probs_all, zero)
        p_v  = torch.where(mask_t, probs_all,     zero)
        entropies = -(p_v * lp_v).sum(dim=-1)                   # (N,)
        return new_log_probs, entropies, values

    def get_value_batched(
        self,
        graph_obs_list: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Batched V(s); same encoder + global-pool + critic path as the
        N-fold sequential ``get_value`` calls but in one forward."""
        # forward_batched does an extra actor-score forward — for the
        # bootstrap path (single call after each rollout) this is cheap
        # relative to the encoder forward, so we accept the small waste
        # in exchange for keeping a single batched code path.
        return self.forward_batched(graph_obs_list)["values"]

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
        task_in_dim:    int = 8,
        proc_in_dim:    int = 7,
    ):
        super().__init__(action_mode="auto_only", K_delay=5)
        self.num_nodes     = int(num_nodes)
        self.deterministic = bool(deterministic)
        self.device        = device

        self._model = HGATEActorCritic(
            hidden_dim     = hidden_dim,
            num_procs      = num_nodes,
            num_heads      = num_heads,
            num_gat_layers = num_gat_layers,
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
        pass    # stateless (Wu 2025 policy is purely reactive)

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
