"""
hetero_encoder.py
=================

Stage C — Heterogeneous Graph Encoder for the v2 CPO scheduler.

Implements §4.1 of the reconstruction plan: a **dual-graph GNN** that
processes the task DAG and the processor topology in parallel, then
fuses them via cross-graph attention.  All three sub-graphs carry edge
attributes whose dimensions match what ``cpo_thermal_env.py`` produces:

    task → task   :  edge_dim = 1   (normalised traffic)
    proc → proc   :  edge_dim = 1   (RC-matrix coupling A_ij)
    task → proc   :  edge_dim = 2   (estimated exec_time, estimated ΔT)

Why this design (vs the v1 single-GAT-on-bipartite encoder)
-----------------------------------------------------------
1. **Physics-aligned message passing**: the proc-proc graph is built
   directly from the discrete-time RC matrix A, so the GNN's lateral
   thermal reasoning is anchored in the actual heat-conduction graph,
   not a hand-engineered topology.
2. **Edge attributes in every layer**: GATv2Conv with ``edge_dim>0``
   modulates attention by physical signals (traffic / heat-conductance
   / time-cost), preventing the encoder from collapsing to "ASIC is
   special" patterns that fail when N changes.
3. **Parametric in num_proc**: the encoder reads ``proc_x.shape[0]``
   from the input batch — no compile-time knowledge of N.  This is what
   lets the same trained encoder zero-shot to N ∈ {9, 13, 17, 24, 33}.

Input format
------------
The encoder takes a PyG ``HeteroData`` (or a batched version) with three
node types (``task``, ``proc``) and three edge types::

    data['task'].x         -> (num_tasks, 8)
    data['proc'].x         -> (num_proc,  7)
    data['task','dep','task'].edge_index, .edge_attr  -> (2, |E_t2t|), (|E_t2t|, 1)
    data['proc','therm','proc'].edge_index, .edge_attr -> (2, |E_p2p|), (|E_p2p|, 1)
    data['task','place','proc'].edge_index, .edge_attr -> (2, |E_t2p|), (|E_t2p|, 2)

Output
------
``encode(data)`` returns ``(x_task, x_proc, batch_task, batch_proc)`` where
``x_task`` and ``x_proc`` are post-cross-attention embeddings of dim ``hidden``.

"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports — present on the A100 server.  This file is intentionally
# torch_geometric-only (no fall-back), since the encoder is the core of
# the model and isn't expected to run on CPU-only sandboxes.
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData

# Edge type tuples (PyG stores them as tuples)
ET_T2T = ("task", "dep",   "task")
ET_P2P = ("proc", "therm", "proc")
ET_T2P = ("task", "place", "proc")


# =====================================================================
# Sub-modules
# =====================================================================
class GraphLayer(nn.Module):
    """One round of edge-attribute-aware GAT message passing on a single
    relation.  Wraps :class:`GATv2Conv` with residual + LayerNorm."""

    def __init__(
        self,
        in_dim_src: int,
        in_dim_dst: int,
        out_dim:    int,
        edge_dim:   int,
        heads:      int = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        # add_self_loops=False because PyG's heterogeneous self-loops
        # require ``in_dim_src == in_dim_dst``, which is not the case for
        # the cross-graph relation.  We add explicit residual connections
        # below to preserve self-information.
        self.gat = GATv2Conv(
            in_channels=(in_dim_src, in_dim_dst),
            out_channels=out_dim,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
            concat=False,           # average heads -> out_dim (not heads*out_dim)
        )
        # Residual projection (only if dst dim differs from out_dim)
        self.residual = (
            nn.Identity() if in_dim_dst == out_dim
            else nn.Linear(in_dim_dst, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x_src:      torch.Tensor,
        x_dst:      torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            # No edges of this relation in the current batch — bypass GAT.
            # This happens often: e.g. small DAGs with no internal deps,
            # or warm-start episodes with all procs masked out.
            return self.norm(self.residual(x_dst))
        msg = self.gat((x_src, x_dst), edge_index, edge_attr=edge_attr)
        return self.norm(self.residual(x_dst) + msg)


class CrossGraphAttention(nn.Module):
    """One round of bidirectional task↔proc cross-attention.

    Uses :class:`GATv2Conv` on the bipartite ``task → proc`` relation,
    then a second time on the reversed relation, so both task and proc
    embeddings see each other.  Edge attributes (``[est_time/50,
    est_dT/20]``) modulate attention.
    """

    def __init__(self, hidden: int, edge_dim: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.t2p = GraphLayer(hidden, hidden, hidden, edge_dim,
                              heads=heads, dropout=dropout)
        self.p2t = GraphLayer(hidden, hidden, hidden, edge_dim,
                              heads=heads, dropout=dropout)

    def forward(
        self,
        x_task:    torch.Tensor,
        x_proc:    torch.Tensor,
        ei_t2p:    torch.Tensor,
        ea_t2p:    Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # task -> proc
        x_proc_new = self.t2p(x_task, x_proc, ei_t2p, ea_t2p)
        # proc -> task: reverse edge_index (swap rows)
        if ei_t2p.numel() == 0:
            ei_p2t = ei_t2p
        else:
            ei_p2t = torch.stack([ei_t2p[1], ei_t2p[0]], dim=0)
        x_task_new = self.p2t(x_proc_new, x_task, ei_p2t, ea_t2p)
        return x_task_new, x_proc_new


# =====================================================================
# Top-level encoder
# =====================================================================
class HeteroEncoder(nn.Module):
    """Dual-graph encoder + cross-graph attention.

    Parameters
    ----------
    task_in_dim
        Input task feature dim (default 8, matches ``dag.task_features``).
    proc_in_dim
        Input proc feature dim (default 7, matches env's ``_proc_features``).
    edge_dim_t2t / _p2p / _t2p
        Edge attribute dims, must match what the env produces (1, 1, 2).
    hidden
        Embedding dim (default 128).  Same dim used for task and proc.
    num_layers
        How many rounds of {within-graph + cross-graph} message passing.
        Default 2 (matches plan §4.1).
    heads / dropout
        GAT heads / dropout — standard.
    """

    def __init__(
        self,
        task_in_dim: int = 8,
        proc_in_dim: int = 7,
        edge_dim_t2t: int = 1,
        edge_dim_p2p: int = 1,
        edge_dim_t2p: int = 2,
        hidden:      int = 128,
        num_layers:  int = 2,
        heads:       int = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden

        # Input projections (raw features -> hidden)
        self.task_proj = nn.Sequential(
            nn.Linear(task_in_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.proc_proj = nn.Sequential(
            nn.Linear(proc_in_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
        )

        # Stacked layers: each block does
        #   1. within-graph: task->task + proc->proc
        #   2. cross-graph:  task<->proc (bidirectional)
        self.t2t_layers = nn.ModuleList([
            GraphLayer(hidden, hidden, hidden, edge_dim_t2t,
                       heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.p2p_layers = nn.ModuleList([
            GraphLayer(hidden, hidden, hidden, edge_dim_p2p,
                       heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.cross_layers = nn.ModuleList([
            CrossGraphAttention(hidden, edge_dim=edge_dim_t2p,
                                heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    # -----------------------------------------------------------------
    def forward(
        self,
        data: HeteroData,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one HeteroData (single sample or batched).

        Returns
        -------
        x_task : (num_tasks_total, hidden)
        x_proc : (num_proc_total,  hidden)
        batch_task : (num_tasks_total,) — graph membership for batched input,
                     all-zeros for unbatched.
        batch_proc : (num_proc_total,)  — same convention.
        """
        # 1. Project raw features
        x_task = self.task_proj(data["task"].x)
        x_proc = self.proc_proj(data["proc"].x)

        # 2. Pull edges
        ei_t2t = data[ET_T2T].edge_index
        ea_t2t = data[ET_T2T].edge_attr if "edge_attr" in data[ET_T2T] else None
        ei_p2p = data[ET_P2P].edge_index
        ea_p2p = data[ET_P2P].edge_attr if "edge_attr" in data[ET_P2P] else None
        ei_t2p = data[ET_T2P].edge_index
        ea_t2p = data[ET_T2P].edge_attr if "edge_attr" in data[ET_T2P] else None

        # 3. Stacked message passing
        for t2t, p2p, cross in zip(
            self.t2t_layers, self.p2p_layers, self.cross_layers,
        ):
            # within-graph
            x_task = t2t(x_task, x_task, ei_t2t, ea_t2t)
            x_proc = p2p(x_proc, x_proc, ei_p2p, ea_p2p)
            # cross-graph
            x_task, x_proc = cross(x_task, x_proc, ei_t2p, ea_t2p)

        # 4. Batch indices for downstream pooling.  In an unbatched single
        # graph, both batch tensors are all zeros.  PyG's Batch.from_data_list
        # populates these as data['task'].batch / data['proc'].batch.
        batch_task = (
            data["task"].batch if "batch" in data["task"]
            else torch.zeros(x_task.size(0), dtype=torch.long, device=x_task.device)
        )
        batch_proc = (
            data["proc"].batch if "batch" in data["proc"]
            else torch.zeros(x_proc.size(0), dtype=torch.long, device=x_proc.device)
        )
        return x_task, x_proc, batch_task, batch_proc


# =====================================================================
# graph_obs (dict from env) -> HeteroData converter
# =====================================================================
def graph_obs_to_hetero_data(
    graph_obs: dict,
    device: torch.device | str = "cpu",
    edge_dim_t2p: int = 2,
    task_in_dim:  int = 8,
    proc_in_dim:  int = 7,
) -> HeteroData:
    """Convert one ``info['graph_obs'][0]`` dict to a PyG :class:`HeteroData`.

    The env emits plain Python lists (for AsyncVectorEnv pickling); here we
    materialise them as torch tensors on the requested device.

    Parameters
    ----------
    edge_dim_t2p : int
        Expected dim of task→proc edge attrs.  Default 2 ([est_time,
        est_rise]); pass 1 for fair-Decima where est_rise has been
        stripped.  This matters for the empty-edge fast path which
        creates a zeros tensor of shape (0, edge_dim_t2p) — wrong dim
        causes shape-mismatch errors when batched with non-empty cells.
    task_in_dim : int
        Expected task-feature dim.  Default 8 (matches
        ``dag_parser.task_features`` + Ours full schema).  Used by the
        empty-task fast path which creates a zeros tensor of shape
        (0, task_in_dim) when ``graph_obs["task_x"]`` is empty (e.g.
        the brief moment between DAG completion and the next DAG load).
        Wrong dim causes (1, 0) vs (8, 128) matmul errors at encoder
        forward when PyG batches the empty cell with non-empty siblings.
    proc_in_dim : int
        Expected proc-feature dim.  Default 7 (Ours full); pass 3 for
        Ours-NoThermal (thermal cols dropped).  Same empty-list
        fast-path rationale as ``task_in_dim`` and ``edge_dim_t2p``.

    Notes
    -----
    * If a relation has no edges in this step (e.g. a leaf-node DAG with
      no t2t edges), we still create an empty (2, 0) edge_index and a
      (0, edge_dim) edge_attr — that way the encoder layers still see
      consistent shapes and the empty-edge fast path triggers.
    * ``current_task_idx`` is preserved on ``data['task'].current_idx`` so
      the actor's cross-attention head can grab it after batching.
    """
    data = HeteroData()
    # Node features (with empty-list fallback mirroring the edge path
    # below).  graph_obs["task_x"] / ["proc_x"] can be [] briefly
    # between DAG completion and reload, producing torch.tensor([]) of
    # shape (0,) rather than (0, dim) — downstream PyG
    # Batch.from_data_list collapses to (1, 0), then encoder's
    # nn.Linear(dim, hidden) raises shape-mismatch.
    task_x_list = graph_obs["task_x"]
    if len(task_x_list) == 0:
        data["task"].x = torch.zeros((0, task_in_dim),
                                     dtype=torch.float32, device=device)
    else:
        data["task"].x = torch.tensor(task_x_list,
                                      dtype=torch.float32, device=device)

    proc_x_list = graph_obs["proc_x"]
    if len(proc_x_list) == 0:
        data["proc"].x = torch.zeros((0, proc_in_dim),
                                     dtype=torch.float32, device=device)
    else:
        data["proc"].x = torch.tensor(proc_x_list,
                                      dtype=torch.float32, device=device)
    # Optional: keep the index of the task being scheduled this step
    if "current_task_idx" in graph_obs:
        data["task"].current_idx = torch.tensor(
            [int(graph_obs["current_task_idx"])],
            dtype=torch.long, device=device,
        )
    elif "ready_idx" in graph_obs and len(graph_obs["ready_idx"]) > 0:
        # Fall back to first ready task (matches FIFO popping in env)
        data["task"].current_idx = torch.tensor(
            [int(graph_obs["ready_idx"][0])],
            dtype=torch.long, device=device,
        )

    # Edges
    def _edges(key_idx: str, key_attr: str, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = graph_obs.get(key_idx, [])
        attr = graph_obs.get(key_attr, [])
        if len(idx) == 0:
            ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            ea = torch.zeros((0, dim), dtype=torch.float32, device=device)
        else:
            # idx is a list of [u, v]; transpose to (2, |E|)
            ei = torch.tensor(idx, dtype=torch.long, device=device).t().contiguous()
            ea = torch.tensor(attr, dtype=torch.float32, device=device)
            if ea.dim() == 1:
                ea = ea.unsqueeze(-1)
        return ei, ea

    ei, ea = _edges("edges_t2t", "edges_t2t_attr", 1)
    data[ET_T2T].edge_index = ei
    data[ET_T2T].edge_attr  = ea
    ei, ea = _edges("edges_p2p", "edges_p2p_attr", 1)
    data[ET_P2P].edge_index = ei
    data[ET_P2P].edge_attr  = ea
    ei, ea = _edges("edges_t2p", "edges_t2p_attr", edge_dim_t2p)
    data[ET_T2P].edge_index = ei
    data[ET_T2P].edge_attr  = ea
    return data
