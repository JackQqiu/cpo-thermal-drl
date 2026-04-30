"""
compute_dag_features.py
=======================
Stage A — Infrastructure: pre-compute extended DAG features for the v2 env.

Reads ``alibaba_dags.json`` (existing format from ``trace_cleaner.py``),
computes additional fields used by the new env and GNN encoder, and dumps
``alibaba_dags_v2.json``.

Per-node fields written
-----------------------
    workload         : float, original (carried through)
    slack            : float, slack time (CPM, same time unit as workload)
    depth            : int,   longest topological depth from any source
    in_degree        : int
    out_degree       : int
    is_critical      : 0/1,   on the critical path (slack < 1e-3)
    rho              : float in [0.1, 5.0], thermal characteristic factor
                       (relative heat density, weighted compute + traffic)
    total_traffic    : float, sum of incoming + outgoing traffic

Per-DAG fields written (top level)
----------------------------------
    trace_id              : str, original
    nodes                 : dict { id: enriched_attrs }
    edges                 : list [[u, v, traffic], ...]
    makespan_lower_bound  : float, critical-path length (best possible makespan)
    median_workload       : float, used for normalisation downstream
    median_traffic        : float, used for normalisation downstream
    num_nodes / num_edges : int
    num_critical          : int

Definition of rho (thermal characteristic factor)
-------------------------------------------------
    rho_i = clip( 0.7 * (workload_i / median_workload)
                + 0.3 * (total_traffic_i / median_traffic),
                  0.1, 5.0 )

Interpretation:
    rho ~= 1   average task
    rho >  1   heat-heavy task (high compute or heavy data movement)
    rho <  1   light task

The 0.7 / 0.3 split reflects that on-die compute (ASIC) is the dominant
heat source in CPO, with optical I/O contributing meaningfully but
secondarily.

Usage
-----
    python compute_dag_features.py \\
        --input  data_pipeline/process/alibaba_dags.json \\
        --output data_pipeline/process/alibaba_dags_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------
def _build_graph(dag_dict: Dict[str, Any]) -> nx.DiGraph:
    """Build a NetworkX DiGraph from one entry of alibaba_dags.json.

    Expected schema (from trace_cleaner.py)::

        {
            "trace_id": str,
            "nodes":    { node_id: {"workload": float, ...}, ... },
            "edges":    [[u, v], ...]  OR  [[u, v, traffic], ...]
        }
    """
    G = nx.DiGraph()
    for nid, attrs in dag_dict.get("nodes", {}).items():
        attrs = dict(attrs)
        attrs["workload"] = float(attrs.get("workload", 1.0))
        G.add_node(nid, **attrs)

    for edge in dag_dict.get("edges", []):
        if len(edge) >= 3:
            u, v, traffic = edge[0], edge[1], float(edge[2])
        elif len(edge) == 2:
            u, v, traffic = edge[0], edge[1], 0.0
        else:
            continue
        if u in G and v in G:
            G.add_edge(u, v, traffic=traffic)
    return G


def _break_cycles(G: nx.DiGraph) -> int:
    """Greedily remove edges until G is acyclic. Returns # edges removed.

    Alibaba traces are *mostly* DAGs but a small fraction contain cycles
    (retries / loops in the call graph).  CPM requires acyclicity.
    """
    removed = 0
    while not nx.is_directed_acyclic_graph(G):
        try:
            cyc = list(nx.find_cycle(G))
        except nx.NetworkXNoCycle:
            break
        # find_cycle on a DiGraph returns 2-tuples (u, v)
        u, v = cyc[-1][0], cyc[-1][1]
        G.remove_edge(u, v)
        removed += 1
    return removed


# ---------------------------------------------------------------------
# Topology features
# ---------------------------------------------------------------------
def _compute_slack(G: nx.DiGraph) -> Tuple[float, Dict[Any, float]]:
    """Forward / backward CPM. Returns (makespan, {node: slack})."""
    if G.number_of_nodes() == 0:
        return 0.0, {}

    topo = list(nx.topological_sort(G))

    eft: Dict[Any, float] = {}
    for n in topo:
        w = G.nodes[n]["workload"]
        preds = list(G.predecessors(n))
        eft[n] = w if not preds else max(eft[p] for p in preds) + w
    makespan = max(eft.values()) if eft else 0.0

    lft: Dict[Any, float] = {}
    for n in reversed(topo):
        succs = list(G.successors(n))
        if not succs:
            lft[n] = makespan
        else:
            lft[n] = min(lft[s] - G.nodes[s]["workload"] for s in succs)

    slacks = {n: max(0.0, lft[n] - eft[n]) for n in topo}
    return makespan, slacks


def _compute_depth(G: nx.DiGraph) -> Dict[Any, int]:
    """Longest path *in node count* from any source to each node."""
    depth: Dict[Any, int] = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        depth[n] = 0 if not preds else max(depth[p] for p in preds) + 1
    return depth


def _compute_traffic(G: nx.DiGraph) -> Dict[Any, float]:
    """Total incoming + outgoing traffic per node."""
    return {
        n: (sum(G.edges[u, n].get("traffic", 0.0) for u in G.predecessors(n))
            + sum(G.edges[n, v].get("traffic", 0.0) for v in G.successors(n)))
        for n in G.nodes
    }


# ---------------------------------------------------------------------
# Thermal characteristic factor rho_i
# ---------------------------------------------------------------------
def _compute_rho(workloads: Dict[Any, float],
                 traffics:  Dict[Any, float]) -> Dict[Any, float]:
    """rho_i in [0.1, 5.0]; see module docstring for the formula."""
    w_pos = np.array([w for w in workloads.values() if w > 0], dtype=float)
    t_pos = np.array([t for t in traffics.values()  if t > 0], dtype=float)
    w_med = max(float(np.median(w_pos)) if w_pos.size > 0 else 1.0, 1e-6)
    t_med = max(float(np.median(t_pos)) if t_pos.size > 0 else 1.0, 1e-6)

    rho: Dict[Any, float] = {}
    for n in workloads:
        w_norm = workloads[n] / w_med
        t_norm = traffics[n]  / t_med
        rho[n] = float(np.clip(0.7 * w_norm + 0.3 * t_norm, 0.1, 5.0))
    return rho


# ---------------------------------------------------------------------
# Public single-DAG enrichment
# ---------------------------------------------------------------------
def enrich_dag(dag_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a v2-enriched copy of one DAG entry, or None if degenerate.

    The input dict is not mutated.
    """
    G = _build_graph(dag_dict)
    if G.number_of_nodes() == 0:
        return None

    _break_cycles(G)  # ensures CPM is well-defined

    makespan, slacks = _compute_slack(G)
    depths   = _compute_depth(G)
    traffics = _compute_traffic(G)
    workloads = {n: float(G.nodes[n]["workload"]) for n in G.nodes}
    rhos     = _compute_rho(workloads, traffics)

    enriched_nodes: Dict[str, Dict[str, Any]] = {}
    for n in G.nodes:
        s = slacks[n]
        enriched_nodes[str(n)] = {
            "workload":      workloads[n],
            "slack":         float(s),
            "depth":         int(depths[n]),
            "in_degree":     int(G.in_degree(n)),
            "out_degree":    int(G.out_degree(n)),
            "is_critical":   int(s < 1e-3),
            "rho":           rhos[n],
            "total_traffic": float(traffics[n]),
        }

    enriched_edges = [
        [str(u), str(v), float(d.get("traffic", 0.0))]
        for u, v, d in G.edges(data=True)
    ]

    median_w = float(np.median(list(workloads.values())))
    pos_traffic = [t for t in traffics.values() if t > 0]
    median_t = float(np.median(pos_traffic)) if pos_traffic else 0.0

    return {
        "trace_id":             dag_dict.get("trace_id", ""),
        "nodes":                enriched_nodes,
        "edges":                enriched_edges,
        "makespan_lower_bound": float(makespan),
        "median_workload":      median_w,
        "median_traffic":       median_t,
        "num_nodes":            G.number_of_nodes(),
        "num_edges":            G.number_of_edges(),
        "num_critical":         int(sum(1 for s in slacks.values() if s < 1e-3)),
    }


# ---------------------------------------------------------------------
# Batch driver / CLI
# ---------------------------------------------------------------------
def enrich_file(input_path: str,
                output_path: str,
                min_nodes: int = 2,
                verbose: bool = True) -> List[Dict[str, Any]]:
    """Enrich an entire JSON file. Returns the list of enriched DAGs."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if verbose:
        print(f"📂 Loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        dags = json.load(f)
    if verbose:
        print(f"   loaded {len(dags)} raw DAGs")

    enriched: List[Dict[str, Any]] = []
    skipped_empty = 0
    skipped_small = 0
    failed = 0

    for i, d in enumerate(dags):
        if verbose and (i + 1) % 1000 == 0:
            print(f"   processed {i + 1}/{len(dags)} ...")
        try:
            e = enrich_dag(d)
        except Exception as ex:
            failed += 1
            if verbose and failed <= 5:
                print(f"   ⚠️  DAG {d.get('trace_id', i)} failed: {ex}")
            continue
        if e is None:
            skipped_empty += 1
            continue
        if e["num_nodes"] < min_nodes:
            skipped_small += 1
            continue
        enriched.append(e)

    if verbose:
        print(f"\n✅ enriched {len(enriched)} DAGs  "
              f"(skipped {skipped_empty} empty, {skipped_small} too-small, "
              f"{failed} failed)")
        if enriched:
            sizes      = [d["num_nodes"]   for d in enriched]
            rhos_all   = [n["rho"] for d in enriched for n in d["nodes"].values()]
            crit_rates = [d["num_critical"] / d["num_nodes"] for d in enriched]
            print(f"   |V| stats : min={min(sizes)}, "
                  f"median={int(np.median(sizes))}, max={max(sizes)}")
            print(f"   ρ stats   : min={np.min(rhos_all):.2f}, "
                  f"median={np.median(rhos_all):.2f}, "
                  f"max={np.max(rhos_all):.2f}")
            print(f"   crit-frac : median={np.median(crit_rates):.2f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False)

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n💾 wrote {output_path} ({size_mb:.2f} MB)")
    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Compute v2 DAG features (rho, depth, critical, ...)"
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="Path to alibaba_dags.json (from trace_cleaner.py)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write alibaba_dags_v2.json")
    parser.add_argument("--min-nodes", type=int, default=2,
                        help="Drop DAGs with fewer than this many nodes "
                             "(default 2)")
    parser.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = parser.parse_args()

    enrich_file(args.input, args.output,
                min_nodes=args.min_nodes,
                verbose=not args.quiet)


if __name__ == "__main__":
    main()
