"""
dag_parser.py — v2 extended
===========================

``MicroserviceDAG`` with v2 feature support and dict-input compatibility.

Backward-compatible changes vs the v1 parser
--------------------------------------------
1. ``__init__`` now accepts EITHER:

     * a dict from ``alibaba_dags(_v2).json`` — the full graph is built
       immediately, including v2 features (slack/depth/rho/...) read from
       the JSON or computed on the fly if absent.
     * a string ``dag_id`` — produces an empty graph; the caller adds
       tasks/edges manually and then calls ``compute_slack_time()``.

2. New per-node attributes are read from the dict if present, otherwise
   computed via CPM and the same definitions as ``compute_dag_features.py``:

     - ``slack``, ``depth``, ``in_degree``, ``out_degree``
     - ``is_critical``, ``rho``, ``total_traffic``

3. New properties for the v2 env:

     - ``num_completed``, ``completion_progress``, ``is_done``
     - ``makespan_lower_bound`` (from JSON if available)

4. ``ready_tasks`` / ``completed_tasks`` are tracked **on the parser
   itself**, so the env doesn't need to maintain duplicate state. The
   v1 API (``get_ready_tasks(completed_list)``) still works.

Usage in the v2 env
-------------------
    dag = MicroserviceDAG(dag_dict_from_json)   # auto-built, ready to use
    ready = dag.get_ready_tasks()               # uses internal completed set
    dag.mark_completed(task_id)
    progress = dag.completion_progress          # in [0, 1]

Legacy v1 use (still works)
---------------------------
    dag = MicroserviceDAG("trace-001")
    dag.add_task("a", workload=10.0)
    dag.add_task("b", workload=20.0)
    dag.add_dependency("a", "b", traffic=100.0)
    dag.compute_slack_time()
    ready = dag.get_ready_tasks(completed_tasks=[])
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


class MicroserviceDAG:
    """Microservice DAG with v2 thermal-aware features."""

    # =================================================================
    # Construction
    # =================================================================
    def __init__(self, source: Union[str, Dict[str, Any]]):
        """Create a DAG from a JSON dict, or an empty one from a string id.

        Parameters
        ----------
        source
            Either the full alibaba_dags(_v2).json entry (dict) or a
            free-form string identifier (manual build mode).
        """
        self.graph: nx.DiGraph = nx.DiGraph()

        # Per-DAG metadata; populated from JSON or compute_slack_time
        self.makespan_lower_bound: float = 0.0
        self.median_workload:      float = 1.0
        self.median_traffic:       float = 1.0

        # Runtime progress state
        self._completed: set = set()

        # Hot-path caches; rebuilt whenever the topology is finalised
        self.successors_cache:   Dict[Any, List[Any]] = {}
        self.predecessors_cache: Dict[Any, List[Any]] = {}
        self.traffic_cache:      Dict[Tuple[Any, Any], float] = {}

        if isinstance(source, dict):
            self.dag_id = str(source.get("trace_id", "unnamed"))
            self._init_from_dict(source)
        else:
            # Empty graph; caller will populate via add_task / add_dependency
            self.dag_id = str(source)

    # -----------------------------------------------------------------
    # Dict-based initialisation (alibaba_dags_v2.json)
    # -----------------------------------------------------------------
    def _init_from_dict(self, d: Dict[str, Any]) -> None:
        # Add nodes with whatever v2 attributes the JSON has
        for nid, attrs in d.get("nodes", {}).items():
            self.graph.add_node(
                nid,
                workload=float(attrs.get("workload", 1.0)),
                slack=float(attrs.get("slack", 0.0)),
                depth=int(attrs.get("depth", 0)),
                in_degree=int(attrs.get("in_degree", 0)),
                out_degree=int(attrs.get("out_degree", 0)),
                is_critical=int(attrs.get("is_critical", 0)),
                rho=float(attrs.get("rho", 1.0)),
                total_traffic=float(attrs.get("total_traffic", 0.0)),
            )

        # Add edges in either [u,v] or [u,v,traffic] form
        for edge in d.get("edges", []):
            if len(edge) >= 3:
                u, v, traffic = edge[0], edge[1], float(edge[2])
            elif len(edge) == 2:
                u, v, traffic = edge[0], edge[1], 0.0
            else:
                continue
            if u in self.graph and v in self.graph:
                self.graph.add_edge(u, v, traffic=traffic)

        # Per-DAG metadata
        self.makespan_lower_bound = float(d.get("makespan_lower_bound", 0.0))
        self.median_workload      = float(d.get("median_workload", 1.0))
        self.median_traffic       = float(d.get("median_traffic", 1.0))

        # If the JSON predates v2 (no makespan_lower_bound key), compute now
        is_v2 = "makespan_lower_bound" in d
        if not is_v2:
            self.compute_slack_time()
        else:
            self._build_caches()

    # =================================================================
    # Manual building (legacy v1 API)
    # =================================================================
    def add_task(self, task_id: Any, workload: float) -> None:
        """Legacy v1 API. Creates a task with default v2 attributes."""
        self.graph.add_node(
            task_id,
            workload=float(workload),
            slack=0.0, depth=0,
            in_degree=0, out_degree=0,
            is_critical=0, rho=1.0,
            total_traffic=0.0,
        )

    def add_dependency(self, u: Any, v: Any, traffic: float = 0.0) -> None:
        """Legacy v1 API. Adds an edge with optional traffic weight."""
        self.graph.add_edge(u, v, traffic=float(traffic))

    # =================================================================
    # Slack / topology computation
    # =================================================================
    def compute_slack_time(self) -> None:
        """Compute slack, depth, degrees, traffic, rho via CPM.

        Idempotent. Should be called after manual building or whenever
        the topology has changed.
        """
        # Cycle defence (CPM requires acyclicity)
        while not nx.is_directed_acyclic_graph(self.graph):
            try:
                cyc = list(nx.find_cycle(self.graph))
            except nx.NetworkXNoCycle:
                break
            u, v = cyc[-1][0], cyc[-1][1]
            self.graph.remove_edge(u, v)

        if self.graph.number_of_nodes() == 0:
            self.makespan_lower_bound = 0.0
            self._build_caches()
            return

        topo = list(nx.topological_sort(self.graph))

        # Forward pass (EFT)
        eft: Dict[Any, float] = {}
        for n in topo:
            w = self.graph.nodes[n]["workload"]
            preds = list(self.graph.predecessors(n))
            eft[n] = w if not preds else max(eft[p] for p in preds) + w
        makespan = max(eft.values()) if eft else 0.0
        self.makespan_lower_bound = float(makespan)

        # Backward pass (LFT)
        lft: Dict[Any, float] = {}
        for n in reversed(topo):
            succs = list(self.graph.successors(n))
            if not succs:
                lft[n] = makespan
            else:
                lft[n] = min(
                    lft[s] - self.graph.nodes[s]["workload"] for s in succs
                )

        # Slack + critical-path flag
        for n in self.graph.nodes:
            s = max(0.0, lft[n] - eft[n])
            self.graph.nodes[n]["slack"] = float(s)
            self.graph.nodes[n]["is_critical"] = int(s < 1e-3)

        # Depth
        for n in topo:
            preds = list(self.graph.predecessors(n))
            self.graph.nodes[n]["depth"] = (
                0 if not preds else
                max(self.graph.nodes[p]["depth"] for p in preds) + 1
            )

        # Degrees + total traffic
        for n in self.graph.nodes:
            self.graph.nodes[n]["in_degree"]  = self.graph.in_degree(n)
            self.graph.nodes[n]["out_degree"] = self.graph.out_degree(n)
            in_t  = sum(self.graph.edges[u, n].get("traffic", 0.0)
                        for u in self.graph.predecessors(n))
            out_t = sum(self.graph.edges[n, v].get("traffic", 0.0)
                        for v in self.graph.successors(n))
            self.graph.nodes[n]["total_traffic"] = float(in_t + out_t)

        # rho — same definition as compute_dag_features.py
        workloads = {n: self.graph.nodes[n]["workload"] for n in self.graph.nodes}
        traffics  = {n: self.graph.nodes[n]["total_traffic"]
                     for n in self.graph.nodes}
        w_pos = np.array([w for w in workloads.values() if w > 0], dtype=float)
        t_pos = np.array([t for t in traffics.values()  if t > 0], dtype=float)
        w_med = max(float(np.median(w_pos)) if w_pos.size > 0 else 1.0, 1e-6)
        t_med = max(float(np.median(t_pos)) if t_pos.size > 0 else 1.0, 1e-6)
        self.median_workload = w_med
        self.median_traffic  = t_med if t_pos.size > 0 else 1.0
        for n in self.graph.nodes:
            w_norm = workloads[n] / w_med
            t_norm = traffics[n]  / t_med
            self.graph.nodes[n]["rho"] = float(
                np.clip(0.7 * w_norm + 0.3 * t_norm, 0.1, 5.0)
            )

        self._build_caches()

    # =================================================================
    # Hot-path caches
    # =================================================================
    def _build_caches(self) -> None:
        self.successors_cache = {
            n: list(self.graph.successors(n)) for n in self.graph.nodes
        }
        self.predecessors_cache = {
            n: list(self.graph.predecessors(n)) for n in self.graph.nodes
        }
        self.traffic_cache = {
            (u, v): float(self.graph.edges[u, v].get("traffic", 0.0))
            for u, v in self.graph.edges
        }

    # =================================================================
    # Runtime API used by the env
    # =================================================================
    def get_ready_tasks(self,
                        completed: Optional[List[Any]] = None) -> List[Any]:
        """Return tasks whose predecessors are all complete.

        If ``completed`` is provided (legacy v1 call style), use it as the
        completion set; otherwise fall back to the parser's internal
        ``_completed`` set.
        """
        done = self._completed if completed is None else set(completed)
        ready: List[Any] = []
        for n in self.graph.nodes:
            if n in done:
                continue
            preds = self.predecessors_cache.get(n, [])
            if all(p in done for p in preds):
                ready.append(n)
        return ready

    def mark_completed(self, task_id: Any) -> None:
        self._completed.add(task_id)

    def reset_progress(self) -> None:
        self._completed.clear()

    @property
    def num_completed(self) -> int:
        return len(self._completed)

    @property
    def num_total(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def completion_progress(self) -> float:
        n = self.num_total
        return 0.0 if n == 0 else float(self.num_completed) / n

    @property
    def is_done(self) -> bool:
        return self.num_completed >= self.num_total > 0

    # =================================================================
    # Convenience accessors
    # =================================================================
    def workload(self, task_id: Any) -> float:
        return float(self.graph.nodes[task_id]["workload"])

    def slack(self, task_id: Any) -> float:
        return float(self.graph.nodes[task_id].get("slack", 0.0))

    def rho(self, task_id: Any) -> float:
        return float(self.graph.nodes[task_id].get("rho", 1.0))

    def depth(self, task_id: Any) -> int:
        return int(self.graph.nodes[task_id].get("depth", 0))

    def is_critical(self, task_id: Any) -> bool:
        return bool(self.graph.nodes[task_id].get("is_critical", 0))

    def task_features(self, task_id: Any) -> List[float]:
        """Return the 8-d task feature vector consumed by the v2 GNN encoder.

        Layout (matches §3.2 of the reconstruction plan)::

            [ workload/50, slack/100, rho, depth/10,
              in_deg/5, out_deg/5, critical_flag, dag_progress ]
        """
        nd = self.graph.nodes[task_id]
        return [
            nd["workload"]               / 50.0,
            nd.get("slack", 0.0)         / 100.0,
            nd.get("rho", 1.0),
            nd.get("depth", 0)           / 10.0,
            nd.get("in_degree", 0)       / 5.0,
            nd.get("out_degree", 0)      / 5.0,
            float(nd.get("is_critical", 0)),
            self.completion_progress,
        ]

    # =================================================================
    # Dunder methods
    # =================================================================
    def __len__(self) -> int:
        return self.num_total

    def __repr__(self) -> str:
        return (f"MicroserviceDAG(id={self.dag_id!r}, "
                f"|V|={self.num_total}, |E|={self.graph.number_of_edges()}, "
                f"completion={self.completion_progress:.0%})")
