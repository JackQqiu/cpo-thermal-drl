"""
baselines/heft.py — Classic HEFT (Heterogeneous Earliest Finish Time)
======================================================================

The canonical scheduling baseline for heterogeneous platforms (Topcuoglu
et al. 2002).  Two phases:

  1. **Upward rank**: assigns each task a priority based on the longest
     path from itself to any exit task, weighted by the average
     execution time across processors.  Tasks with higher ranks are
     scheduled first.

  2. **Processor selection**: for each task in upward-rank order, pick
     the processor that gives the earliest **finish time**, accounting
     for processor availability and inter-task communication.

Adaptation to our env
---------------------
* The env hands us tasks one at a time in DAG-topological order
  (the env itself does the upward-rank-like sorting via ``ready_tasks``).
  HEFT's ranking phase therefore degenerates to "use the order the env
  gives us".
* What HEFT contributes here is the **processor selection** policy:
  given the current task, pick the proc that minimises
  ``available_time + execution_time``.

This is the "thermally-blind" version — it ignores temperature
entirely.  See :mod:`baselines.thermal_heft` for the PROCHOT-aware
variant.

Notes
-----
* For our hardware: ASIC is fast (high throughput) but high-power; OEs
  are slow but low-power.  HEFT will tend to push everything to the
  ASIC because its execution time is shortest, even though that
  thermally bottlenecks the system.  This is exactly the failure mode
  Thermal-HEFT addresses.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Action, BaseScheduler


class HEFTScheduler(BaseScheduler):
    """Classic HEFT processor selection.

    For the current task, pick the proc i that minimises::

        next_free[i] + estimated_exec_time[i]

    where ``estimated_exec_time[i]`` is read from the task→proc edge
    attributes the env exposes.  ``next_free`` is tracked locally
    (each proc becomes free after its current task's execution ends).
    """

    name = "HEFT"

    def __init__(
        self,
        action_mode: str = "auto_only",
        K_delay: int = 5,
        num_nodes: int = 17,
    ):
        super().__init__(action_mode=action_mode, K_delay=K_delay)
        self.num_nodes = num_nodes
        # Per-proc "next available time" (since episode start, in ms)
        self._next_free = np.zeros(num_nodes, dtype=np.float64)
        self._global_clock = 0.0   # tracks elapsed wall time

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        self._next_free = np.zeros(self.num_nodes, dtype=np.float64)
        self._global_clock = 0.0

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        mask = info["action_mask"]
        graph = info["graph_obs"][0] if isinstance(info["graph_obs"],
                                                    (list, np.ndarray)) \
                else info["graph_obs"]
        # Estimate execution time per proc from task→proc edge attributes.
        # graph['edges_t2p'] is a list of [task_local_idx, proc_idx] and
        # graph['edges_t2p_attr'] is a list of [est_time_ms_norm, est_dT_norm].
        # We need est_time for the CURRENT task only.
        cur = int(graph.get("current_task_idx", 0))
        edges = graph.get("edges_t2p", [])
        attrs = graph.get("edges_t2p_attr", [])

        # Build per-proc estimated execution time (default to a large
        # constant if not advertised — discourages picking that proc)
        est_time = np.full(self.num_nodes, 1e6, dtype=np.float64)
        for (u, v), attr in zip(edges, attrs):
            if int(u) == cur and 0 <= int(v) < self.num_nodes:
                # attr[0] is normalized exec time; multiply by 50 to undo
                # the env's normalisation (see _build_graph_obs in env).
                est_time[int(v)] = float(attr[0]) * 50.0

        # Earliest finish time per proc
        eft = self._next_free + est_time
        # Mask out unavailable procs
        eft[~mask] = np.inf
        chosen = int(np.argmin(eft))

        # Update local proc-availability tracker
        self._next_free[chosen] = eft[chosen]
        self._global_clock = max(self._global_clock, self._next_free[chosen])

        return self._wrap_action(chosen, delay_idx=0)
