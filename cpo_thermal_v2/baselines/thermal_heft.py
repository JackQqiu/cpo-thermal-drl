"""
baselines/thermal_heft.py — PROCHOT-aware HEFT
==============================================

Augments the classic HEFT processor-selection rule with a thermal
penalty term.  Instead of selecting the proc that minimises pure
execution time, this scheduler picks the proc minimising::

    cost(i) = next_free[i] + α · est_time[i] + β · est_dT[i] + γ · (T_i - T_target)_+

where:
  * ``est_time[i]`` and ``est_dT[i]`` are exposed by the env as
    task→proc edge attributes (the same features the GNN sees);
  * ``(T_i - T_target)_+`` is the rectified excess over the target
    operating temperature — a soft barrier discouraging hot procs.

Origin
------
Variants of this scheme have appeared in the thermal-aware DAG
scheduling literature for over a decade (e.g. Coskun et al. 2008,
Hanumaiah & Vrudhula 2012, Lee et al. 2013).  They are referred to
collectively as "Thermal-HEFT" or "Heat-aware HEFT".  We use a
deliberately simple linear-combination form — adding more terms or
quadratic penalties tends to overfit to one operating condition.

Why it matters as a baseline
----------------------------
This is the **strongest non-learning baseline** for our paper:
* It uses the same observation dimensions the GNN sees (so the GNN's
  advantage isn't from richer observations, it's from a learned policy).
* It uses a hand-engineered thermal heuristic — exactly the kind of
  approach we claim to outperform.

If the GNN-DRL doesn't beat Thermal-HEFT, the paper has a problem.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Action, BaseScheduler


class ThermalHEFTScheduler(BaseScheduler):
    """HEFT + thermal penalty in the proc-selection cost.

    Hyper-parameters (defaults tuned on a separate validation set; the
    paper should report sensitivity)::

        alpha = 1.0    weight on raw execution time
        beta  = 5.0    weight on estimated temperature rise
        gamma = 0.5    weight on excess over T_target

    With these defaults, a proc 5°C above T_target effectively adds
    ``0.5 × 5 = 2.5`` ms-equivalent cost — comparable to choosing a
    slightly slower-but-cooler alternative proc.
    """

    name = "ThermalHEFT"

    def __init__(
        self,
        action_mode: str = "auto_only",
        K_delay: int = 5,
        num_nodes: int = 17,
        T_target: float = 75.0,
        alpha: float = 1.0,
        beta:  float = 5.0,
        gamma: float = 0.5,
    ):
        super().__init__(action_mode=action_mode, K_delay=K_delay)
        self.num_nodes = num_nodes
        self.T_target  = T_target
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self._next_free = np.zeros(num_nodes, dtype=np.float64)

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        self._next_free = np.zeros(self.num_nodes, dtype=np.float64)

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        mask = info["action_mask"]
        graph = info["graph_obs"][0] if isinstance(info["graph_obs"],
                                                    (list, np.ndarray)) \
                else info["graph_obs"]

        # Per-proc estimated execution time and temperature rise from
        # task→proc edge attributes (same as HEFT).
        cur = int(graph.get("current_task_idx", 0))
        edges = graph.get("edges_t2p", [])
        attrs = graph.get("edges_t2p_attr", [])
        est_time = np.full(self.num_nodes, 1e6, dtype=np.float64)
        est_dT   = np.full(self.num_nodes, 1e3, dtype=np.float64)
        for (u, v), attr in zip(edges, attrs):
            if int(u) == cur and 0 <= int(v) < self.num_nodes:
                est_time[int(v)] = float(attr[0]) * 50.0
                est_dT[int(v)]   = float(attr[1]) * 20.0

        # Per-proc current temperature, recovered from obs (which is the
        # raw temperature vector — see env: returns thermal_engine.temperatures)
        T = np.asarray(obs, dtype=np.float64).flatten()
        if T.size != self.num_nodes:
            # Fall back to proc_x[:, 0] (normalised T) if obs has wrong shape
            proc_x = np.asarray(graph.get("proc_x", []), dtype=np.float64)
            if proc_x.ndim == 2 and proc_x.shape[0] == self.num_nodes:
                T = proc_x[:, 0] * 60.0 + 25.0     # un-normalise
            else:
                T = np.full(self.num_nodes, self.T_target)

        excess = np.maximum(0.0, T - self.T_target)

        # Combined cost
        cost = (self._next_free
                + self.alpha * est_time
                + self.beta  * est_dT
                + self.gamma * excess)
        cost[~mask] = np.inf
        chosen = int(np.argmin(cost))

        self._next_free[chosen] = self._next_free[chosen] + est_time[chosen]
        return self._wrap_action(chosen, delay_idx=0)
