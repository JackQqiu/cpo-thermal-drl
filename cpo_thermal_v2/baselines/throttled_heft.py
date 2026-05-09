"""
baselines/throttled_heft.py — HEFT with reactive thermal throttling
=====================================================================

Throttled-HEFT addresses a methodological concern raised in peer review:
the apparent makespan ranking between vanilla HEFT and the proposed
DRL methods is not like-for-like, because vanilla HEFT episodes are
truncated mid-execution under the hard-wall thermal policy (mean DAG
completion ~27%), while the DRL methods complete 100% of the DAG
workload.

Throttled-HEFT augments classical HEFT with a *reactive* thermal-
safety mechanism that operates on TWO axes, in priority order:

  1. **Delay axis** (primary): when a dispatch is predicted unsafe and
     the task has positive slack, emit a maximum-delay action (k=K_delay-1).
     The env applies ``delay_fractions[k] * slack`` ms of cooling
     before dispatching, capped at ``max_agent_delay_ms``.
  2. **Placement axis** (fallback): when a dispatch is predicted unsafe
     AND the task is on the critical path (slack = 0), Throttled-HEFT
     re-selects the fastest *currently coolest-enough* processor
     instead of HEFT's original "fastest" pick.

The placement fallback activates only when the delay axis is
structurally unable to help.  For non-critical tasks the algorithm
preserves HEFT's "min EFT" placement rule.

Trigger predicate (when to react)
---------------------------------
A dispatch is predicted unsafe iff
    current_T[i] + dT_inflation * est_dT[i]  >=  T_warn,
where T_warn = T_crit - safety_margin.  Aggressive on purpose:
the env truncates on PEAK temperature DURING execution, which can
exceed `current_T + est_dT` by 5-15 °C in extreme regimes.

Safety filter (which procs are admissible alternatives)
-------------------------------------------------------
For the placement-fallback branch, a proc is admissible iff
    current_T[i]  <  T_warn.
Notice this uses a DIFFERENT predicate from the trigger: trigger uses
inflated lookahead (catch real threats); fallback uses observed-T
only (don't filter out viable alternatives).  Without this asymmetry,
the algorithm degenerates: in stress regimes most procs fail the
trigger predicate, but enough still satisfy `T_pre < T_warn` to be
useful fallbacks.  The dispatch-on-cooler-proc may still cause some
heating, but starting from a cooler baseline gives more headroom
versus the env's RC overshoot during execution.

Calibration history
-------------------
Calibrated on the extreme-ambient unit test (50 episodes,
T_0 in [60, 75] °C, N=17, hard truncate).  Three iterations:

  * v1 (predicted_T >= T_crit, no fallback): trigger fired on
    < 0.5% of dispatches; degenerate to vanilla HEFT.
  * v2 (added PROCHOT current_T trigger): no improvement; current_T
    rarely crosses T_crit-10 because heat spikes happen DURING exec.
  * v3 (T_warn=75 trigger, dT_inflation=2.0, no fallback): trigger
    fires on 7% of dispatches but 91% of those land on slack=0
    critical-path tasks where env's slack-bounded delay collapses
    to 0 ms; effective cooling per episode 15 ms.
  * v4 (added placement fallback with same safety filter as trigger):
    no improvement; safety filter too strict, `safe.any()` false in
    most stressful states.
  * v5 (current): safety filter relaxed to T_pre only.  Trigger
    stays aggressive (catch real threats); fallback stays permissive
    (find any reasonable alternative).

A pure-delay-only sub-ablation (``placement_fallback=False``) is
preserved to support a paragraph in the paper showing that the
delay axis ALONE — even with the same action space as our PPO
agent — is unable to prevent truncation in stress regimes (>90% of
triggers fall on critical-path tasks).

Action-mode coverage
--------------------
Both ``hybrid`` and ``agent_only`` are supported; two separate rows
in the results.  Note: in our env, ``_maybe_precool`` is gated on
the target proc already being above ``precool_target_temp``, so
hybrid and agent_only often produce near-identical numbers — the
agent's own logic carries the work in both modes.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Action, BaseScheduler


class ThrottledHEFTScheduler(BaseScheduler):
    """HEFT processor selection + reactive thermal stall (delay axis)
    + slack=0 placement fallback (T_pre-based safety filter).

    Parameters
    ----------
    action_mode : str
        Must be 'hybrid' or 'agent_only' to expose the delay head.
    K_delay : int
        Number of discrete delay levels (default 5).
    num_nodes : int
        Number of processors in the system (default 17).
    T_crit : float
        Hard thermal limit in degrees Celsius (default 85).
    safety_margin : float
        Margin in °C below T_crit defining ``T_warn = T_crit - safety_margin``.
        Default 10.0 (T_warn = 75, matching env.thermal_target).
    dT_inflation : float
        Multiplier on the static est_dT in the trigger predicate.
        Default 2.0.  Used ONLY in the trigger; the safety filter uses
        T_pre alone.
    placement_fallback : bool
        If True (default), activate the slack=0 placement fallback.
        If False, behave as the pure delay-only variant (sub-ablation).
    """

    name = "Throttled-HEFT"

    _ALLOWED_MODES = ("hybrid", "agent_only")

    def __init__(
        self,
        action_mode: str   = "hybrid",
        K_delay: int       = 5,
        num_nodes: int     = 17,
        T_crit: float      = 85.0,
        safety_margin: float = 10.0,
        dT_inflation: float  = 2.0,
        placement_fallback: bool = True,
    ):
        if action_mode not in self._ALLOWED_MODES:
            raise ValueError(
                f"Throttled-HEFT requires action_mode in "
                f"{self._ALLOWED_MODES} to expose the delay action; "
                f"got '{action_mode}'."
            )
        super().__init__(action_mode=action_mode, K_delay=K_delay)
        self.num_nodes        = num_nodes
        self.T_crit           = T_crit
        self.safety_margin    = safety_margin
        self.dT_inflation     = dT_inflation
        self.placement_fallback = bool(placement_fallback)

        # HEFT bookkeeping
        self._next_free    = np.zeros(num_nodes, dtype=np.float64)
        self._global_clock = 0.0

        # Telemetry: counts decisions taken per category for end-of-run
        # reporting / debugging.  Reset each episode.
        self._tally = {
            "n_dispatch":          0,
            "n_unsafe":            0,
            "n_throttle_path":     0,  # slack > 0 → max delay
            "n_fallback_relocate": 0,  # slack = 0 → re-picked safer proc
            "n_fallback_no_safe":  0,  # slack = 0 → no safer proc, kept HEFT
            "n_pure_delay":        0,  # placement_fallback=False, slack=0
        }

    @property
    def tally(self) -> Dict[str, int]:
        """Read-only telemetry of the most recent episode."""
        return dict(self._tally)

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        self._next_free    = np.zeros(self.num_nodes, dtype=np.float64)
        self._global_clock = 0.0
        for k in self._tally:
            self._tally[k] = 0

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        mask = info["action_mask"]
        graph = info["graph_obs"][0] if isinstance(info["graph_obs"],
                                                    (list, np.ndarray)) \
                else info["graph_obs"]

        # ----- Step 1: HEFT processor selection (vanilla rule) -----
        cur   = int(graph.get("current_task_idx", 0))
        edges = graph.get("edges_t2p", [])
        attrs = graph.get("edges_t2p_attr", [])

        # est_time and est_dT from t2p edge attributes
        est_time = np.full(self.num_nodes, 1e6, dtype=np.float64)
        est_dT   = np.zeros(self.num_nodes,    dtype=np.float64)
        for (u, v), attr in zip(edges, attrs):
            if int(u) == cur and 0 <= int(v) < self.num_nodes:
                est_time[int(v)] = float(attr[0]) * 50.0
                est_dT[int(v)]   = float(attr[1]) * 20.0

        eft = self._next_free + est_time
        eft[~mask] = np.inf
        heft_chosen = int(np.argmin(eft))

        # ----- Step 2: Per-proc current and predicted temperatures -----
        proc_x = graph.get("proc_x", None)
        if proc_x is None:
            T_pre_all = np.full(self.num_nodes, 25.0, dtype=np.float64)
        else:
            proc_x_arr = np.asarray(proc_x, dtype=np.float64)
            if proc_x_arr.ndim == 2 and proc_x_arr.shape[0] >= self.num_nodes:
                T_pre_all = proc_x_arr[: self.num_nodes, 0] * 60.0 + 25.0
            else:
                T_pre_all = np.full(self.num_nodes, 25.0, dtype=np.float64)

        predicted_T_all = T_pre_all + self.dT_inflation * est_dT

        # ----- Step 3: Read slack of the current head task -----
        task_x = graph.get("task_x", [])
        slack_ms = 0.0
        if 0 <= cur < len(task_x):
            tx = task_x[cur]
            if len(tx) >= 7:
                slack_ms = float(tx[1]) * 100.0

        # ----- Step 4: Decide chosen proc + delay -----
        T_warn    = self.T_crit - self.safety_margin
        is_unsafe = predicted_T_all[heft_chosen] >= T_warn

        chosen    = heft_chosen
        delay_idx = 0

        self._tally["n_dispatch"] += 1
        if is_unsafe:
            self._tally["n_unsafe"] += 1
            if slack_ms > 1e-3:
                # Delay axis admissible
                chosen    = heft_chosen
                delay_idx = self.K_delay - 1
                self._tally["n_throttle_path"] += 1
            elif self.placement_fallback:
                # slack = 0 → placement fallback.
                # Safety filter uses T_pre alone (NOT predicted_T) to
                # avoid filtering out all viable alternatives in stress
                # regimes.  A proc with T_pre < T_warn has at least
                # some headroom for the upcoming dispatch.
                safe = (T_pre_all < T_warn) & mask
                if safe.any():
                    eft_safe = eft.copy()
                    eft_safe[~safe] = np.inf
                    chosen    = int(np.argmin(eft_safe))
                    delay_idx = 0
                    self._tally["n_fallback_relocate"] += 1
                else:
                    # No proc below T_warn — keep HEFT's pick.
                    chosen    = heft_chosen
                    delay_idx = 0
                    self._tally["n_fallback_no_safe"] += 1
            else:
                # Sub-ablation: pure delay only, no fallback.  Emit max
                # delay; env collapses to 0 ms because slack=0.
                chosen    = heft_chosen
                delay_idx = self.K_delay - 1
                self._tally["n_pure_delay"] += 1

        # ----- Step 5: Local proc-availability bookkeeping -----
        # The env will dispatch on `chosen` THIS step.
        self._next_free[chosen] = self._next_free[chosen] + est_time[chosen]
        self._global_clock      = max(self._global_clock,
                                      self._next_free[chosen])

        return self._wrap_action(chosen, delay_idx=delay_idx)
