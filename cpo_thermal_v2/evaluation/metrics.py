"""
evaluation/metrics.py — Per-episode and per-DAG metric collection
==================================================================

The trainer logs PPO health (loss/KL/entropy) but NOT the publication-
relevant *physics* metrics.  Stage E measures the latter, end-to-end on
held-out episodes::

    Per-step (collected during episode rollout):
        peak_temp_during                — max chip T while task executing
        peak_temp_idle                  — max chip T at end of step
        cooling_overhead_ms             — env auto-cool insertion (ms)
        agent_delay_ms                  — agent-chosen delay (ms)
        would_violate_without_delay     — bool, "did env have to save us?"
        proc_utilisation[i]             — per-proc busy fraction
        ...

    Per-DAG (one row per DAG completed):
        dag_id                          — alibaba trace_id
        num_tasks                       — |V| of this DAG
        rho                             — communication-to-computation ratio
        is_critical_heavy               — fraction of critical-path tasks
        makespan_ms                     — actual wall-time
        makespan_lower_bound            — from compute_dag_features (theoretical)
        makespan_normalized             — actual / lower_bound (≥ 1.0)
        peak_temp_dag                   — max T during this DAG
        cooling_overhead_total_ms
        violations_count                — number of would_violate==True steps

    Per-episode (one row per episode):
        episode_id                      — sequential index
        scheduler                       — name of policy under test
        num_nodes                       — N (topology size)
        action_mode                     — auto_only / agent_only / hybrid
        total_makespan_ms               — sum across DAGs in this episode
        mean_makespan_normalized        — averaged over DAGs
        peak_temp_episode               — max over all steps
        violations_total                — sum of per-step violations
        cooling_total_ms                — sum of env-cooling time
        agent_delay_total_ms            — sum of agent-controlled delay
        dag_completion_rate             — # of DAGs completed
        truncated                       — episode hit T_crit hard wall
        episode_return                  — total reward
        wallclock_seconds               — real time spent

API
---
``EpisodeRecorder`` — accumulates per-step / per-DAG / per-episode data
across a single episode.  Reset between episodes.

``run_episode(env, scheduler, recorder)`` — drives a full episode under
the given scheduler, populating ``recorder`` and returning the per-DAG
and per-episode summary dicts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# =====================================================================
# Recorder
# =====================================================================
@dataclass
class StepRecord:
    """One per-step entry — kept lean (~100 bytes/step)."""
    step:                   int
    placement:              int
    delay_ms:               float
    exec_time_ms:           float
    cooling_ms:             float
    peak_temp_during:       float
    peak_temp_idle:         float
    would_violate:          bool
    dag_done:               bool
    truncated:              bool
    reward:                 float
    reward_placement:       float
    reward_delay:           float
    proc_utilisation_mask:  Optional[np.ndarray] = None  # which proc was busy


@dataclass
class DAGRecord:
    """One per-DAG entry — emitted on dag_done==True."""
    dag_index_in_episode: int
    trace_id:             str
    num_tasks:            int
    num_edges:            int
    num_critical:         int
    rho:                  float
    makespan_lower_bound: float
    makespan_actual_ms:   float
    makespan_normalized:  float
    peak_temp_dag:        float
    cooling_total_ms:     float
    agent_delay_total_ms: float
    violations_count:     int
    n_steps:              int


@dataclass
class EpisodeRecord:
    """Top-level episode summary.  This is what gets aggregated for tables."""
    episode_id:                int
    scheduler:                 str
    num_nodes:                 int
    action_mode:               str
    total_makespan_ms:         float
    mean_makespan_normalized:  float
    peak_temp_episode:         float
    mean_temp_episode:         float
    violations_total:          int
    cooling_total_ms:          float
    agent_delay_total_ms:      float
    dag_completion_rate:       float          # DAGs completed / DAGs available
    truncated:                 bool
    episode_return:            float
    wallclock_seconds:         float
    n_steps:                   int
    # Per-proc utilisation: fraction of steps each proc was the active one
    proc_utilisation:          List[float]    = field(default_factory=list)


class EpisodeRecorder:
    """Accumulates step / DAG / episode records for a single episode."""

    def __init__(
        self,
        scheduler_name: str,
        num_nodes:      int,
        action_mode:    str,
        episode_id:     int,
    ):
        self.scheduler_name = scheduler_name
        self.num_nodes      = num_nodes
        self.action_mode    = action_mode
        self.episode_id     = episode_id

        self.steps:        List[StepRecord] = []
        self.dags:         List[DAGRecord]  = []
        # Track running per-DAG state (cleared each time a DAG completes)
        self._cur_dag_makespan_ms:    float = 0.0
        self._cur_dag_peak_temp:      float = 0.0
        self._cur_dag_cooling_ms:     float = 0.0
        self._cur_dag_agent_delay_ms: float = 0.0
        self._cur_dag_violations:     int   = 0
        self._cur_dag_n_steps:        int   = 0
        self._cur_dag_meta:           Optional[Dict[str, Any]] = None
        self._wallclock_start:        float = time.time()

        # Proc-utilisation counters: how many steps each proc was active
        self._proc_busy_count = np.zeros(num_nodes, dtype=np.int64)
        self._total_steps     = 0

    # -----------------------------------------------------------------
    def update_dag_meta(self, info: Dict[str, Any]) -> None:
        """Called at the start of each new DAG to capture its metadata.

        ``info`` must contain a ``dag_meta`` dict with keys:
          trace_id, num_nodes, num_edges, num_critical, rho,
          makespan_lower_bound.

        If the env's info doesn't expose this directly, we fall back to
        sentinel defaults (``rho=0``, ``makespan_lower_bound=0``) so the
        per-DAG record is still produced.
        """
        meta = info.get("dag_meta", None)
        if meta is None:
            # Synthesise from whatever's available
            meta = {
                "trace_id":             info.get("trace_id", "unknown"),
                "num_tasks":            info.get("num_tasks_in_dag", 0),
                "num_edges":            info.get("num_edges_in_dag", 0),
                "num_critical":         0,
                "rho":                  0.0,
                "makespan_lower_bound": 0.0,
            }
        self._cur_dag_meta = dict(meta)

    # -----------------------------------------------------------------
    def add_step(
        self,
        *,
        step:              int,
        placement:         int,
        delay_ms:          float,
        info:              Dict[str, Any],
        reward:            float,
        reward_placement:  float,
        reward_delay:      float,
        truncated:         bool,
    ) -> None:
        """Record one env step.

        ``info`` is the env's per-step info dict (single-env, NOT vector).
        """
        exec_ms      = float(info.get("pure_compute_ms",     0.0))
        cooling_ms   = float(info.get("cooling_overhead_ms", 0.0))
        agent_dly    = float(info.get("agent_delay_ms",      delay_ms))
        peak_during  = float(info.get("max_temp",            0.0))
        peak_idle    = float(info.get("idle_max_temp",       peak_during))
        violated     = bool(info.get("would_violate_without_delay", False))
        dag_done     = bool(info.get("dag_done", False))

        self.steps.append(StepRecord(
            step             = step,
            placement        = placement,
            delay_ms         = agent_dly,
            exec_time_ms     = exec_ms,
            cooling_ms       = cooling_ms,
            peak_temp_during = peak_during,
            peak_temp_idle   = peak_idle,
            would_violate    = violated,
            dag_done         = dag_done,
            truncated        = truncated,
            reward           = reward,
            reward_placement = reward_placement,
            reward_delay     = reward_delay,
        ))

        # Per-DAG running totals
        step_total_ms = exec_ms + cooling_ms + agent_dly
        self._cur_dag_makespan_ms    += step_total_ms
        self._cur_dag_peak_temp       = max(self._cur_dag_peak_temp, peak_during)
        self._cur_dag_cooling_ms     += cooling_ms
        self._cur_dag_agent_delay_ms += agent_dly
        if violated:
            self._cur_dag_violations += 1
        self._cur_dag_n_steps += 1

        # Proc utilisation: this step the active proc is `placement`
        if 0 <= placement < self.num_nodes:
            self._proc_busy_count[placement] += 1
        self._total_steps += 1

        # If the DAG just completed, snapshot a DAGRecord and reset
        if dag_done and self._cur_dag_meta is not None:
            self._finalise_current_dag()

    def _finalise_current_dag(self) -> None:
        meta = self._cur_dag_meta or {}
        lb = float(meta.get("makespan_lower_bound", 0.0))
        actual = float(self._cur_dag_makespan_ms)
        self.dags.append(DAGRecord(
            dag_index_in_episode = len(self.dags),
            trace_id             = str(meta.get("trace_id", "unknown")),
            num_tasks            = int(meta.get("num_tasks", meta.get("num_nodes", 0))),
            num_edges            = int(meta.get("num_edges", 0)),
            num_critical         = int(meta.get("num_critical", 0)),
            rho                  = float(meta.get("rho", 0.0)),
            makespan_lower_bound = lb,
            makespan_actual_ms   = actual,
            makespan_normalized  = (actual / lb) if lb > 0 else float("nan"),
            peak_temp_dag        = self._cur_dag_peak_temp,
            cooling_total_ms     = self._cur_dag_cooling_ms,
            agent_delay_total_ms = self._cur_dag_agent_delay_ms,
            violations_count     = self._cur_dag_violations,
            n_steps              = self._cur_dag_n_steps,
        ))
        # Reset DAG-running state
        self._cur_dag_makespan_ms    = 0.0
        self._cur_dag_peak_temp      = 0.0
        self._cur_dag_cooling_ms     = 0.0
        self._cur_dag_agent_delay_ms = 0.0
        self._cur_dag_violations     = 0
        self._cur_dag_n_steps        = 0
        self._cur_dag_meta           = None

    # -----------------------------------------------------------------
    def finalise_episode(
        self,
        episode_return:       float,
        truncated:            bool,
        n_dags_available:     int = 0,
    ) -> EpisodeRecord:
        """Build and return the top-level :class:`EpisodeRecord`."""
        # If there's a half-finished DAG (e.g., truncated mid-DAG), drop it
        # rather than emit a fake DAG record with a partial makespan.
        wallclock = time.time() - self._wallclock_start

        if not self.steps:
            # No steps recorded — synthesise an empty record
            return EpisodeRecord(
                episode_id              = self.episode_id,
                scheduler               = self.scheduler_name,
                num_nodes               = self.num_nodes,
                action_mode             = self.action_mode,
                total_makespan_ms       = 0.0,
                mean_makespan_normalized= float("nan"),
                peak_temp_episode       = 0.0,
                mean_temp_episode       = 0.0,
                violations_total        = 0,
                cooling_total_ms        = 0.0,
                agent_delay_total_ms    = 0.0,
                dag_completion_rate     = 0.0,
                truncated               = truncated,
                episode_return          = episode_return,
                wallclock_seconds       = wallclock,
                n_steps                 = 0,
                proc_utilisation        = [0.0] * self.num_nodes,
            )

        steps_arr = self.steps
        peak_temps = [s.peak_temp_during for s in steps_arr]
        idle_temps = [s.peak_temp_idle   for s in steps_arr]
        violations = [s.would_violate    for s in steps_arr]
        coolings   = [s.cooling_ms       for s in steps_arr]
        delays     = [s.delay_ms         for s in steps_arr]
        durations  = [s.exec_time_ms + s.cooling_ms + s.delay_ms for s in steps_arr]

        # Per-DAG normalised makespan, averaged
        if self.dags:
            norm_list = [d.makespan_normalized for d in self.dags
                         if not np.isnan(d.makespan_normalized)]
            mean_norm = float(np.mean(norm_list)) if norm_list else float("nan")
            completion_rate = (len(self.dags) / max(1, n_dags_available)
                               if n_dags_available else 1.0)
        else:
            mean_norm = float("nan")
            completion_rate = 0.0

        # Proc utilisation as fraction
        util = (self._proc_busy_count / max(1, self._total_steps)).tolist()

        return EpisodeRecord(
            episode_id              = self.episode_id,
            scheduler               = self.scheduler_name,
            num_nodes               = self.num_nodes,
            action_mode             = self.action_mode,
            total_makespan_ms       = float(np.sum(durations)),
            mean_makespan_normalized= mean_norm,
            peak_temp_episode       = float(np.max(peak_temps)),
            mean_temp_episode       = float(np.mean(idle_temps)),
            violations_total        = int(np.sum(violations)),
            cooling_total_ms        = float(np.sum(coolings)),
            agent_delay_total_ms    = float(np.sum(delays)),
            dag_completion_rate     = float(completion_rate),
            truncated               = bool(truncated),
            episode_return          = float(episode_return),
            wallclock_seconds       = float(wallclock),
            n_steps                 = len(steps_arr),
            proc_utilisation        = util,
        )


# =====================================================================
# Episode driver
# =====================================================================
def run_episode(
    env,
    scheduler,
    recorder:    EpisodeRecorder,
    max_steps:   int = 100_000,
    seed:        Optional[int] = None,
) -> EpisodeRecord:
    """Drive a single episode under ``scheduler`` and return the summary.

    The scheduler must implement :meth:`reset(env_obs, info)` and
    :meth:`schedule(env_obs, info) -> action`.  See ``baselines/base.py``.
    """
    obs, info = env.reset(seed=seed)
    scheduler.reset(obs, info)
    recorder.update_dag_meta(info)

    # n_dags_available: how many DAGs the env will try to give in this
    # episode.  Pulled from env's ``dags_per_episode`` attribute when
    # exposed, else estimated from info.
    n_dags_available = getattr(env, "dags_per_episode", 0)

    episode_return = 0.0
    truncated_flag = False
    step_idx = 0

    for step_idx in range(max_steps):
        action = scheduler.schedule(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        episode_return += float(reward)

        # Pull placement / delay from the action shape
        if isinstance(action, (int, np.integer)):
            placement = int(action)
            delay_ms  = 0.0
        else:
            arr = np.asarray(action).flatten()
            placement = int(arr[0]) if arr.size > 0 else 0
            delay_ms  = (float(next_info.get("agent_delay_ms", 0.0))
                         if arr.size > 1 else 0.0)

        # Reward channel decomposition (env emits these as flat keys)
        rp = float(next_info.get("reward_placement", 0.0))
        rd = float(next_info.get("reward_delay",     0.0))

        recorder.add_step(
            step             = step_idx,
            placement        = placement,
            delay_ms         = delay_ms,
            info             = next_info,
            reward           = float(reward),
            reward_placement = rp,
            reward_delay     = rd,
            truncated        = bool(truncated),
        )

        # If a DAG just completed, capture the next DAG's metadata
        if next_info.get("dag_done", False) and not (terminated or truncated):
            recorder.update_dag_meta(next_info)

        if terminated or truncated:
            truncated_flag = bool(truncated)
            break

        obs = next_obs
        info = next_info

    return recorder.finalise_episode(
        episode_return=episode_return,
        truncated=truncated_flag,
        n_dags_available=n_dags_available,
    )
