"""
cpo_thermal_env.py — v2 CPO Thermal-Aware DAG Scheduling Environment
=====================================================================

Stage B of the reconstruction plan.  Replaces ``cpo_thermal_env.py`` and
``cpo_thermal_env_radical.py`` with a clean, physics-aligned environment.

Key changes vs v1
-----------------
1. **Action space simplified**:  ``Discrete(9)`` (placement only).
   Delay is no longer an explicit action — the env auto-inserts cooling
   when a placement is predicted to overflow the soft thermal wall.

2. **Auto-delay (Thermal Smoothing)**:  before committing to execute a
   task on the chosen node, the env *lookahead-simulates* the execution.
   If the predicted peak temperature exceeds ``T_pen``, idle steps are
   inserted on the target node until it cools to ``T_target``.
   Both the cooling time AND a ``would_violate_without_delay`` flag are
   passed to the reward function so active cooling has positive feedback.

3. **State features (matches plan §3.2)**:

       Processor (7-d): [norm_T, dT/dt, leakage_norm, headroom_norm,
                         busy_flag, remaining_time_norm, is_ASIC]
       Task      (8-d): [workload_norm, slack_norm, rho, depth_norm,
                         in_deg_norm, out_deg_norm, critical_flag,
                         dag_progress]      ← from dag.task_features()

4. **Edge features**:

       task → task    (1-d): traffic / 100
       proc → proc    (1-d): physical thermal coupling A_ij
                              (extracted ONCE from the RC matrix —
                              this is the thermal-physics-aligned
                              message-passing innovation)
       task → proc    (2-d): [est_exec_time / 50, est_temp_rise / 20]
                              (only emitted for ready tasks × non-masked
                              processors)

5. **Reward**: delegated entirely to ``reward_shaping.compute_reward``.
   No more 5 contradictory penalty terms; all penalties bounded.

6. **Removed**: PROCHOT mid-execution looping, THERMTRIP timeout
   spinning, unbounded-clip penalties, variance/energy penalties.

Outputs (consumed downstream by train.py)
----------------------------------------
``step()`` returns a standard Gymnasium 5-tuple. The ``info`` dict
contains, in addition to the legacy telemetry keys:

    action_mask                 : np.ndarray(bool, shape=(9,))
    graph_obs                   : np.array([dict], dtype=object)
                                  (wrapper for AsyncVectorEnv safety)
    pure_compute_ms             : float, just task execution
    cooling_overhead_ms         : float, auto-delay pre-cooling time
    actual_workload_ms          : float, sum of the two
    max_temp / exec_peak_temp   : float, peak T during execution (°C)
    idle_max_temp               : float, peak T at end of step (°C)
    would_violate_without_delay : bool,  for downstream auditing
    dag_done                    : bool

The graph_obs dict schema is documented at :func:`_build_graph_obs`.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Gymnasium import (with sandbox fallback)
# ---------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
        GYM_AVAILABLE = True
    except ImportError:
        # Sandbox-only stub.  Server install has the real gymnasium.
        GYM_AVAILABLE = False

        class _Box:
            def __init__(self, low, high, shape, dtype):
                self.low, self.high = low, high
                self.shape, self.dtype = shape, dtype

        class _Discrete:
            def __init__(self, n: int):
                self.n = n
            def sample(self) -> int:
                return int(np.random.randint(self.n))
            def seed(self, *_a, **_kw):  # noqa: D401
                pass

        class _MultiDiscrete:
            def __init__(self, nvec):
                self.nvec = list(nvec)
            def sample(self):
                return np.array([int(np.random.randint(n)) for n in self.nvec],
                                dtype=np.int64)
            def seed(self, *_a, **_kw):
                pass

        class spaces:                     # type: ignore[no-redef]
            Box = _Box
            Discrete = _Discrete
            MultiDiscrete = _MultiDiscrete

        class gym:                         # type: ignore[no-redef]
            class Env:
                metadata: Dict[str, Any] = {}
                def __init__(self):
                    self.np_random = np.random.default_rng()
                def reset(self, **_kw):  pass
                def step(self, _a):       pass
                def close(self):          pass

# ---------------------------------------------------------------------
# Local v2 modules — works both as ``from cpo_thermal_v2 import ...``
# and when this file is imported with cwd inside the folder.
# ---------------------------------------------------------------------
try:
    from .rc_dynamics    import RCThermalDynamics
    from .dag_parser     import MicroserviceDAG
    from .reward_shaping import (
        compute_reward, compute_reward_channels, RewardConfig,
    )
except ImportError:  # pragma: no cover — direct-script / sandbox path
    # Fall-through covers two scenarios:
    #   1. Running ``python envs/cpo_thermal_env.py`` directly (cwd = pkg root)
    #   2. Sandbox tests that put ``envs/`` on ``sys.path``
    from rc_dynamics    import RCThermalDynamics      # type: ignore
    from dag_parser     import MicroserviceDAG        # type: ignore
    from reward_shaping import (                      # type: ignore
        compute_reward, compute_reward_channels, RewardConfig,
    )


# =====================================================================
# Constants
# =====================================================================
_NUM_PROCESSORS = 9          # 1 ASIC + 8 OE
_DT_DEFAULT     = 1e-3       # 1 ms per RC step

# Edge-extraction threshold for the proc-proc thermal coupling graph.
# Off-diagonal A[i, j] entries below this magnitude are dropped as numerical
# noise (typical |A_ii| ~ 1, meaningful |A_ij| ~ 1e-3).
_THERMAL_EDGE_THRESHOLD = 1e-4


# =====================================================================
# Environment
# =====================================================================
class CPOThermalDAGEnvV2(gym.Env):
    """v2 CPO thermal-aware DAG scheduler.  See module docstring for design."""

    metadata = {"render_modes": []}

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_obj: Optional[List[Dict[str, Any]]] = None,
        *,
        # ---- topology ----
        num_nodes:           int   = _NUM_PROCESSORS,
        dt:                  float = _DT_DEFAULT,
        # ---- thermal limits ----
        thermal_target:      float = 75.0,
        thermal_guardband:   float = 80.0,    # T_pen
        thermal_critical:    float = 85.0,    # T_crit
        mask_temp:           float = 82.0,    # action mask: T_i ≥ this → masked
        # ---- curriculum-controllable ----
        initial_temp:        Optional[float]                 = None,
        initial_temp_range:  Optional[Tuple[float, float]]   = None,
        max_dag_size:        Optional[int]                   = None,
        dags_per_episode:    int   = 50,
        # ---- auto-delay ----
        max_cooling_steps:   int   = 100,         # 100 ms cap
        precool_target_temp: Optional[float]      = None,  # default = thermal_target
        # ---- action mode (NEW: three-mode dispatch for v2 ablation) ----
        # "auto_only"  : action = Discrete(N).  Env auto-cools.        [Stage B default]
        # "agent_only" : action = MultiDiscrete([N, K_delay]).  No env auto-cool.
        #                Agent owns the entire delay decision (matches v1 framing).
        # "hybrid"     : action = MultiDiscrete([N, K_delay]).  Agent picks delay,
        #                env *still* pre-cools on top if it predicts violation
        #                (acts as a thermal safety net; small reward penalty
        #                applied because the agent under-anticipated).
        action_mode:         str   = "auto_only",
        K_delay:             int   = 5,
        # delay_fractions[k] = fraction of the task's slack used as agent-delay
        # at level k.  Default reproduces v1's 5 levels.  Length must = K_delay.
        delay_fractions:     Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        # absolute cap on agent-controlled delay, regardless of slack
        max_agent_delay_ms:  float = 50.0,
        # ---- termination semantics ----
        # "hard": exceeding ``thermal_critical`` truncates the episode
        #          immediately (default — matches real HW thermtrip
        #          behaviour and gives a strong signal during training).
        # "soft": exceeding T_crit applies a violation penalty but the
        #          episode continues, with idle steps inserted until temp
        #          returns to a safe level.  This is intended for
        #          *evaluation* in stress-testing settings where every
        #          episode otherwise truncates after ~1 task, making
        #          full-schedule metrics like makespan unobservable.
        #          Training should keep "hard" so the agent has a sharp
        #          incentive to avoid violations.
        truncate_mode:       str   = "hard",
        # When truncate_mode='soft', cool until max-T ≤ this before
        # resuming task execution.
        soft_truncate_recovery_temp: float = 75.0,
        # Cap on idle steps inserted by soft-truncate per violation event.
        max_soft_recovery_steps:     int   = 200,
        # ---- power model (CPO physics) ----
        asic_active_power:        float = 150.0,
        oe_active_power:          float = 40.0,
        oe_serialization_power:   float = 20.0,
        cpo_bandwidth_gbps:       float = 800.0,
        oe_conversion_delay_ms:   float = 0.005,
        leakage_base_power:       float = 30.0,
        leakage_beta:             float = 0.015,
        # ---- temp-rise heuristics for task→proc edge attrs ----
        temp_rise_per_ms_asic: float = 0.5,
        temp_rise_per_ms_oe:   float = 0.1,
        # ---- reward ----
        reward_config: Optional[RewardConfig] = None,
        # ---- RC matrices ----
        rc_matrix_dir: Optional[str] = None,
        rc_A: Optional[np.ndarray] = None,
        rc_B: Optional[np.ndarray] = None,
        rc_D: Optional[np.ndarray] = None,
    ):
        super().__init__()

        # -------- topology --------
        self.num_nodes = num_nodes
        self.dt        = dt

        # -------- thermal --------
        self.thermal_target    = thermal_target
        self.thermal_guardband = thermal_guardband
        self.thermal_critical  = thermal_critical
        self.mask_temp         = mask_temp

        # -------- curriculum --------
        self.initial_temp        = initial_temp
        self.initial_temp_range  = initial_temp_range
        self.max_dag_size        = max_dag_size
        self.dags_per_episode    = dags_per_episode

        # -------- auto-delay --------
        self.max_cooling_steps   = max_cooling_steps
        self.precool_target_temp = (
            precool_target_temp if precool_target_temp is not None
            else thermal_target
        )

        # -------- action mode (auto_only / agent_only / hybrid) --------
        if action_mode not in ("auto_only", "agent_only", "hybrid"):
            raise ValueError(
                f"action_mode must be one of "
                f"('auto_only', 'agent_only', 'hybrid'), got {action_mode!r}"
            )
        if K_delay != len(delay_fractions):
            raise ValueError(
                f"K_delay ({K_delay}) must equal len(delay_fractions) "
                f"({len(delay_fractions)})"
            )
        if delay_fractions[0] != 0.0:
            raise ValueError(
                "delay_fractions[0] must be 0.0 (the 'no delay' option)."
            )
        self.action_mode        = action_mode
        self.K_delay            = int(K_delay)
        self.delay_fractions    = tuple(float(x) for x in delay_fractions)
        self.max_agent_delay_ms = float(max_agent_delay_ms)
        if truncate_mode not in ("hard", "soft"):
            raise ValueError(
                f"truncate_mode must be 'hard' or 'soft', got {truncate_mode!r}"
            )
        self.truncate_mode               = truncate_mode
        self.soft_truncate_recovery_temp = float(soft_truncate_recovery_temp)
        self.max_soft_recovery_steps     = int(max_soft_recovery_steps)

        # -------- power model --------
        self.asic_active_power      = asic_active_power
        self.oe_active_power        = oe_active_power
        self.oe_serialization_power = oe_serialization_power
        self.cpo_bandwidth_gbps     = cpo_bandwidth_gbps
        self.oe_conversion_delay_ms = oe_conversion_delay_ms
        self.leakage_base_power     = leakage_base_power
        self.leakage_beta           = leakage_beta

        # -------- temp-rise heuristics --------
        self.temp_rise_per_ms_asic = temp_rise_per_ms_asic
        self.temp_rise_per_ms_oe   = temp_rise_per_ms_oe

        # -------- reward config --------
        self.reward_cfg = reward_config or RewardConfig(
            T_pen=thermal_guardband,
            T_crit=thermal_critical,
            T_target=thermal_target,
        )

        # -------- gym spaces --------
        if self.action_mode == "auto_only":
            self.action_space = spaces.Discrete(self.num_nodes)
        else:  # agent_only or hybrid
            self.action_space = spaces.MultiDiscrete(
                [self.num_nodes, self.K_delay]
            )
        self.observation_space = spaces.Box(
            low=20.0, high=150.0,
            shape=(self.num_nodes,), dtype=np.float32,
        )

        # -------- thermal engine --------
        self.thermal_engine = RCThermalDynamics(
            num_nodes=self.num_nodes, dt=self.dt,
            matrix_dir=rc_matrix_dir,
            A=rc_A, B=rc_B, D=rc_D,
        )

        # -------- proc-proc edge cache (from RC matrix A) --------
        self._cached_p2p_edges, self._cached_p2p_attrs = self._extract_thermal_edges()

        # -------- dataset --------
        self.dataset: List[Dict[str, Any]] = self._load_dataset(
            dataset_path, dataset_obj
        )

        # -------- runtime state --------
        self.current_dag:       Optional[MicroserviceDAG] = None
        self.ready_tasks:       List[Any] = []
        self.current_dag_count: int  = 0
        self.prev_temperatures: np.ndarray = np.full(
            self.num_nodes, 25.0, dtype=np.float32,
        )
        self._step_in_episode:  int = 0
        self._curriculum_stage: Optional[str] = None     # set by set_curriculum_stage()

    # -----------------------------------------------------------------
    # Dataset loading
    # -----------------------------------------------------------------
    def _load_dataset(
        self,
        dataset_path: Optional[str],
        dataset_obj:  Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if dataset_obj is not None:
            return dataset_obj

        # Resolve the path with a multi-location search (similar to
        # rc_dynamics.py's RC-matrix autodiscovery).  We try, in order:
        #   1. explicit ``dataset_path`` if absolute and exists
        #   2. explicit ``dataset_path`` resolved relative to: cwd, package
        #      root, project root (parent of package), and via env var
        #   3. default name ``alibaba_dags_v2.json`` in the same locations
        # The first hit wins.  This makes the env robust to whether the
        # user runs ``python -m`` from the project root, from inside the
        # package, or from an arbitrary cwd.
        candidates = self._enumerate_dataset_candidates(dataset_path)
        for c in candidates:
            if c is not None and os.path.exists(c):
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)

        # Nothing found — produce a maximally-helpful error message
        nice_list = "\n".join(f"    - {c}" for c in candidates if c is not None)
        raise FileNotFoundError(
            f"alibaba_dags_v2.json not found.  Searched these locations:\n"
            f"{nice_list}\n\n"
            f"Fixes (any one):\n"
            f"  • Run: python -m cpo_thermal_v2.data_pipeline.compute_dag_features \\\n"
            f"           --input  <raw alibaba_dags.json> \\\n"
            f"           --output data_pipeline/process/alibaba_dags_v2.json\n"
            f"  • Or pass an absolute path via:\n"
            f"           --override env.dataset_path=/abs/path/to/alibaba_dags_v2.json\n"
            f"  • Or set the CPO_DAGS_V2 environment variable to the file's path."
        )

    @staticmethod
    def _enumerate_dataset_candidates(
        dataset_path: Optional[str],
    ) -> List[Optional[str]]:
        """Return the ordered list of paths the env will probe for the dataset.

        Public-ish helper so the trainer can introspect / pre-flight check
        without instantiating the env.
        """
        here = os.path.dirname(os.path.abspath(__file__))         # .../envs/
        pkg_root = os.path.dirname(here)                          # .../cpo_thermal_v2/
        project_root = os.path.dirname(pkg_root)                  # parent of package

        # If the user passed a path, try it as-is first (absolute or not),
        # then in a few candidate base directories.
        candidates: List[Optional[str]] = []
        env_override = os.environ.get("CPO_DAGS_V2")
        if env_override:
            candidates.append(env_override)

        if dataset_path:
            # As-is (might be absolute or cwd-relative)
            candidates.append(dataset_path)
            # Resolved relative to common roots if it was a relative path
            if not os.path.isabs(dataset_path):
                for base in (os.getcwd(), pkg_root, project_root):
                    candidates.append(os.path.join(base, dataset_path))

        # Default-name fallbacks: ``data_pipeline/process/alibaba_dags_v2.json``
        # under a few candidate roots, then plain filename in those roots.
        default_rel = os.path.join("data_pipeline", "process",
                                    "alibaba_dags_v2.json")
        for base in (os.getcwd(), pkg_root, project_root):
            candidates.append(os.path.join(base, default_rel))
            candidates.append(os.path.join(base, "alibaba_dags_v2.json"))

        # Deduplicate while preserving order (dict trick)
        seen: dict = {}
        for c in candidates:
            if c is None:
                continue
            real = os.path.abspath(c)
            if real not in seen:
                seen[real] = real
        return list(seen.values())

    # -----------------------------------------------------------------
    # Proc-Proc thermal coupling edges  (KEY INNOVATION §3.3)
    # -----------------------------------------------------------------
    def _extract_thermal_edges(self) -> Tuple[List[List[int]], List[List[float]]]:
        """Extract physical thermal-coupling edges directly from RC matrix A.

        ``A[i, j]`` (i ≠ j) is the per-step linear transfer of T_j onto T_i,
        i.e. the *physical* heat-conduction coupling between dies i and j.
        Off-diagonals below ``_THERMAL_EDGE_THRESHOLD`` are numerical noise
        and dropped.

        The returned edge list and 1-d edge attributes give the GNN message-
        passing graph a foundation in actual thermal physics, not an ad hoc
        topology like "ASIC connects to all OE".
        """
        edges:  List[List[int]]   = []
        attrs:  List[List[float]] = []
        A = self.thermal_engine.A
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                w = float(A[i, j])
                if abs(w) > _THERMAL_EDGE_THRESHOLD:
                    edges.append([i, j])
                    attrs.append([w])
        return edges, attrs

    # =================================================================
    # gym API: reset
    # =================================================================
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            try:
                super().reset(seed=seed)
            except TypeError:
                pass
            np.random.seed(seed)
            random.seed(seed)

        # Initial temperature: fixed, range, or default ambient
        if self.initial_temp_range is not None:
            lo, hi = self.initial_temp_range
            T0 = float(np.random.uniform(lo, hi))
        elif self.initial_temp is not None:
            T0 = float(self.initial_temp)
        else:
            T0 = self.thermal_engine.ambient_temp
        T_init = np.full(self.num_nodes, T0, dtype=np.float32)
        self.thermal_engine.reset(initial_temperatures=T_init)

        self.prev_temperatures = T_init.copy()
        self.current_dag_count = 0
        self._step_in_episode  = 0
        self._load_next_dag()

        info = self._make_info(
            pure_compute_ms=0.0, cooling_overhead_ms=0.0,
            max_temp=T0, idle_max_temp=T0, exec_peak_temp=T0,
            would_violate=False, dag_done=False,
        )
        return self.thermal_engine.temperatures.copy(), info

    def _load_next_dag(self) -> None:
        """Pick a non-degenerate DAG.  Skips empties and over-large DAGs."""
        attempts = 0
        while attempts < 1000:
            attempts += 1
            d = random.choice(self.dataset)
            if self.max_dag_size is not None \
                    and d.get("num_nodes", len(d.get("nodes", {}))) > self.max_dag_size:
                continue
            self.current_dag = MicroserviceDAG(d)
            self.ready_tasks = self.current_dag.get_ready_tasks()
            if len(self.current_dag) > 0 and len(self.ready_tasks) > 0:
                return
        raise RuntimeError(
            "_load_next_dag: 1000 consecutive draws were degenerate. "
            "Check the dataset or relax max_dag_size."
        )

    # =================================================================
    # gym API: step
    # =================================================================
    def step(self, action):
        # -------- 1. Defensive: nothing to do? --------
        if not self.ready_tasks:
            self._load_next_dag()

        # -------- 2. Pop the head ready task --------
        current_task = self.ready_tasks[0]
        base_workload = self.current_dag.workload(current_task)

        # -------- 3. Parse action based on action_mode --------
        target_node, agent_delay_ms = self._parse_action(action, current_task)

        # -------- 4. Execution-time estimate --------
        total_traffic = self._task_total_traffic(current_task)
        exec_time_ms  = self._compute_exec_time(target_node, base_workload, total_traffic)

        # -------- 5a. Compute the "no-delay would-violate" signal up front --
        # Using a snapshot/restore so it doesn't perturb engine state.
        # This is the reward function's `would_violate_without_delay` flag:
        # it answers "if NO cooling at all (neither agent's nor env's) had
        # been applied, would the placement have breached T_pen?"
        snap_pre_decision = self.thermal_engine.snapshot()
        predicted_peak_no_delay = self._simulate_execution_peak(
            target_node, total_traffic, exec_time_ms,
        )
        self.thermal_engine.restore(snap_pre_decision)
        would_violate = bool(predicted_peak_no_delay > self.thermal_guardband)

        # -------- 5b. Apply agent-controlled delay (only in non-auto_only modes) --
        if agent_delay_ms > 0.0:
            self._apply_idle_cooling_steps(int(round(agent_delay_ms)))

        # -------- 5c. Auto-delay (env safety net) ----------
        # auto_only & hybrid: env may insert additional cooling.
        # agent_only:        no env intervention; agent owns the delay decision.
        if self.action_mode == "agent_only":
            env_cooling_ms = 0.0
        else:
            env_cooling_ms, _ = self._maybe_precool(
                target_node, total_traffic, exec_time_ms,
            )

        cooling_used = float(agent_delay_ms) + float(env_cooling_ms)

        # -------- 6. Snapshot pre-execution (for prev_temperatures bookkeeping)
        pre_exec_temps = self.thermal_engine.snapshot()

        # -------- 7. Actually execute the task --------
        max_temp_during = self._execute_task(target_node, total_traffic, exec_time_ms)
        final_temps     = self.thermal_engine.temperatures.copy()

        # -------- 8. Truncate or recover on critical-temperature breach --------
        # Two modes:
        #   hard  (training default): a single step over T_crit ends the
        #         episode immediately.  Strong signal that the agent must
        #         not violate.
        #   soft  (evaluation): apply the truncation as a violation flag
        #         (still counted in metrics), but force-cool the system
        #         until max(T) <= soft_truncate_recovery_temp, then keep
        #         going.  Lets us observe full-schedule performance even
        #         in stress-test settings where hard mode would cause
        #         100% early termination.
        violation_event = bool(max_temp_during > self.thermal_critical)
        soft_recovery_ms = 0.0
        if violation_event and self.truncate_mode == "soft":
            # Force idle cooling until safe, capped by max_soft_recovery_steps
            recovery_steps = 0
            while recovery_steps < self.max_soft_recovery_steps:
                if (float(np.max(self.thermal_engine.temperatures))
                        <= self.soft_truncate_recovery_temp):
                    break
                self.thermal_engine.step(self._compute_power(-1, 0.0))
                recovery_steps += 1
            soft_recovery_ms = float(recovery_steps)
            final_temps = self.thermal_engine.temperatures.copy()
            truncated = False    # episode continues
        else:
            truncated = violation_event   # hard mode: violation → truncate

        # -------- 9. Update DAG & ready set --------
        self.current_dag.mark_completed(current_task)
        self.ready_tasks = self.current_dag.get_ready_tasks()

        dag_done = self.current_dag.is_done
        episode_done = False

        if dag_done and not truncated:
            self.current_dag_count += 1
            if self.current_dag_count >= self.dags_per_episode:
                episode_done = True
            else:
                self._load_next_dag()

        # -------- 10. Reward --------
        # `cooling_used` includes BOTH agent's delay and env's auto-cool, so
        # the reward function correctly credits whichever party did the work.
        # In hybrid mode, if env had to step in (env_cooling_ms > 0), the
        # reward already pays a small cool_waste_pen for the env's portion
        # if it wasn't strictly needed; this is the "agent under-anticipated"
        # penalty that pushes hybrid agents toward proactive delays.
        reward = compute_reward(
            exec_time=exec_time_ms + cooling_used,
            base_workload=base_workload,
            max_temp_during=max_temp_during,
            final_temps=final_temps,
            target_node=target_node,
            cooling_used=cooling_used,
            would_violate_without_delay=would_violate,
            dag_done=dag_done,
            episode_done=episode_done,
            truncated=truncated,
            cfg=self.reward_cfg,
        )

        # Dual-advantage channeled reward (consumed by the dual-critic PPO
        # in agent_only / hybrid modes; ignored by single-head auto_only
        # which only uses the scalar reward).
        reward_channels = compute_reward_channels(
            exec_time=exec_time_ms + cooling_used,
            base_workload=base_workload,
            max_temp_during=max_temp_during,
            final_temps=final_temps,
            target_node=target_node,
            cooling_used=cooling_used,
            agent_delay_ms=agent_delay_ms,
            env_cooling_ms=env_cooling_ms,
            would_violate_without_delay=would_violate,
            dag_done=dag_done,
            episode_done=episode_done,
            truncated=truncated,
            cfg=self.reward_cfg,
        )

        # -------- 11. Bookkeeping --------
        self.prev_temperatures = pre_exec_temps  # used in proc feature dT/dt
        self._step_in_episode += 1

        # -------- 12. Build info / return --------
        # Roll the soft-recovery cooling time into the env-cooling
        # accounting so it shows up in evaluation metrics correctly.
        info_cooling_overhead = cooling_used + soft_recovery_ms
        info = self._make_info(
            pure_compute_ms     = exec_time_ms,
            cooling_overhead_ms = info_cooling_overhead,
            agent_delay_ms      = float(agent_delay_ms),
            env_cooling_ms      = float(env_cooling_ms) + soft_recovery_ms,
            max_temp            = float(max_temp_during),
            exec_peak_temp      = float(max_temp_during),
            idle_max_temp       = float(np.max(final_temps)),
            would_violate       = bool(would_violate),
            dag_done            = bool(dag_done),
            reward_channels     = reward_channels,
        )
        terminated = bool(episode_done)
        return (
            self.thermal_engine.temperatures.copy(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    # -----------------------------------------------------------------
    # Action parsing (mode-aware)
    # -----------------------------------------------------------------
    def _parse_action(self, action, current_task) -> Tuple[int, float]:
        """Convert a raw action into ``(target_node, agent_delay_ms)``.

        Action shapes by mode:
            auto_only   : int                    -> (target_node, 0.0)
            agent_only  : array-like length 2    -> (target_node, agent_delay_ms)
            hybrid      : array-like length 2    -> (target_node, agent_delay_ms)

        ``agent_delay_ms`` = ``delay_fractions[delay_idx] * slack(current_task)``
        capped at ``max_agent_delay_ms``.

        Notes
        -----
        For tasks on the critical path (slack == 0), all five delay levels
        collapse to 0 ms — the agent simply has no slack to spend.  This
        is the v1 semantic, deliberately preserved so the agent_only mode
        reproduces the v1 setup faithfully.  In hybrid mode, the env's
        auto-cool still kicks in for these tasks.

        Implementation note
        -------------------
        We accept any array-like or scalar input (Python int, numpy
        scalar, 0-d ndarray, 1-d ndarray of length-1 or length-2,
        torch.Tensor) and coerce to plain ints up front via
        ``np.asarray(...).flatten()``.  This is necessary because
        gymnasium's SyncVectorEnv hands down per-env actions as 0-d
        ndarrays in some numpy versions, on which ``int(arr)`` raises
        TypeError.  Coercing through asarray + flatten handles every
        shape uniformly.
        """
        # Coerce to a 1-d int64 ndarray with at least 1 element
        a_arr = np.asarray(action).flatten().astype(np.int64, casting="unsafe")

        if self.action_mode == "auto_only":
            if a_arr.size == 0:
                raise ValueError("auto_only got empty action array")
            target_node = int(a_arr[0]) % self.num_nodes
            return target_node, 0.0

        # agent_only or hybrid: must have at least 2 entries
        if a_arr.size < 2:
            raise ValueError(
                f"In action_mode={self.action_mode!r}, action must be a "
                f"length-2 array [proc_idx, delay_idx]; got {action!r} "
                f"(flattened to size {a_arr.size})"
            )
        target_node = int(a_arr[0]) % self.num_nodes
        delay_idx   = int(a_arr[1])
        delay_idx = max(0, min(self.K_delay - 1, delay_idx))

        slack = self.current_dag.slack(current_task)
        delay_frac = self.delay_fractions[delay_idx]
        agent_delay_ms = min(delay_frac * float(slack), self.max_agent_delay_ms)
        return target_node, float(agent_delay_ms)

    def _apply_idle_cooling_steps(self, n_steps: int) -> None:
        """Run ``n_steps`` ms of idle (leakage-only) RC dynamics."""
        if n_steps <= 0:
            return
        idle_power = self._compute_power(-1, 0.0)
        for _ in range(int(n_steps)):
            self.thermal_engine.step(idle_power)

    # =================================================================
    # Power / execution-time helpers
    # =================================================================
    def _compute_power(self, target_node: int, _total_traffic: float) -> np.ndarray:
        """Per-node instantaneous power array (W).

        ``target_node = -1`` ⇒ all-idle (leakage only).
        """
        T = self.thermal_engine.temperatures
        # Clamp temperature for the leakage exponent so np.exp() can't
        # overflow if a transient simulator drift sends T → ∞.  In healthy
        # training this clamp never fires (temps stay in [25, T_crit]);
        # it's pure numerical-safety insurance.
        T_clamped = np.clip(T, -50.0, 200.0)
        # Leakage everywhere
        P = self.leakage_base_power * np.exp(self.leakage_beta * (T_clamped - 25.0))
        P = P.astype(np.float32, copy=True)
        if target_node < 0:
            return P
        if target_node == 0:                # ASIC active
            P[0] += self.asic_active_power
        else:                               # OE active + ASIC serialisation
            P[0]            += self.oe_serialization_power
            P[target_node]  += self.oe_active_power
        return P

    def _compute_exec_time(
        self, target_node: int, base_workload: float, total_traffic: float
    ) -> float:
        if target_node == 0:
            return float(base_workload)
        serialisation_ms = (total_traffic / self.cpo_bandwidth_gbps) * 1000.0
        return float(base_workload + serialisation_ms + self.oe_conversion_delay_ms)

    def _task_total_traffic(self, task_id: Any) -> float:
        cache = self.current_dag.traffic_cache
        in_t  = sum(cache.get((p, task_id), 0.0)
                    for p in self.current_dag.predecessors_cache.get(task_id, []))
        out_t = sum(cache.get((task_id, s), 0.0)
                    for s in self.current_dag.successors_cache.get(task_id, []))
        return float(in_t + out_t)

    # =================================================================
    # Auto-delay
    # =================================================================
    def _maybe_precool(
        self,
        target_node: int,
        total_traffic: float,
        exec_time_ms: float,
    ) -> Tuple[float, bool]:
        """Lookahead-based auto-delay.

        1. Snapshot the current thermal state.
        2. Simulate the planned execution (does NOT commit).
        3. Restore.  If the predicted peak ``≤ thermal_guardband``: no cooling.
        4. Otherwise insert idle steps until the *target* node falls below
           ``precool_target_temp`` (or ``max_cooling_steps`` is reached).

        Returns
        -------
        cooling_steps : float (ms)
        would_violate : bool, True iff lookahead predicted ``> T_pen``.
        """
        snap = self.thermal_engine.snapshot()
        try:
            predicted_peak = self._simulate_execution_peak(
                target_node, total_traffic, exec_time_ms,
            )
        finally:
            self.thermal_engine.restore(snap)

        would_violate_predicted = bool(predicted_peak > self.thermal_guardband)
        if not would_violate_predicted:
            return 0.0, False

        # Pre-cool with idle power until target node is safe
        cooling_steps = 0
        while cooling_steps < self.max_cooling_steps:
            if self.thermal_engine.temperatures[target_node] <= self.precool_target_temp:
                break
            self.thermal_engine.step(self._compute_power(-1, 0.0))
            cooling_steps += 1

        # Honest accounting: only claim "cooling averted a violation" if we
        # actually cooled.  When the target node is already at/below the
        # precool target temperature, no idle steps run and we have made
        # no scheduling decision to credit — let the truncation path handle it.
        if cooling_steps == 0:
            return 0.0, False

        return float(cooling_steps), True

    def _simulate_execution_peak(
        self, target_node: int, total_traffic: float, exec_time_ms: float
    ) -> float:
        """Run the RC dynamics for ``exec_time_ms`` ms; return the peak max-T.

        Caller is responsible for snapshot/restore — this method *does*
        mutate ``thermal_engine.temperatures``.

        For numerical safety, the simulation early-exits if the system
        diverges past 200 °C (well above ``T_crit``), since at that point
        the placement is already certain to truncate and continuing only
        produces NaN-prone exponentials.

        The returned peak is clamped to ``thermal_critical + 10`` so
        doomed-trajectory metrics report physically meaningful numbers
        (e.g. 95 °C rather than 600 °C); the truncation flag in the
        caller still fires identically for any value > T_crit.  Without
        this clamp, periods of RC overshoot during a single step (where
        the discrete-time integration briefly produces non-physical
        spikes before the next step normalises) would show up in
        ``peak_temp`` metrics as 200-600 °C, which makes published
        plots look ridiculous.
        """
        steps = max(1, int(np.ceil(exec_time_ms)))
        max_t = float(np.max(self.thermal_engine.temperatures))
        for _ in range(steps):
            P = self._compute_power(target_node, total_traffic)
            T = self.thermal_engine.step(P)
            cur = float(np.max(T))
            if cur > max_t:
                max_t = cur
            if max_t > 200.0:           # already doomed; stop wasting cycles
                break
        # Clamp to a physically meaningful upper bound so metrics make
        # sense.  Truncation logic in the caller (max_temp > T_crit) is
        # unaffected: T_crit + 10 still triggers truncate.
        max_t = min(max_t, self.thermal_critical + 10.0)
        return max_t

    def _execute_task(
        self, target_node: int, total_traffic: float, exec_time_ms: float
    ) -> float:
        """Commit the task execution (mutates engine state).  Returns peak T."""
        return self._simulate_execution_peak(
            target_node, total_traffic, exec_time_ms,
        )

    # =================================================================
    # Action mask
    # =================================================================
    def _action_mask(self) -> np.ndarray:
        """Bool mask of length ``num_nodes``: True ⇒ legal placement."""
        mask = self.thermal_engine.temperatures < self.mask_temp
        if not mask.any():
            mask[int(np.argmin(self.thermal_engine.temperatures))] = True
        return mask.astype(bool, copy=False)

    # =================================================================
    # Feature extractors
    # =================================================================
    def _proc_features(self, i: int) -> List[float]:
        """7-D processor feature vector consumed by the v2 GNN encoder.

        Defensive against numerical drift: temperatures are clamped to a
        reasonable range before normalisation, and the leakage exponent
        is computed in clamped space so it can never overflow.  In
        well-behaved training, these clamps never fire (temps stay in
        [25, T_crit]); they exist purely to prevent a single bad
        simulator step from poisoning the encoder with NaN/Inf.
        """
        T_max = 200.0   # well above any physically plausible chip temp
        T_i      = float(np.clip(self.thermal_engine.temperatures[i], -50.0, T_max))
        T_prev_i = float(np.clip(self.prev_temperatures[i],          -50.0, T_max))
        # Leakage exponent capped so np.exp() can never overflow float32.
        # exp(0.015 * (200-25)) = exp(2.625) ≈ 13.8, finite by design.
        leakage  = self.leakage_base_power * float(np.exp(
            self.leakage_beta * (T_i - 25.0)
        ))
        feats = [
            (T_i - 25.0) / 60.0,                                  # 0 norm temp
            float(np.clip((T_i - T_prev_i) / 1.0, -1.0, 1.0)),    # 1 dT/dt
            float(leakage) / 200.0,                               # 2 leakage
            max(0.0, (self.thermal_critical - T_i)) / 60.0,       # 3 headroom
            0.0,                                                  # 4 busy_flag (reserved)
            0.0,                                                  # 5 remaining time (reserved)
            1.0 if i == 0 else 0.0,                               # 6 is_ASIC
        ]
        # Final NaN/Inf scrub — should be a no-op given the clamps above,
        # but cheap and bulletproof.  The encoder's GATv2Conv silently
        # propagates NaN through softmax, so it's worth one extra line
        # to guarantee clean inputs.
        return [float(np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0))
                for x in feats]

    def _task_proc_edge_attr(
        self, task_id: Any, processor_idx: int
    ) -> List[float]:
        """[est_exec_time / 50, est_temp_rise / 20]."""
        wl = self.current_dag.workload(task_id)
        if processor_idx == 0:
            est_time = wl
            est_rise = self.temp_rise_per_ms_asic * wl
        else:
            traffic = self._task_total_traffic(task_id)
            est_time = (
                wl
                + (traffic / self.cpo_bandwidth_gbps) * 1000.0
                + self.oe_conversion_delay_ms
            )
            est_rise = self.temp_rise_per_ms_oe * wl
        return [
            float(est_time) / 50.0,
            float(est_rise) / 20.0,
        ]

    # =================================================================
    # Heterograph dict (consumed by train.py & evaluate.py)
    # =================================================================
    def _build_graph_obs(self) -> Dict[str, Any]:
        """Return the heterograph state as plain Python — multiprocess-safe.

        Schema::

            {
                proc_x          : [[7-d float] * num_nodes],
                task_x          : [[8-d float] * num_uncompleted],
                edges_t2t       : [[u_idx, v_idx], ...],
                edges_t2t_attr  : [[traffic_norm], ...],
                edges_p2p       : [[i, j], ...],
                edges_p2p_attr  : [[A_ij], ...],
                edges_t2p       : [[task_idx, proc_idx], ...],
                edges_t2p_attr  : [[est_time_norm, est_rise_norm], ...],
                current_task_idx: int,
                task_id_order   : [task_id, ...]   # for downstream debugging
                num_uncompleted : int,
            }
        """
        # ---- processor side ----
        proc_x = [self._proc_features(i) for i in range(self.num_nodes)]

        # ---- task side: only uncompleted tasks ----
        completed_set = self.current_dag._completed
        uncompleted = [
            t for t in self.current_dag.graph.nodes
            if t not in completed_set
        ]
        task_id_to_idx = {t: i for i, t in enumerate(uncompleted)}
        task_x = [self.current_dag.task_features(t) for t in uncompleted]

        # ---- task→task edges (within uncompleted set) ----
        edges_t2t:      List[List[int]]   = []
        edges_t2t_attr: List[List[float]] = []
        for u, v in self.current_dag.graph.edges:
            if u in task_id_to_idx and v in task_id_to_idx:
                edges_t2t.append([task_id_to_idx[u], task_id_to_idx[v]])
                tr = self.current_dag.traffic_cache.get((u, v), 0.0)
                edges_t2t_attr.append([float(tr) / 100.0])

        # ---- proc↔proc edges (precomputed at init) ----
        edges_p2p      = [list(e) for e in self._cached_p2p_edges]
        edges_p2p_attr = [list(a) for a in self._cached_p2p_attrs]

        # ---- task→proc edges (only ready × non-masked) ----
        mask = self._action_mask()
        edges_t2p:      List[List[int]]   = []
        edges_t2p_attr: List[List[float]] = []
        for ready_t in self.ready_tasks:
            if ready_t not in task_id_to_idx:
                continue
            t_idx = task_id_to_idx[ready_t]
            for p in range(self.num_nodes):
                if mask[p]:
                    edges_t2p.append([t_idx, p])
                    edges_t2p_attr.append(self._task_proc_edge_attr(ready_t, p))

        # ---- current task pointer ----
        current_task = self.ready_tasks[0] if self.ready_tasks else None
        current_task_idx = (
            task_id_to_idx[current_task] if current_task in task_id_to_idx
            else 0
        )

        return {
            "proc_x":           proc_x,
            "task_x":           task_x,
            "edges_t2t":        edges_t2t,
            "edges_t2t_attr":   edges_t2t_attr,
            "edges_p2p":        edges_p2p,
            "edges_p2p_attr":   edges_p2p_attr,
            "edges_t2p":        edges_t2p,
            "edges_t2p_attr":   edges_t2p_attr,
            "current_task_idx": int(current_task_idx),
            "task_id_order":    [str(t) for t in uncompleted],
            "num_uncompleted":  len(uncompleted),
        }

    # =================================================================
    # info dict assembly  (keeps legacy keys for train.py compatibility)
    # =================================================================
    def _make_info(
        self,
        *,
        pure_compute_ms:     float,
        cooling_overhead_ms: float,
        max_temp:            float,
        exec_peak_temp:      float,
        idle_max_temp:       float,
        would_violate:       bool,
        dag_done:            bool,
        agent_delay_ms:      float = 0.0,
        env_cooling_ms:      float = 0.0,
        reward_channels:     Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        action_mask = self._action_mask()
        graph_obs   = self._build_graph_obs()
        # Wrap the dict for AsyncVectorEnv safety (each env returns a 1-element
        # object array containing its own graph dict).
        graph_wrapper = np.empty(1, dtype=object)
        graph_wrapper[0] = graph_obs

        total_ms = float(pure_compute_ms + cooling_overhead_ms)

        # Reset-path fallback: if step hasn't happened yet, synthesise an
        # all-zero reward_channels dict.  This keeps the per-key types
        # consistent across reset() and step() (both emit float scalars
        # for the three reward_* keys), which gymnasium's vector info
        # broadcaster requires.
        rc = (reward_channels if reward_channels is not None
              else {"placement": 0.0, "delay": 0.0, "total": 0.0})

        return {
            "action_mask":                 action_mask,
            "graph_obs":                   graph_wrapper,
            "ready_tasks":                 list(self.ready_tasks),
            "max_temp":                    float(max_temp),
            "exec_peak_temp":               float(exec_peak_temp),
            "idle_max_temp":               float(idle_max_temp),
            "pure_compute_ms":             float(pure_compute_ms),
            "cooling_overhead_ms":         float(cooling_overhead_ms),
            # Decomposition of cooling: agent's slack-based delay vs env's
            # auto-cool.  In auto_only mode agent_delay_ms is always 0; in
            # agent_only mode env_cooling_ms is always 0.  In hybrid mode
            # both can be positive.  Useful for TensorBoard ablation.
            "agent_delay_ms":              float(agent_delay_ms),
            "env_cooling_ms":              float(env_cooling_ms),
            "actual_workload_ms":          total_ms,
            "would_violate_without_delay": bool(would_violate),
            "dag_done":                    bool(dag_done),
            "action_mode":                 self.action_mode,
            # Dual-advantage channels for PPO trainer.
            #
            # Emitted as FLAT keys (reward_placement / reward_delay /
            # reward_total) rather than a nested dict, because
            # gymnasium 1.x's vector-env info broadcaster has a known bug
            # when a nested dict's type drifts across steps:
            # ``vector_infos.get(...)`` is called on what's been silently
            # cast to an ndarray, raising ``AttributeError: 'numpy.ndarray'
            # object has no attribute 'get'``.  Flat scalars dodge the
            # whole recursion.  The trainer reassembles them into a dict
            # in `_extract_reward_channels`.
            "reward_placement":            float(rc["placement"]),
            "reward_delay":                float(rc["delay"]),
            "reward_total":                float(rc["total"]),
        }

    # =================================================================
    # Misc
    # =================================================================
    def set_curriculum_stage(
        self,
        *,
        initial_temp_range: Optional[Tuple[float, float]] = None,
        max_dag_size:       Optional[int]                 = None,
        stage_name:         Optional[str]                 = None,
    ) -> None:
        """Hot-update curriculum-controlled parameters.

        Called by ``training/curriculum.py`` on stage transitions.  The
        change takes effect from the *next* call to ``reset()``; ongoing
        episodes (initial_temp from a previous stage) are NOT interrupted
        — that would corrupt the rollout buffer.

        Parameters
        ----------
        initial_temp_range
            New (lo, hi) range for ``reset()``-time initial chip temperature.
        max_dag_size
            New upper-bound on |V| for ``_load_next_dag()``-time DAG sampling.
            ``None`` means no upper bound.
        stage_name
            Optional human-readable tag, stored on ``self._curriculum_stage``
            so the trainer can include it in TensorBoard scalars.
        """
        if initial_temp_range is not None:
            lo, hi = initial_temp_range
            self.initial_temp_range = (float(lo), float(hi))
        self.max_dag_size = max_dag_size
        self._curriculum_stage = stage_name

    def render(self):  # noqa: D401
        T = self.thermal_engine.temperatures
        print(f"step={self._step_in_episode:4d} | T = "
              + " ".join(f"{t:5.1f}" for t in T)
              + f"  | dag_progress = {self.current_dag.completion_progress:.0%}")

    def close(self):
        return None


# Convenience alias matching the v1 export name pattern
CPOThermalDAGEnv = CPOThermalDAGEnvV2
