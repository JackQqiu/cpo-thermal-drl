"""
reward_shaping.py
=================

Independent, bounded reward function for the v2 CPO thermal-scheduling env.

Design principles (§2 of the reconstruction plan)
-------------------------------------------------
1. **All penalties are bounded** — via ``tanh``, ``clip`` or fixed scalars.
   No more "if the agent picks a 500 000-step delay, give it a 5 000-point
   penalty" landmines.

2. **Soft thermal wall sits ABOVE the target zone**. The agent has
   headroom to *learn* before any thermal penalty fires.  Old version put
   the soft wall at 75 °C while *also* asking the agent to push the chip
   toward 80 °C; that was a contradiction.

3. **One signal per concept** — no two penalties measuring the same
   physical thing.  Removed: ``variance_penalty`` (CPO physics says ASIC
   *must* be hotter than OE), ``energy_penalty`` (highly correlated with
   makespan), ``prochot_penalty`` (already covered by ``truncate``).

4. **Reward base and penalties are on the same order of magnitude** (~10),
   so PPO advantage estimates don't blow up.

5. **Active cooling has a positive learning signal** when it actually
   averted a violation — the old reward had no positive feedback for the
   thermal-smoothing action, so the agent never learned to use it.

Public API
----------
``compute_reward(...)``   -> float        the main reward
``decompose_reward(...)`` -> dict[str,float]  same, broken out for logging
``RewardConfig``          dataclass        all hyper-parameters in one place

This module is deliberately stateless and ``numpy``-only, so it can be
unit-tested without Gym, PyTorch, or the env.

Self-test
---------
    python reward_shaping.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# =====================================================================
# Hyper-parameters
# =====================================================================
@dataclass(frozen=True)
class RewardConfig:
    """Hyper-parameters of the v2 reward function (all temps in °C).

    Defaults match the reconstruction plan; overridable per-curriculum-stage.
    """
    # Thermal thresholds
    T_pen:           float = 80.0    # soft wall (penalty kicks in above)
    T_crit:          float = 85.0    # hard wall (large fixed penalty)
    T_target:        float = 75.0    # informational; cool-node bonus uses mean

    # Penalty caps / magnitudes
    truncate_pen:    float = 20.0    # thermtrip — episode aborted
    soft_wall_max:   float = 10.0    # cap on the soft-wall quadratic
    soft_wall_scale: float = 5.0     # k in  k * ((T - T_pen) / 5)^2
    hard_wall_pen:   float = 30.0    # one-shot at T > T_crit

    # Progress / shaping
    base_step_reward:  float = 1.0   # +1 per completed task
    # NEW (v2.2): penalise cooling/idle time directly.  Replaces the
    # broken tanh(exec/base - 1) formulation which gave ~0 signal in
    # this environment.  At 0.05 ms^-1, a 100ms cooling event costs 5
    # reward (≈ dag_done_bonus), and a 1000ms cumulative cooling spree
    # over an episode costs 50 reward — large enough to dominate the
    # per-step base reward but not the truncate penalty.
    cooling_time_w:    float = 0.05  # reward per ms of cooling/delay
    # DEPRECATED: kept for backward compatibility but no longer used.
    # The original formula tanh((exec_time/base_workload)-1) yielded
    # near-zero signal because per-task overhead was ~10⁻⁴ × base.
    time_overhead_w:   float = 0.0   # disabled
    cool_avoid_bonus:  float = 2.0   # cooling actually averted a violation
    cool_waste_pen:    float = 0.5   # cooling not needed
    cool_node_bonus:   float = 0.3   # weak guidance toward sub-mean nodes
    cool_node_margin:  float = 2.0   # node temp must be > this much below mean

    # Phase bonuses
    dag_done_bonus:     float = 5.0
    episode_done_bonus: float = 20.0

    # Dual-advantage shaping (used only by compute_reward_channels)
    under_anticipate_pen: float = 1.0   # agent picked 0 delay but env had to cool
    # NEW (v2.1): positive reward for the agent's delay actually
    # *anticipating* a thermal event — i.e. the agent inserted delay
    # AND the env didn't need to step in afterwards.  This creates a
    # Stage-2-specific learning signal: in auto_only mode (Stage 1),
    # agent_delay_ms is always 0 by construction so this bonus is
    # never earned, meaning Stage 2's hybrid contribution is well
    # differentiated from Stage 1's auto-cool-fallback baseline.
    #
    # v2.2: gated on would_violate_without_delay (see
    # compute_reward_channels for the trigger logic). Magnitude
    # unchanged.
    anticipation_bonus:   float = 1.5


_DEFAULT_CFG = RewardConfig()


# =====================================================================
# Core reward
# =====================================================================
def compute_reward(
    *,
    exec_time:                    float,
    base_workload:                float,
    max_temp_during:              float,
    final_temps:                  np.ndarray,
    target_node:                  int,
    cooling_used:                 float,
    would_violate_without_delay:  bool,
    dag_done:                     bool,
    episode_done:                 bool,
    truncated:                    bool,
    cfg: RewardConfig = _DEFAULT_CFG,
) -> float:
    """Return a scalar reward for one ``env.step``.

    All arguments are keyword-only so call-sites stay self-documenting.

    Parameters
    ----------
    exec_time
        Total wall time of the step (compute + serialisation + cooling), ms.
    base_workload
        Nominal task workload (best-case execution time on ASIC), ms.
    max_temp_during
        Peak temperature observed *during* the step, °C.
    final_temps
        Per-node temperature array at the *end* of the step, °C, shape ``(N,)``.
    target_node
        Index of the node the task was placed on (for the cool-node bonus).
    cooling_used
        ms of cooling time inserted before / during this task (auto-delay).
    would_violate_without_delay
        ``True`` iff cooling actually averted a ``T > T_pen`` excursion.
    dag_done
        ``True`` iff this step finished a whole DAG.
    episode_done
        ``True`` iff this step finished the whole episode.
    truncated
        ``True`` iff the env aborted via thermtrip / timeout.
    cfg
        Hyper-parameters; default = :data:`_DEFAULT_CFG`.

    Returns
    -------
    float
        Reward for this step, approximately in ``[-42, +29]``.
    """
    # 1. Termination override — bounded large penalty
    if truncated:
        return -float(cfg.truncate_pen)

    r = float(cfg.base_step_reward)

    # 2. Makespan penalty — saturating per-step.
    #
    #    The original formulation used  tanh((exec_time/base_workload) - 1)
    #    which collapsed to ≈ 0 in this environment because the
    #    serialisation / conversion overhead is ~6 microseconds vs
    #    ~30ms task workloads.  So neither weight=2 nor weight=8
    #    produced a meaningful signal, and agents trained with this
    #    term in the loss had no incentive to keep schedules short.
    #
    #    Replacement: penalise the cooling component of step time
    #    directly.  ``cooling_used`` is the time the env or agent spent
    #    NOT doing compute (auto-cool + agent delay).  Each ms of it
    #    costs the agent `cooling_time_w` reward — small enough to not
    #    overwhelm the truncate signal, but large enough to be felt
    #    when summed over a 100-200 step episode.
    #
    #    NOTE: this penalises BOTH agent_delay and env_cooling, which
    #    is what we want — the agent should learn to avoid both.  The
    #    delay-channel routing in compute_reward_channels then gives
    #    agent-controlled delay its own positive signal via
    #    `anticipation_bonus` when delay is well-targeted.
    r -= cfg.cooling_time_w * float(cooling_used)

    # 3. Soft thermal wall — quadratic, capped
    if max_temp_during > cfg.T_pen:
        soft_pen = cfg.soft_wall_scale * ((max_temp_during - cfg.T_pen) / 5.0) ** 2
        r -= float(min(cfg.soft_wall_max, soft_pen))

    # 4. Hard thermal wall — one-shot, NOT accumulated
    if max_temp_during > cfg.T_crit:
        r -= float(cfg.hard_wall_pen)

    # 5. Active cooling — positive when it actually helps
    if cooling_used > 0:
        if would_violate_without_delay:
            r += cfg.cool_avoid_bonus
        else:
            r -= cfg.cool_waste_pen

    # 6. Cool-node guidance — weak shaping toward sub-mean nodes
    if final_temps is not None and len(final_temps) > 0 \
            and 0 <= target_node < len(final_temps):
        mean_t = float(np.mean(final_temps))
        if final_temps[target_node] < mean_t - cfg.cool_node_margin:
            r += cfg.cool_node_bonus

    # 7. Phase bonuses
    if dag_done:
        r += cfg.dag_done_bonus
    if episode_done:
        r += cfg.episode_done_bonus

    return float(r)


# =====================================================================
# Decomposed version (for TensorBoard scalars)
# =====================================================================
def decompose_reward(
    *,
    exec_time:                    float,
    base_workload:                float,
    max_temp_during:              float,
    final_temps:                  np.ndarray,
    target_node:                  int,
    cooling_used:                 float,
    would_violate_without_delay:  bool,
    dag_done:                     bool,
    episode_done:                 bool,
    truncated:                    bool,
    cfg: RewardConfig = _DEFAULT_CFG,
) -> Dict[str, float]:
    """Same as :func:`compute_reward` but returns a per-component breakdown.

    Useful for TensorBoard logging — knowing which term dominates explains
    most of the policy's behaviour.
    """
    components: Dict[str, float] = {
        "base":          0.0, "time":          0.0,
        "soft_wall":     0.0, "hard_wall":     0.0,
        "cool_bonus":    0.0, "cool_waste":    0.0,
        "cool_node":     0.0,
        "dag_bonus":     0.0, "episode_bonus": 0.0,
        "truncate":      0.0, "total":         0.0,
    }

    if truncated:
        components["truncate"] = -float(cfg.truncate_pen)
        components["total"]    = components["truncate"]
        return components

    components["base"] = float(cfg.base_step_reward)

    # Replaced tanh(exec/base-1) with linear cooling-time penalty.
    components["time"] = -cfg.cooling_time_w * float(cooling_used)

    if max_temp_during > cfg.T_pen:
        soft_pen = cfg.soft_wall_scale * ((max_temp_during - cfg.T_pen) / 5.0) ** 2
        components["soft_wall"] = -float(min(cfg.soft_wall_max, soft_pen))

    if max_temp_during > cfg.T_crit:
        components["hard_wall"] = -float(cfg.hard_wall_pen)

    if cooling_used > 0:
        if would_violate_without_delay:
            components["cool_bonus"] = float(cfg.cool_avoid_bonus)
        else:
            components["cool_waste"] = -float(cfg.cool_waste_pen)

    if final_temps is not None and len(final_temps) > 0 \
            and 0 <= target_node < len(final_temps):
        mean_t = float(np.mean(final_temps))
        if final_temps[target_node] < mean_t - cfg.cool_node_margin:
            components["cool_node"] = float(cfg.cool_node_bonus)

    if dag_done:
        components["dag_bonus"] = float(cfg.dag_done_bonus)
    if episode_done:
        components["episode_bonus"] = float(cfg.episode_done_bonus)

    components["total"] = sum(v for k, v in components.items() if k != "total")
    return components


# =====================================================================
# Channeled reward (for dual-advantage credit assignment)
# =====================================================================
def compute_reward_channels(
    *,
    exec_time:                    float,
    base_workload:                float,
    max_temp_during:              float,
    final_temps:                  np.ndarray,
    target_node:                  int,
    cooling_used:                 float,
    agent_delay_ms:               float,
    env_cooling_ms:               float,
    would_violate_without_delay:  bool,
    dag_done:                     bool,
    episode_done:                 bool,
    truncated:                    bool,
    cfg: RewardConfig = _DEFAULT_CFG,
) -> Dict[str, float]:
    """Return a per-channel reward decomposition for **dual-advantage PPO**.

    Returns a dict with keys ``placement``, ``delay``, ``total``.  The
    sum of ``placement`` and ``delay`` equals ``compute_reward``'s scalar.

    Channel routing rationale
    -------------------------
    Each reward term is routed to the channel whose head **directly
    controls** the term:

        Term                         |   Channel    |   Why
        -----------------------------|--------------|----------------------------------
        base_step_reward             |  placement   |   placement caused a task to
                                     |              |   complete -> credit it
        time_overhead penalty        |  placement   |   driven by which proc was chosen
        soft / hard thermal walls    |  placement   |   peak temp depends on placement
        cool_node_bonus              |  placement   |   placement-conditional shaping
        cool_avoid_bonus             |  delay       |   the delay decision averted heat
        cool_waste_pen               |  delay       |   the delay decision wasted time
        agent-under-anticipated -1   |  delay       |   see "Under" section below
        agent-anticipated +1.5       |  delay       |   v2.1; v2.2 gated, see "Anticipation" below
        dag_done / episode_done      |  split 50/50 |   both heads contributed
        truncate                     |  split 50/50 |   both heads share blame

    "Agent-under-anticipated" signal (the Q2-B mitigation)
    ------------------------------------------------------
    In hybrid mode, if ``agent_delay_ms == 0 and env_cooling_ms > 0``,
    the agent's delay head implicitly chose "no delay" while the env
    had to intervene. A small additional penalty (``-1``, controlled
    by ``cfg.under_anticipate_pen``) is routed to the **delay channel
    only**, giving the delay head a clean local signal that says "you
    should have anticipated this and delayed".

    In auto_only mode (where ``agent_delay_ms`` is always 0 by
    construction), this penalty is suppressed by the env passing
    ``agent_delay_ms=0`` *and* ``env_cooling_ms=cooling_used`` — the
    caller controls whether to apply it via the explicit kwargs.
    """
    # Truncation: split across both channels so neither head escapes blame.
    if truncated:
        half = -float(cfg.truncate_pen) / 2.0
        return {"placement": half, "delay": half, "total": 2 * half}

    p = 0.0  # placement channel
    d = 0.0  # delay channel

    # Base reward -> placement (placement caused a task to be schedulable)
    p += float(cfg.base_step_reward)

    # Cooling-time penalty: split by who owned the cooling time.
    # env_cooling_ms blames placement (bad proc choice forced env to cool).
    # agent_delay_ms blames delay head (it chose to insert delay).
    # This replaces the previous tanh(overhead)-based formula which had
    # near-zero signal due to small per-task serialisation overhead.
    p -= cfg.cooling_time_w * float(env_cooling_ms)
    d -= cfg.cooling_time_w * float(agent_delay_ms)

    # Thermal walls -> placement (placement is what drives peak temp)
    if max_temp_during > cfg.T_pen:
        soft_pen = cfg.soft_wall_scale * ((max_temp_during - cfg.T_pen) / 5.0) ** 2
        p -= float(min(cfg.soft_wall_max, soft_pen))
    if max_temp_during > cfg.T_crit:
        p -= float(cfg.hard_wall_pen)

    # Cooling outcomes -> delay
    if cooling_used > 0:
        if would_violate_without_delay:
            d += float(cfg.cool_avoid_bonus)
        else:
            d -= float(cfg.cool_waste_pen)

    # Cool-node placement guidance -> placement
    if (final_temps is not None and len(final_temps) > 0
            and 0 <= target_node < len(final_temps)):
        mean_t = float(np.mean(final_temps))
        if final_temps[target_node] < mean_t - cfg.cool_node_margin:
            p += float(cfg.cool_node_bonus)

    # Under-anticipation: agent picked 0 delay but env had to cool -> delay
    if agent_delay_ms == 0.0 and env_cooling_ms > 0.0:
        d -= float(cfg.under_anticipate_pen)

    # Anticipation success: agent delayed AND env didn't need to step in
    # -> positive signal for delay channel.  This is the Stage-2-specific
    # reward that auto_only mode cannot earn (since agent_delay_ms is
    # forced to 0 there), giving Stage 2's hybrid mode a clean
    # differentiation from Stage 1.
    #
    # Conditions (v2.2):
    #   - agent inserted some delay (the action was non-trivial)
    #   - env subsequently did NOT need to insert auto-cool (anticipation
    #     was successful — the agent's delay was sufficient)
    #   - max_temp_during stayed below the soft wall (the trajectory
    #     itself was thermally safe — i.e. the delay wasn't merely
    #     redundant in a doomed trajectory)
    #   - would_violate_without_delay (NEW gate — see below)
    #
    # v2.2: Bonus is now gated on `would_violate_without_delay`.
    # Previously the bonus fired whenever the trajectory was thermally
    # safe, which made it trivially earnable in warm/hot regimes (where
    # peak_T saturates well below T_pen even without delay) and
    # incentivised the agent to fire delay for free. The gate aligns
    # the bonus shape with cool_avoid_bonus: both now require the env's
    # lookahead to confirm the delay was necessary.
    if (agent_delay_ms > 0.0
            and env_cooling_ms == 0.0
            and max_temp_during <= cfg.T_pen
            and would_violate_without_delay):
        d += float(cfg.anticipation_bonus)

    # Phase bonuses: split 50/50 (both heads contributed)
    if dag_done:
        p += float(cfg.dag_done_bonus) / 2.0
        d += float(cfg.dag_done_bonus) / 2.0
    if episode_done:
        p += float(cfg.episode_done_bonus) / 2.0
        d += float(cfg.episode_done_bonus) / 2.0

    return {"placement": float(p), "delay": float(d), "total": float(p + d)}


# =====================================================================
# Inline self-tests (run: python reward_shaping.py)
# =====================================================================
def _test_truncate_overrides_everything():
    """Truncation returns exactly -truncate_pen regardless of other inputs."""
    r = compute_reward(
        exec_time=1.0, base_workload=1.0, max_temp_during=70.0,
        final_temps=np.full(9, 30.0, dtype=np.float32), target_node=0,
        cooling_used=0.0, would_violate_without_delay=False,
        dag_done=True, episode_done=True, truncated=True,
    )
    assert r == -_DEFAULT_CFG.truncate_pen, f"truncate didn't override: r={r}"
    print(f"  ✅ truncate override: r={r}")


def _test_phase_bonuses():
    """Finishing a DAG and the episode adds 5 + 20 on top of base."""
    r = compute_reward(
        exec_time=10.0, base_workload=10.0, max_temp_during=70.0,
        final_temps=np.full(9, 70.0, dtype=np.float32), target_node=0,
        cooling_used=0.0, would_violate_without_delay=False,
        dag_done=True, episode_done=True, truncated=False,
    )
    expected = (
        _DEFAULT_CFG.base_step_reward
        + _DEFAULT_CFG.dag_done_bonus
        + _DEFAULT_CFG.episode_done_bonus
    )
    assert abs(r - expected) < 1e-6, f"phase bonus mismatch: {r} vs {expected}"
    print(f"  ✅ phase bonuses: r={r}")


def _test_active_cooling_signs():
    """Helpful cooling > wasteful cooling."""
    common = dict(
        exec_time=10.0, base_workload=10.0, max_temp_during=70.0,
        final_temps=np.full(9, 70.0, dtype=np.float32), target_node=0,
        dag_done=False, episode_done=False, truncated=False,
    )
    r_help  = compute_reward(cooling_used=5.0,
                             would_violate_without_delay=True,  **common)
    r_waste = compute_reward(cooling_used=5.0,
                             would_violate_without_delay=False, **common)
    assert r_help > r_waste, f"helpful cooling not rewarded: {r_help} <= {r_waste}"
    diff = (_DEFAULT_CFG.cool_avoid_bonus + _DEFAULT_CFG.cool_waste_pen)
    assert abs((r_help - r_waste) - diff) < 1e-6, \
        f"unexpected cooling delta: {r_help - r_waste} vs {diff}"
    print(f"  ✅ cooling signs: help={r_help:.2f}, waste={r_waste:.2f}")


def _test_thermal_walls_monotone():
    """Higher temp ⇒ stricter penalty, monotonically below the cap."""
    common = dict(
        exec_time=10.0, base_workload=10.0,
        final_temps=np.full(9, 60.0, dtype=np.float32), target_node=0,
        cooling_used=0.0, would_violate_without_delay=False,
        dag_done=False, episode_done=False, truncated=False,
    )
    rewards = [compute_reward(max_temp_during=t, **common)
               for t in [70.0, 78.0, 80.5, 82.0, 84.0, 86.0]]
    # Monotonically non-increasing
    for a, b in zip(rewards, rewards[1:]):
        assert a >= b - 1e-6, f"non-monotone thermal wall: {rewards}"
    # 86 °C must trigger BOTH soft cap and hard wall
    diff_84_86 = rewards[-2] - rewards[-1]
    assert diff_84_86 >= _DEFAULT_CFG.hard_wall_pen - 1.0, \
        f"hard wall didn't trigger as expected: 84->86 = {diff_84_86}"
    print(f"  ✅ thermal walls monotone:"
          f" 70°C={rewards[0]:.2f} → 86°C={rewards[-1]:.2f}")


def _test_reward_bounds():
    """99% of random rewards lie within an expected envelope."""
    cfg = _DEFAULT_CFG
    rng = np.random.default_rng(0)

    rewards = []
    for _ in range(2000):
        exec_time     = float(rng.uniform(1, 200))
        base_workload = float(rng.uniform(1, 50))
        max_temp      = float(rng.uniform(40, 90))
        final_temps   = rng.uniform(40, 85, size=9).astype(np.float32)
        tgt_node      = int(rng.integers(0, 9))
        cooling_used  = float(rng.uniform(0, 30))
        would_avoid   = bool(rng.random() > 0.5)
        dag_done      = bool(rng.random() > 0.9)
        ep_done       = dag_done and rng.random() > 0.9
        truncated     = bool(rng.random() > 0.97)

        rewards.append(compute_reward(
            exec_time=exec_time, base_workload=base_workload,
            max_temp_during=max_temp, final_temps=final_temps,
            target_node=tgt_node, cooling_used=cooling_used,
            would_violate_without_delay=would_avoid,
            dag_done=dag_done, episode_done=ep_done, truncated=truncated,
            cfg=cfg,
        ))

    rewards = np.array(rewards)
    in_band = ((rewards >= -50) & (rewards <= 30)).mean()
    assert in_band >= 0.99, f"reward out-of-band rate too high: {1 - in_band:.3f}"
    print(f"  ✅ bounded test: n={len(rewards)},"
          f" min={rewards.min():.2f}, max={rewards.max():.2f},"
          f" mean={rewards.mean():.2f}")


def _test_decompose_sums_to_total():
    """``decompose_reward['total']`` must equal ``compute_reward``."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        kwargs = dict(
            exec_time=float(rng.uniform(1, 200)),
            base_workload=float(rng.uniform(1, 50)),
            max_temp_during=float(rng.uniform(40, 90)),
            final_temps=rng.uniform(40, 85, size=9).astype(np.float32),
            target_node=int(rng.integers(0, 9)),
            cooling_used=float(rng.uniform(0, 30)),
            would_violate_without_delay=bool(rng.random() > 0.5),
            dag_done=bool(rng.random() > 0.9),
            episode_done=False,
            truncated=bool(rng.random() > 0.95),
        )
        r = compute_reward(**kwargs)
        d = decompose_reward(**kwargs)
        assert abs(d["total"] - r) < 1e-5, \
            f"decompose mismatch: {d['total']} vs {r}\n{d}"
    print("  ✅ decompose sums to compute_reward total (200 random samples)")


def _test_channels_consistent_with_total():
    """compute_reward_channels['total'] should equal compute_reward when
    agent_delay_ms == cooling_used (i.e. no env contribution -> no
    under_anticipate penalty), within +/- 0 (exact).

    Note: as of v2.1, the channels function adds an anticipation_bonus
    when (agent_delay_ms > 0 AND env_cooling_ms == 0 AND max_T <= T_pen).
    The scalar compute_reward doesn't have this term (it operates without
    knowing how cooling was split between agent and env), so for
    randomly-sampled inputs that satisfy the bonus condition, we expect
    ch['total'] = r + anticipation_bonus.  We test both branches.

    v2.2 (HK-1.5.8 R-C): anticipation_bonus is now additionally gated
    on `would_violate_without_delay == True`.  The expected_delta calc
    below was updated to reflect this.
    """
    rng = np.random.default_rng(2)
    matches      = 0
    bonus_cases  = 0
    for _ in range(200):
        cooling = float(rng.uniform(0, 20))
        max_T   = float(rng.uniform(40, 84))
        kwargs_main = dict(
            exec_time=float(rng.uniform(1, 200)),
            base_workload=float(rng.uniform(1, 50)),
            max_temp_during=max_T,
            final_temps=rng.uniform(40, 80, size=9).astype(np.float32),
            target_node=int(rng.integers(0, 9)),
            cooling_used=cooling,
            would_violate_without_delay=bool(rng.random() > 0.5),
            dag_done=bool(rng.random() > 0.9),
            episode_done=False,
            truncated=False,  # truncate splits differently; tested below
        )
        r = compute_reward(**kwargs_main)
        # Setting agent_delay_ms = cooling_used, env_cooling_ms = 0
        # suppresses under_anticipate_pen.  If max_T <= T_pen AND
        # would_violate_without_delay (v2.2 R-C gate), also triggers
        # anticipation_bonus.
        ch = compute_reward_channels(
            agent_delay_ms=cooling,
            env_cooling_ms=0.0,
            **kwargs_main,
        )
        # Expected delta: anticipation_bonus iff ALL conditions met.
        # R-C (v2.2): bonus also requires would_violate_without_delay=True.
        expects_bonus = (
            cooling > 0.0
            and max_T <= _DEFAULT_CFG.T_pen
            and kwargs_main.get('would_violate_without_delay', False)
        )
        expected_delta = _DEFAULT_CFG.anticipation_bonus if expects_bonus else 0.0
        if expects_bonus:
            bonus_cases += 1
        assert abs(ch["total"] - r - expected_delta) < 1e-5, \
            f"channels mismatch: {ch} vs r={r} (expected delta={expected_delta})\n"\
            f"kwargs={kwargs_main}"
        matches += 1
    print(f"  ✅ channels['total'] == compute_reward + anticipation_bonus "
          f"({matches} samples, {bonus_cases} earned bonus)")


def _test_channels_under_anticipate_routes_to_delay():
    """When agent=0 but env cooled, the routing should reflect both:
    - delay channel: -under_anticipate_pen (penalised for not pre-empting)
    - placement channel: -cooling_time_w * env_cooling_ms (paid the time cost)

    When agent=5 (no env), delay gets the cooling-time penalty AND
    +anticipation_bonus.

    Both routings work as designed if the channel deltas reflect the
    cause-and-effect attribution.
    """
    common = dict(
        exec_time=20.0, base_workload=10.0, max_temp_during=70.0,
        final_temps=np.full(9, 60.0, dtype=np.float32), target_node=0,
        cooling_used=5.0, would_violate_without_delay=False,
        dag_done=False, episode_done=False, truncated=False,
    )
    # Case A: agent_delay=5, env_cool=0 (good — agent anticipated)
    case_a = compute_reward_channels(
        agent_delay_ms=5.0, env_cooling_ms=0.0, **common,
    )
    # Case B: agent_delay=0, env_cool=5 (bad — env had to step in)
    case_b = compute_reward_channels(
        agent_delay_ms=0.0, env_cooling_ms=5.0, **common,
    )
    # Placement should be more punished in B than A (env_cool > 0 in B)
    assert case_a["placement"] > case_b["placement"], \
        f"Case A placement should beat B: {case_a} vs {case_b}"
    # Delay should be more rewarded in A than B (agent gets bonus in A,
    # under-anticipate penalty in B)
    assert case_a["delay"] > case_b["delay"], \
        f"Case A delay should beat B: {case_a} vs {case_b}"
    # Total should be higher in A (anticipation worked)
    assert case_a["total"] > case_b["total"]
    print(f"  ✅ channel routing: anticipated (case A): "
          f"p={case_a['placement']:.2f}, d={case_a['delay']:.2f}; "
          f"reactive (case B): p={case_b['placement']:.2f}, "
          f"d={case_b['delay']:.2f}")


def _test_channels_truncate_splits():
    """On truncate, both channels get -truncate_pen/2."""
    common = dict(
        exec_time=10.0, base_workload=10.0, max_temp_during=86.0,
        final_temps=np.full(9, 80.0, dtype=np.float32), target_node=0,
        cooling_used=0.0,
        agent_delay_ms=0.0, env_cooling_ms=0.0,
        would_violate_without_delay=False,
        dag_done=False, episode_done=False, truncated=True,
    )
    ch = compute_reward_channels(**common)
    half = -_DEFAULT_CFG.truncate_pen / 2.0
    assert abs(ch["placement"] - half) < 1e-6
    assert abs(ch["delay"] - half) < 1e-6
    assert abs(ch["total"] - 2 * half) < 1e-6
    print(f"  ✅ truncate splits 50/50: placement={ch['placement']}, delay={ch['delay']}")


def _test_smoke_typical_episode():
    """A realistic 'happy-path' step: low temp, no cooling -> just base reward."""
    r = compute_reward(
        exec_time=12.0, base_workload=10.0, max_temp_during=72.0,
        final_temps=np.array([70., 65., 64., 63., 62., 61., 61., 60., 60.],
                             dtype=np.float32),
        target_node=0,
        cooling_used=0.0, would_violate_without_delay=False,
        dag_done=False, episode_done=False, truncated=False,
    )
    # base(1) - cooling_time_w * cooling_used(0) = 1.0
    # When no cooling is used, agent gets the +1 base reward.
    # Small cooling deductions kick in only when cooling > 0.
    assert abs(r - 1.0) < 0.1, f"happy-path reward off: {r}"
    print(f"  ✅ smoke (typical step, no cooling): r={r:.3f}")


def _run_all_tests():
    print("Running reward_shaping.py self-tests...\n")
    _test_truncate_overrides_everything()
    _test_phase_bonuses()
    _test_active_cooling_signs()
    _test_thermal_walls_monotone()
    _test_smoke_typical_episode()
    _test_reward_bounds()
    _test_decompose_sums_to_total()
    _test_channels_consistent_with_total()
    _test_channels_under_anticipate_routes_to_delay()
    _test_channels_truncate_splits()
    print("\nAll tests passed ✓")


if __name__ == "__main__":
    _run_all_tests()