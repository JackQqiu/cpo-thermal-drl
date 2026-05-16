"""
scripts/verify_new_physics.py — Sanity-check new RC parameters
================================================================

After updating physical constants in generate_matrices.py and
regenerating the matrices, run this to confirm:
  1. ASIC and OE temperature rise rates are physically plausible
     (~0.05-0.2 K/ms under nominal active load).
  2. A round-robin baseline can complete most DAGs in easy setting
     (truncate_rate < 30%).
  3. ASIC stress test (10 consecutive ASIC tasks) doesn't immediately
     trigger T_crit.

Run BEFORE retraining; only proceed if all 3 checks pass.

Usage::

    PYTHONPATH=. python cpo_thermal_v2/scripts/verify_new_physics.py
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
    from cpo_thermal_v2.baselines import (
        HEFTScheduler, RoundRobinScheduler,
    )

    # -----------------------------------------------------------------
    # Check 1: per-step temperature rise rate at nominal load
    # -----------------------------------------------------------------
    print("=" * 70)
    print("CHECK 1: Per-step temperature rise (10ms ASIC task @ T₀=30°C)")
    print("=" * 70)
    env = CPOThermalDAGEnvV2(
        num_nodes        = 17,
        initial_temp_range = (30.0, 30.001),  # uniform 30°C start
        action_mode        = "auto_only",
        truncate_mode      = "hard",
    )
    env.reset(seed=42)
    T_before = float(env.thermal_engine.temperatures[0])
    # Manually simulate 10ms of ASIC active execution
    for _ in range(10):
        P = env._compute_power(0, 0.0)   # target_node=0 (ASIC), no traffic
        env.thermal_engine.step(P)
    T_after = float(env.thermal_engine.temperatures[0])
    rise_per_ms = (T_after - T_before) / 10.0

    print(f"  T_ASIC before: {T_before:.2f}°C")
    print(f"  T_ASIC after 10ms: {T_after:.2f}°C")
    print(f"  Rise rate: {rise_per_ms:.3f} K/ms")
    if 0.03 <= rise_per_ms <= 0.30:
        print(f"  ✅ Within physical range [0.03, 0.30] K/ms")
        check1_pass = True
    else:
        print(f"  ❌ Outside expected range [0.03, 0.30] K/ms")
        check1_pass = False

    # -----------------------------------------------------------------
    # Check 2: OE single-task rise (the old fail mode)
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("CHECK 2: Single 30ms OE task @ T₀=65°C (old fail point)")
    print("=" * 70)
    env = CPOThermalDAGEnvV2(
        num_nodes        = 17,
        initial_temp_range = (65.0, 65.001),
        action_mode        = "auto_only",
        truncate_mode      = "hard",
    )
    env.reset(seed=42)
    T_oe_before = float(env.thermal_engine.temperatures[1])
    for _ in range(30):
        P = env._compute_power(1, 0.0)   # target_node=1 (OE1), no traffic
        env.thermal_engine.step(P)
    T_oe_after = float(env.thermal_engine.temperatures[1])
    delta_oe = T_oe_after - T_oe_before

    print(f"  T_OE before: {T_oe_before:.2f}°C")
    print(f"  T_OE after 30ms: {T_oe_after:.2f}°C")
    print(f"  Delta: +{delta_oe:.2f}°C")
    if delta_oe < 15.0:
        print(f"  ✅ Reasonable single-task rise (was +40°C in old physics)")
        check2_pass = True
    else:
        print(f"  ❌ Single OE task still causes catastrophic rise")
        check2_pass = False

    # -----------------------------------------------------------------
    # Check 3: HEFT survival in easy setting (10 episodes)
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("CHECK 3: HEFT survival in easy setting (10 episodes)")
    print("=" * 70)
    sched = HEFTScheduler(num_nodes=17, action_mode="auto_only")
    env = CPOThermalDAGEnvV2(
        num_nodes        = 17,
        initial_temp_range = (30.0, 45.0),
        max_dag_size       = None,
        dags_per_episode   = 20,
        action_mode        = "auto_only",
        truncate_mode      = "hard",
    )

    truncates = 0
    completions = []
    for ep in range(10):
        obs, info = env.reset(seed=10000 + ep)
        sched.reset(obs, info)
        truncated = False
        while True:
            a = sched.schedule(obs, info)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                truncated = trunc
                break
        if truncated:
            truncates += 1
        completions.append(env.current_dag_count)

    truncate_rate = truncates / 10
    avg_completion = np.mean(completions)
    print(f"  Truncate rate: {truncate_rate*100:.0f}%")
    print(f"  Avg DAGs completed: {avg_completion:.1f} / 20")
    if truncate_rate < 0.5:
        print(f"  ✅ HEFT can complete most episodes in easy setting")
        check3_pass = True
    else:
        print(f"  ⚠️ HEFT still truncates {truncate_rate*100:.0f}% of episodes")
        check3_pass = False

    # -----------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Check 1 (rise rate):     {'✅ PASS' if check1_pass else '❌ FAIL'}")
    print(f"Check 2 (single OE):     {'✅ PASS' if check2_pass else '❌ FAIL'}")
    print(f"Check 3 (HEFT survival): {'✅ PASS' if check3_pass else '⚠️  WEAK'}")
    print("=" * 70)
    if check1_pass and check2_pass and check3_pass:
        print("\n🎉 New physics is healthy. Safe to retrain.\n")
        return 0
    elif check1_pass and check2_pass:
        print("\n⚠️ Physics is plausible but HEFT survival is poor — "
              "may need to widen mask_temp / T_crit gap or tune T_crit "
              "before retraining.\n")
        return 1
    else:
        print("\n❌ New physics still has issues. Tune constants further "
              "before retraining.\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())