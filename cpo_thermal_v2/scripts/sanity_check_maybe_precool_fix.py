"""
sanity_check_maybe_precool_fix.py
=================================

Verify that the _maybe_precool fix changes env behavior as expected.

Runs HEFT (auto_only mode) for 50 episodes in extreme regime and
reports summary statistics.  Compare against the pre-fix HEFT numbers
that you already have in your unit-test logs:

  Before fix (auto_only HEFT, extreme):
    cooling_total mean ≈ 0 ms       ← env auto-cool gated out
    peak_T mean        ≈ 91.5 °C    ← T_crit breached
    truncate_rate      ≈ 96%
    dag_completion     ≈ 0.237

  After fix (expected):
    cooling_total mean significantly > 0   ← env auto-cool now firing
    peak_T mean        well below 85       ← violations actually prevented
    truncate_rate      substantially lower
    dag_completion     substantially higher

The script prints a verdict at the end. Run BEFORE retraining; if the
verdict says the fix is not working, do not retrain.

Usage:
    PYTHONPATH=. python cpo_thermal_v2/scripts/sanity_check_maybe_precool_fix.py

Optional flags:
    --num-episodes 50       (default: 50)
    --setting extreme       (default: extreme; also: easy/warm/hot)
    --num-nodes 17
    --seed-base 10000

Why HEFT specifically:
    HEFT in auto_only mode never emits a delay action of its own
    (auto_only doesn't expose the delay head). So all the cooling that
    happens in the HEFT auto_only run comes from env's auto-cool. This
    isolates the env behavior cleanly: cooling_total = env_cooling.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=50)
    ap.add_argument("--num-nodes",    type=int, default=17)
    ap.add_argument("--setting", type=str, default="extreme",
                    choices=["easy", "warm", "hot", "extreme"])
    ap.add_argument("--seed-base", type=int, default=10000)
    args = ap.parse_args()

    settings = {
        "easy":    (30.0, 45.0),
        "warm":    (40.0, 55.0),
        "hot":     (50.0, 65.0),
        "extreme": (60.0, 75.0),
    }
    T_lo, T_hi = settings[args.setting]

    # Reference numbers from pre-fix HEFT (auto_only, extreme, 50 eps).
    # Used in the "comparison" table below so the user sees the
    # before/after diff at a glance.
    PREFIX_REF = {
        "extreme": dict(cool=0.0,   peak=91.5, trunc=0.96, comp=0.237),
        "hot":     dict(cool=0.0,   peak=90.9, trunc=0.92, comp=0.300),
        "warm":    dict(cool=None,  peak=None, trunc=None, comp=None),
        "easy":    dict(cool=None,  peak=None, trunc=None, comp=None),
    }

    from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
    from cpo_thermal_v2.baselines import HEFTScheduler
    from cpo_thermal_v2.evaluation.metrics import (
        EpisodeRecorder, run_episode,
    )

    print(f"\n{'='*72}")
    print(f"Sanity check: _maybe_precool fix verification")
    print(f"{'='*72}")
    print(f"Setting:        {args.setting}  T_0 in [{T_lo}, {T_hi}] °C")
    print(f"Num nodes:      {args.num_nodes}")
    print(f"Num episodes:   {args.num_episodes}")
    print(f"Action mode:    auto_only  (HEFT — agent emits no delay)")
    print(f"Truncate mode:  hard")
    print()

    env = CPOThermalDAGEnvV2(
        num_nodes=args.num_nodes,
        action_mode="auto_only",
        truncate_mode="hard",
        initial_temp_range=(T_lo, T_hi),
        dags_per_episode=20,
    )
    sched = HEFTScheduler(num_nodes=args.num_nodes, action_mode="auto_only")

    eps = []
    t0 = time.time()
    for ep in range(args.num_episodes):
        rec = EpisodeRecorder("HEFT", args.num_nodes, "auto_only", ep)
        ep_rec = run_episode(
            env=env, scheduler=sched, recorder=rec,
            seed=args.seed_base + ep,
        )
        eps.append(ep_rec)
    elapsed = time.time() - t0

    # Aggregate
    cool   = np.array([e.cooling_total_ms      for e in eps])
    peakT  = np.array([e.peak_temp_episode     for e in eps])
    viol   = np.array([e.violations_total      for e in eps])
    comp   = np.array([e.dag_completion_rate   for e in eps])
    trunc  = np.array([1.0 if e.truncated else 0.0 for e in eps])
    ms     = np.array([e.total_makespan_ms     for e in eps])

    print(f"Results ({args.num_episodes} eps, {elapsed:.1f}s, "
          f"{args.num_episodes/max(elapsed, 1e-3):.1f} ep/s):")
    print(f"  cooling_total_ms (= env auto-cool, since HEFT auto_only):")
    print(f"    mean={cool.mean():8.2f}  median={np.median(cool):.2f}  "
          f"max={cool.max():.0f}")
    print(f"    P(cooling>0)   = {(cool > 0).mean():.0%}  "
          f"({(cool > 0).sum()}/{len(cool)} episodes)")
    print(f"  peak_temp_C:")
    print(f"    mean={peakT.mean():.2f}   max={peakT.max():.2f}")
    print(f"  violations:    mean={viol.mean():.2f}   "
          f">0={(viol>0).sum()}/{args.num_episodes}")
    print(f"  truncate_rate: {trunc.mean():.0%}")
    print(f"  dag_completion: mean={comp.mean():.3f}")
    print(f"  makespan_ms:    mean={ms.mean():.1f}")

    # Comparison table
    ref = PREFIX_REF.get(args.setting, {})
    if ref.get("cool") is not None:
        print()
        print(f"Comparison to your pre-fix HEFT numbers ({args.setting}):")
        print(f"  metric              |   pre-fix  |  post-fix  | direction")
        print(f"  --------------------|------------|------------|----------")
        print(f"  cooling_total mean  | {ref['cool']:>7.1f} ms | "
              f"{cool.mean():>7.2f} ms | should ↑↑")
        print(f"  peak_T mean         | {ref['peak']:>6.2f} °C  | "
              f"{peakT.mean():>6.2f} °C  | should ↓↓")
        print(f"  truncate_rate       | {ref['trunc']*100:>7.0f}%   | "
              f"{trunc.mean()*100:>7.0f}%   | should ↓↓")
        print(f"  dag_completion mean | {ref['comp']:>7.3f}    | "
              f"{comp.mean():>7.3f}    | should ↑↑")

    # Verdict
    print()
    print(f"{'='*72}")
    print(f"Verdict")
    print(f"{'='*72}")

    cool_ok  = cool.mean() > 20.0
    peak_ok  = peakT.mean() < 86.0 or peakT.mean() < (
        ref.get("peak", 100.0) - 5.0 if ref.get("peak") else False)
    trunc_ok = trunc.mean() < 0.5

    if cool_ok and peak_ok and trunc_ok:
        print(f"  ✓ Fix is WORKING.")
        print(f"    env auto-cool now actively prevents violations:")
        print(f"      • cooling_total per episode rose from 0 to "
              f"{cool.mean():.0f} ms (mean)")
        print(f"      • peak_T dropped to {peakT.mean():.1f} °C")
        print(f"      • truncate_rate dropped to {trunc.mean():.0%}")
        print(f"  → SAFE TO PROCEED with retraining.")
    elif cool.mean() < 5.0:
        print(f"  ✗ env auto-cool is still ~0 — fix not applied or not working.")
        print(f"    cooling_total mean = {cool.mean():.2f} ms (was ~0 pre-fix).")
        print(f"  → Check that the new _maybe_precool body was actually saved")
        print(f"    to cpo_thermal_v2/envs/cpo_thermal_env.py and that no")
        print(f"    __pycache__/ has a stale version.")
        print(f"  → DO NOT proceed with retraining until this verdict passes.")
    else:
        print(f"  ⚠ Partial improvement, but not the full fix.")
        print(f"    cooling = {cool.mean():.1f} ms (expected >> 20)")
        print(f"    peak_T  = {peakT.mean():.1f} °C (expected < 86)")
        print(f"    trunc   = {trunc.mean():.0%} (expected < 50%)")
        print(f"  → Inspect the patched function vs the spec in")
        print(f"    maybe_precool_fix.md before retraining.")


if __name__ == "__main__":
    main()
