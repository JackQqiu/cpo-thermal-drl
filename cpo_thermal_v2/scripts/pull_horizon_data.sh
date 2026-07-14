#!/usr/bin/env bash
# =====================================================================
# pull_horizon_data.sh — extract final horizon-scan numbers needed
#                        for Section 5.5 "Robustness to Episode Horizon"
#
# Usage:
#   cd <REPO_ROOT>
#   bash cpo_thermal_v2/scripts/pull_horizon_data.sh \
#       2>&1 | tee horizon_data_$(date +%Y%m%d_%H%M%S).log
# =====================================================================

set -e

python3 << 'PYEOF'
import pandas as pd
import os

# ------------------------------------------------------------------
# Section 1: Unsafe-episode counts per (scheduler x horizon x setting)
# ------------------------------------------------------------------
# Each cell is "n_unsafe / 200" where n_unsafe = (violations_total > 0).sum()

print("=" * 78)
print("HORIZON SCAN — UNSAFE-EPISODE COUNTS (out of 200 per cell)")
print("=" * 78)

schedulers = ['HEFT', 'ThermalHEFT', 'RoundRobin', 'Decima',
              'Ours-auto_only', 'Ours-hybrid', 'Ours-agent_only']
horizons   = [20, 50, 100, 200]
settings   = ['easy', 'warm', 'hot', 'extreme']

for setting in settings:
    print(f"\n### {setting} setting ###")
    print(f"{'scheduler':<22} " + " ".join(f"H={h:<5}" for h in horizons))
    print("-" * 70)
    for sched in schedulers:
        row = [f"{sched:<22}"]
        for h in horizons:
            f = f'eval_results/horizon_scan_v2/{setting}/dags{h}/episodes.csv'
            if not os.path.isfile(f):
                row.append(f"{'MISSING':>7}")
                continue
            df = pd.read_csv(f)
            sub = df[df.scheduler == sched]
            if len(sub) == 0:
                row.append(f"{'?':>7}")
                continue
            n_unsafe = int((sub.violations_total > 0).sum())
            row.append(f"{n_unsafe:>3}/{len(sub):<3}")
        print(" ".join(row))

# ------------------------------------------------------------------
# Section 2: Same data as RATE (% unsafe) for figure plotting
# ------------------------------------------------------------------
print()
print("=" * 78)
print("HORIZON SCAN — UNSAFE-EPISODE RATE (%)")
print("=" * 78)

for setting in settings:
    print(f"\n### {setting} setting ###")
    print(f"{'scheduler':<22} " + " ".join(f"H={h:<6}" for h in horizons))
    print("-" * 70)
    for sched in schedulers:
        row = [f"{sched:<22}"]
        for h in horizons:
            f = f'eval_results/horizon_scan_v2/{setting}/dags{h}/episodes.csv'
            if not os.path.isfile(f):
                row.append(f"{'MISS':>7}")
                continue
            df = pd.read_csv(f)
            sub = df[df.scheduler == sched]
            if len(sub) == 0:
                row.append(f"{'?':>7}")
                continue
            n_unsafe = int((sub.violations_total > 0).sum())
            rate = 100.0 * n_unsafe / len(sub)
            row.append(f"{rate:>6.2f}%")
        print(" ".join(row))

# ------------------------------------------------------------------
# Section 3: DAG completion at horizon (key for thermally-aware vs blind)
# ------------------------------------------------------------------
print()
print("=" * 78)
print("HORIZON SCAN — DAG COMPLETION RATE (mean across episodes)")
print("=" * 78)

for setting in settings:
    print(f"\n### {setting} setting ###")
    print(f"{'scheduler':<22} " + " ".join(f"H={h:<7}" for h in horizons))
    print("-" * 75)
    for sched in schedulers:
        row = [f"{sched:<22}"]
        for h in horizons:
            f = f'eval_results/horizon_scan_v2/{setting}/dags{h}/episodes.csv'
            if not os.path.isfile(f):
                row.append(f"{'MISS':>7}")
                continue
            df = pd.read_csv(f)
            sub = df[df.scheduler == sched]
            if len(sub) == 0:
                row.append(f"{'?':>7}")
                continue
            comp = sub['dag_completion_rate'].mean()
            row.append(f"{comp:>7.4f}")
        print(" ".join(row))

# ------------------------------------------------------------------
# Section 4: Agent delay accumulation (Hybrid only) for narrative
# ------------------------------------------------------------------
print()
print("=" * 78)
print("HORIZON SCAN — AGENT DELAY ACCUMULATION (mean ms, Hybrid only)")
print("=" * 78)

for setting in settings:
    print(f"\n### {setting} setting ###")
    print(f"{'scheduler':<22} " + " ".join(f"H={h:<10}" for h in horizons))
    print("-" * 80)
    for sched in ['Ours-hybrid']:
        row = [f"{sched:<22}"]
        for h in horizons:
            f = f'eval_results/horizon_scan_v2/{setting}/dags{h}/episodes.csv'
            if not os.path.isfile(f):
                row.append(f"{'MISS':>10}")
                continue
            df = pd.read_csv(f)
            sub = df[df.scheduler == sched]
            if len(sub) == 0:
                row.append(f"{'?':>10}")
                continue
            d = sub['agent_delay_total_ms'].mean()
            row.append(f"{d:>10.1f}")
        print(" ".join(row))

# ------------------------------------------------------------------
# Section 5: Mean makespan (for trade-off narrative — does makespan
#            scale linearly with H for Hybrid? answers "is delay
#            overhead horizon-stable?")
# ------------------------------------------------------------------
print()
print("=" * 78)
print("HORIZON SCAN — MEAN MAKESPAN (ms)")
print("=" * 78)
print("(Watch for: does Hybrid makespan stay ~5316 + linear-in-H delay,")
print("or does it grow super-linearly under thermal stress?)")

for setting in settings:
    print(f"\n### {setting} setting ###")
    print(f"{'scheduler':<22} " + " ".join(f"H={h:<8}" for h in horizons))
    print("-" * 75)
    for sched in ['HEFT', 'Decima', 'Ours-auto_only', 'Ours-hybrid']:
        row = [f"{sched:<22}"]
        for h in horizons:
            f = f'eval_results/horizon_scan_v2/{setting}/dags{h}/episodes.csv'
            if not os.path.isfile(f):
                row.append(f"{'MISS':>8}")
                continue
            df = pd.read_csv(f)
            sub = df[df.scheduler == sched]
            if len(sub) == 0:
                row.append(f"{'?':>8}")
                continue
            m = sub['total_makespan_ms'].mean()
            row.append(f"{m:>8.0f}")
        print(" ".join(row))

print()
print("=" * 78)
print("END — all numbers come straight from eval_results/horizon_scan_v2/*.csv.")
print("=" * 78)
PYEOF
