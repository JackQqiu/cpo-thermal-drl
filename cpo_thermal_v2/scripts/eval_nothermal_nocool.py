"""
Quick standalone: Ours-NoThermal eval at hot N=17, cool=0, 200 ep.

Provides the missing "before" side of Component 5.5 (Ours-NoThermal →
Ours-NoRCEdge) for the auto-cool ablation table.

Output: eval_results/nothermal_nocool/paired.csv (Ours-NoThermal only)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
from cpo_thermal_v2.envs.reward_shaping import RewardConfig
from cpo_thermal_v2.baselines.decima_fair import DecimaFairScheduler

# Reuse the single-episode harness
from cpo_thermal_v2.scripts.eval_ours_no_rc_edge_paired import run_one_episode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--seed_base", type=int, default=100000)
    p.add_argument("--num_nodes", type=int, default=17)
    p.add_argument("--max_cooling_steps", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                   default="eval_results/nothermal_nocool")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    env = CPOThermalDAGEnvV2(
        num_nodes          = args.num_nodes,
        dt                 = 1e-3,
        thermal_target     = 75.0, thermal_guardband = 80.0,
        thermal_critical   = 85.0, mask_temp = 82.0,
        action_mode        = "auto_only",
        initial_temp_range = (60.0, 75.0),
        max_dag_size       = None,
        dags_per_episode   = 20,
        temp_rise_per_ms_asic = 0.08,
        temp_rise_per_ms_oe   = 0.18,
        oe_active_power    = 40.0,
        max_cooling_steps  = args.max_cooling_steps,
        reward_config      = RewardConfig(),
        dataset_path       = "./data_pipeline/process/alibaba_dags_v2.json",
    )

    sched = DecimaFairScheduler(
        ckpt_path = "checkpoints_v1/decima_fair_N17/best.pt",
        num_nodes = args.num_nodes,
        deterministic = True, device = "cpu",
    )
    print(f"[eval] Ours-NoThermal cool={args.max_cooling_steps}, n={args.n_episodes}, "
          f"seed_base={args.seed_base}, hot N={args.num_nodes}")

    rows = []
    for i in range(args.n_episodes):
        m = run_one_episode(env, sched, args.seed_base + i)
        m.update(episode_id=i, scheduler="Ours-NoThermal")
        rows.append(m)
        if (i + 1) % 25 == 0:
            v = sum(r["is_unsafe"] for r in rows) / len(rows)
            print(f"  [{i+1:3d}/{args.n_episodes}] viol={v:.4f}")

    fields = ["episode_id", "scheduler", "total_makespan_ms",
              "peak_temp_episode", "violations_total", "dag_completion_rate",
              "truncated", "episode_return", "is_unsafe"]
    csv_path = out / "paired.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in fields})

    viol = sum(r["is_unsafe"] for r in rows) / len(rows)
    mp   = np.mean([r["peak_temp_episode"] for r in rows])
    mk   = np.mean([r["total_makespan_ms"] for r in rows])
    print(f"\n Ours-NoThermal | cool={args.max_cooling_steps} | n={args.n_episodes}")
    print(f"   viol_rate = {viol:.4f} ({sum(r['is_unsafe'] for r in rows)}/{len(rows)})")
    print(f"   mean peak T = {mp:.2f}°C   mean makespan = {mk:.1f} ms")

    # Paired McNemar against auto_only's first 200 eps from existing eval
    import pandas as pd
    try:
        existing = pd.read_csv("eval_results/autocool_ablation_hot_n17/paired.csv")
        ao = existing[(existing.scheduler == "Ours-auto_only") &
                       (existing.episode_id < args.n_episodes)]
        ao_unsafe = {int(r.episode_id): int(r.is_unsafe) for _, r in ao.iterrows()}
        nt_unsafe = {r["episode_id"]: r["is_unsafe"] for r in rows}
        n01 = sum(1 for i in nt_unsafe if ao_unsafe.get(i, 0) == 0 and nt_unsafe[i] == 1)
        n10 = sum(1 for i in nt_unsafe if ao_unsafe.get(i, 0) == 1 and nt_unsafe[i] == 0)
        n11 = sum(1 for i in nt_unsafe if ao_unsafe.get(i, 0) == 1 and nt_unsafe[i] == 1)
        n00 = sum(1 for i in nt_unsafe if ao_unsafe.get(i, 0) == 0 and nt_unsafe[i] == 0)
        if n01 + n10 == 0:
            pv = 1.0
        else:
            pv = binomtest(min(n01, n10), n=n01 + n10, p=0.5,
                           alternative="two-sided").pvalue
        print(f"\n Paired McNemar (Ours-NoThermal vs Ours-auto_only first {args.n_episodes} eps, cool=0):")
        print(f"   n00={n00} n01={n01} (auto safe, NoTherm unsafe) "
              f"n10={n10} (auto unsafe, NoTherm safe) n11={n11}")
        print(f"   p = {pv:.6e}")
    except Exception as e:
        print(f"  paired McNemar skipped: {e}")


if __name__ == "__main__":
    main()
