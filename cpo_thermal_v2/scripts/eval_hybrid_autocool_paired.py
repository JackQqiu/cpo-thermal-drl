"""
eval_hybrid_autocool_paired.py — Tier-2 audit fix.

Run Ours-hybrid (Stage-2 checkpoint, action_mode="hybrid") at
max_cooling_steps=0 on the same paired seed pool as the existing
auto-cool ablation (seed_base=100000, hot ambient N=17, n=500).

This closes the audit blocker that "policy-intrinsic safety" was
claimed in §1/abstract but never tested on Ours-hybrid — the
deployed headline model.

Output:
  eval_results/hybrid_autocool_nocool/paired.csv  (1 scheduler, n=500)
  stdout summary + paired McNemar vs Ours-auto_only anchor
  (anchor reused from eval_results/autocool_ablation_hot_n17_nocool/paired.csv
   if present, otherwise run anchor inline)

Usage:
  python -m cpo_thermal_v2.scripts.eval_hybrid_autocool_paired \\
      --n_episodes 500 --seed_base 100000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import binomtest

from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
from cpo_thermal_v2.envs.reward_shaping import RewardConfig
from cpo_thermal_v2.baselines import TrainedPPOScheduler

from cpo_thermal_v2.scripts.eval_ours_no_rc_edge_paired import run_one_episode


T_PEN = 80.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--seed_base", type=int, default=100000)
    p.add_argument("--num_nodes", type=int, default=17)
    p.add_argument("--max_cooling_steps", type=int, default=0)
    p.add_argument("--output_dir", type=str,
                   default="eval_results/hybrid_autocool_nocool")
    p.add_argument("--anchor_csv", type=str,
                   default="eval_results/autocool_ablation_hot_n17_nocool/paired.csv",
                   help="paired.csv from the existing autocool ablation (for "
                        "Ours-auto_only anchor on the same seed pool)")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    env_hybrid = CPOThermalDAGEnvV2(
        num_nodes          = args.num_nodes,
        dt                 = 1e-3,
        thermal_target     = 75.0, thermal_guardband = 80.0,
        thermal_critical   = 85.0, mask_temp = 82.0,
        action_mode        = "hybrid",
        initial_temp_range = (60.0, 75.0),
        max_dag_size       = None,
        dags_per_episode   = 20,
        temp_rise_per_ms_asic = 0.08,
        temp_rise_per_ms_oe   = 0.18,
        oe_active_power    = 40.0,
        max_cooling_steps  = args.max_cooling_steps,
        reward_config      = RewardConfig(),
        dataset_path       = "./cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json",
        disable_rc_edge    = False,
    )

    sched = TrainedPPOScheduler(
        ckpt_path       = "checkpoints/stage2_hybrid_N17/best.pt",
        action_mode     = "hybrid",
        deterministic   = True,
        device          = "cpu",
        scheduler_label = "Ours-hybrid",
    )

    print(f"[eval] Ours-hybrid cool={args.max_cooling_steps}, "
          f"n={args.n_episodes}, seed_base={args.seed_base}, hot N={args.num_nodes}")

    rows: List[Dict] = []
    for i in range(args.n_episodes):
        m = run_one_episode(env_hybrid, sched, args.seed_base + i)
        m.update(episode_id=i, scheduler="Ours-hybrid")
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
    mp = np.mean([r["peak_temp_episode"] for r in rows])
    mk = np.mean([r["total_makespan_ms"] for r in rows])
    print(f"\n Ours-hybrid cool={args.max_cooling_steps} | n={args.n_episodes}")
    print(f"   viol_rate={viol:.4f} ({sum(r['is_unsafe'] for r in rows)}/{len(rows)})")
    print(f"   mean peak T={mp:.2f}°C   mean makespan={mk:.1f} ms")

    # Paired McNemar against Ours-auto_only anchor from the existing
    # autocool ablation CSV
    import pandas as pd
    try:
        anchor_df = pd.read_csv(args.anchor_csv)
        ao = anchor_df[anchor_df.scheduler == "Ours-auto_only"]
        ao_unsafe = {int(r.episode_id): int(r.is_unsafe) for _, r in ao.iterrows()}
        h_unsafe = {r["episode_id"]: r["is_unsafe"] for r in rows}
        common = set(ao_unsafe) & set(h_unsafe)
        n01 = sum(1 for i in common if ao_unsafe[i] == 0 and h_unsafe[i] == 1)
        n10 = sum(1 for i in common if ao_unsafe[i] == 1 and h_unsafe[i] == 0)
        n11 = sum(1 for i in common if ao_unsafe[i] == 1 and h_unsafe[i] == 1)
        n00 = sum(1 for i in common if ao_unsafe[i] == 0 and h_unsafe[i] == 0)
        if n01 + n10 == 0:
            pv = 1.0
        else:
            pv = binomtest(min(n01, n10), n=n01 + n10, p=0.5,
                           alternative="two-sided").pvalue
        print(f"\n Paired McNemar (Ours-hybrid vs Ours-auto_only, both at cool={args.max_cooling_steps}):")
        print(f"   n_paired={len(common)}, n00={n00} n01={n01} "
              f"n10={n10} n11={n11}, p={pv:.6e}")
    except FileNotFoundError:
        print(f"  paired McNemar skipped: anchor CSV not found at {args.anchor_csv}")
    except Exception as e:
        print(f"  paired McNemar failed: {e}")


if __name__ == "__main__":
    main()
