"""
eval_autocool_ablation.py — multi-scheduler paired eval with env auto-cool
DISABLED (max_cooling_steps=0), exposing intrinsic policy-level safety.

Schedulers tested (paired per-episode, identical env seeds across all four):
  1. HEFT             — classical heuristic (no learning)
  2. Decima-thermal   — homog GCN + REINFORCE + thermal-aware reward
  3. Ours-NoRCEdge    — hetero + cross-attn + thermal-aware reward, no RC-edge
  4. Ours-auto_only   — full architecture (hetero + cross-attn + RC-edge + thermal reward)

For each episode i in [seed_base, seed_base + n), all four schedulers see
the SAME env seed → same DAG + initial temp + same RC physics.

Output:
  eval_results/autocool_ablation/paired.csv
  Per-scheduler viol_rate + paired McNemar against all baselines.

Usage:
  python -m cpo_thermal_v2.scripts.eval_autocool_ablation --n_episodes 500
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Callable

import numpy as np
from scipy.stats import binomtest, wilcoxon

from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
from cpo_thermal_v2.envs.reward_shaping import RewardConfig

# Reuse run_one_episode logic from the paired script
from cpo_thermal_v2.scripts.eval_ours_no_rc_edge_paired import run_one_episode


T_PEN = 80.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--seed_base", type=int, default=100000)
    parser.add_argument("--num_nodes", type=int, default=17)
    parser.add_argument("--dags_per_episode", type=int, default=20)
    parser.add_argument("--initial_temp_low",  type=float, default=60.0)
    parser.add_argument("--initial_temp_high", type=float, default=75.0)
    parser.add_argument("--max_cooling_steps", type=int, default=0,
                        help="set 0 to disable env auto-cool safety net")
    parser.add_argument("--dataset_path", type=str,
                        default="./data_pipeline/process/alibaba_dags_v2.json")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/autocool_ablation_hot_n17_nocool")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    common_env_kwargs = dict(
        num_nodes          = args.num_nodes,
        dt                 = 1e-3,
        thermal_target     = 75.0,
        thermal_guardband  = 80.0,
        thermal_critical   = 85.0,
        mask_temp          = 82.0,
        action_mode        = "auto_only",
        initial_temp_range = (args.initial_temp_low, args.initial_temp_high),
        max_dag_size       = None,
        dags_per_episode   = args.dags_per_episode,
        temp_rise_per_ms_asic = 0.08,
        temp_rise_per_ms_oe   = 0.18,
        oe_active_power    = 40.0,
        max_cooling_steps  = args.max_cooling_steps,
        reward_config      = RewardConfig(),
        dataset_path       = args.dataset_path,
    )

    # Build envs — HEFT/Decima/auto_only use disable_rc_edge=False, NoRCEdge uses True
    env_default = CPOThermalDAGEnvV2(**common_env_kwargs, disable_rc_edge=False)
    env_norc    = CPOThermalDAGEnvV2(**common_env_kwargs, disable_rc_edge=True)

    # --- Build the 4 schedulers ---
    from cpo_thermal_v2.baselines.heft import HEFTScheduler
    from cpo_thermal_v2.baselines.decima_true import DecimaTrueScheduler
    from cpo_thermal_v2.baselines import TrainedPPOScheduler

    sched_heft = HEFTScheduler(num_nodes=args.num_nodes,
                                action_mode="auto_only")
    sched_dec  = DecimaTrueScheduler(
        ckpt_path="checkpoints/decima_true_thermal_N17/best.pt",
        num_nodes=args.num_nodes,
        deterministic=True, device="cpu",
    )
    sched_norc = TrainedPPOScheduler(
        ckpt_path="checkpoints/ours_no_rc_edge_N17/best.pt",
        action_mode="auto_only", deterministic=True, device="cpu",
        scheduler_label="Ours-NoRCEdge",
    )
    sched_auto = TrainedPPOScheduler(
        ckpt_path="checkpoints/stage1_auto_only_N17/best.pt",
        action_mode="auto_only", deterministic=True, device="cpu",
        scheduler_label="Ours-auto_only",
    )

    schedulers: List = [
        ("HEFT",            sched_heft,  env_default),
        ("Decima-thermal",  sched_dec,   env_default),
        ("Ours-NoRCEdge",   sched_norc,  env_norc),
        ("Ours-auto_only",  sched_auto,  env_default),
    ]

    rows: List[Dict] = []
    print(f"[eval] {args.n_episodes} paired eps, seed_base={args.seed_base}, "
          f"N={args.num_nodes}, T_amb∈[{args.initial_temp_low},{args.initial_temp_high}], "
          f"max_cooling_steps={args.max_cooling_steps}")

    for i in range(args.n_episodes):
        seed = args.seed_base + i
        for name, sched, env in schedulers:
            try:
                m = run_one_episode(env, sched, seed)
            except Exception as e:
                print(f"  ep {i:4d} {name}: exception {e}")
                m = dict(total_makespan_ms=0, peak_temp_episode=0,
                         violations_total=0, dag_completion_rate=0,
                         truncated=1, episode_return=0, is_unsafe=1)
            m.update(episode_id=i, scheduler=name)
            rows.append(m)
        if (i + 1) % 25 == 0:
            done = i + 1
            sums = {}
            for name, _, _ in schedulers:
                s = [r for r in rows if r["scheduler"] == name]
                sums[name] = sum(r["is_unsafe"] for r in s) / max(1, len(s))
            line = " | ".join(f"{n}={sums[n]:.4f}" for n in [s[0] for s in schedulers])
            print(f"  [{done:4d}/{args.n_episodes}] " + line)

    # Write CSV
    fields = ["episode_id", "scheduler", "total_makespan_ms",
              "peak_temp_episode", "violations_total", "dag_completion_rate",
              "truncated", "episode_return", "is_unsafe"]
    csv_path = out / "paired.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # --- per-scheduler summary + paired McNemar against Ours-auto_only ---
    print("\n" + "=" * 72)
    print(f" Auto-cool ablation | hot N=17 | n={args.n_episodes} | seed_base={args.seed_base}")
    print(f" max_cooling_steps={args.max_cooling_steps}")
    print("=" * 72)

    sch_rows = {name: [r for r in rows if r["scheduler"] == name]
                for name, _, _ in schedulers}

    print(f" {'Scheduler':18s}  viol_rate    n_unsafe   mean_peak    mean_makespan")
    for name, _, _ in schedulers:
        s = sch_rows[name]
        viol = sum(r["is_unsafe"] for r in s) / len(s)
        n_u  = sum(r["is_unsafe"] for r in s)
        mp   = np.mean([r["peak_temp_episode"] for r in s])
        mk   = np.mean([r["total_makespan_ms"] for r in s])
        print(f" {name:18s}  {viol:8.4f}    {n_u:4d}/{len(s):<4d}   {mp:6.2f}°C   {mk:8.1f}ms")

    print()
    print(" Paired McNemar exact tests on is_unsafe (Ours-auto_only as anchor):")
    anchor = {r["episode_id"]: r["is_unsafe"] for r in sch_rows["Ours-auto_only"]}
    for name, _, _ in schedulers:
        if name == "Ours-auto_only":
            continue
        other = {r["episode_id"]: r["is_unsafe"] for r in sch_rows[name]}
        n01 = sum(1 for i in anchor
                  if anchor[i] == 0 and other.get(i, 0) == 1)
        n10 = sum(1 for i in anchor
                  if anchor[i] == 1 and other.get(i, 0) == 0)
        n11 = sum(1 for i in anchor
                  if anchor[i] == 1 and other.get(i, 0) == 1)
        n00 = sum(1 for i in anchor
                  if anchor[i] == 0 and other.get(i, 0) == 0)
        if n01 + n10 == 0:
            p = 1.0
        else:
            p = binomtest(min(n01, n10), n=n01 + n10, p=0.5,
                          alternative="two-sided").pvalue
        if p < 0.001: sig = "***"
        elif p < 0.01: sig = " **"
        elif p < 0.05: sig = "  *"
        else: sig = " ns"
        print(f"   auto_only vs {name:16s}: n00={n00:3d} n01={n01:3d} "
              f"n10={n10:3d} n11={n11:3d}  p={p:.6f}  {sig}")

    print()
    print(f" CSV: {csv_path}")


if __name__ == "__main__":
    main()
