"""
eval_ours_no_rc_edge_paired.py — paired Ours-NoRCEdge vs Ours-auto_only eval.

Runs n_episodes paired episodes at hot N=17 against both Ours-auto_only
(checkpoints/stage1_auto_only_N17/best.pt, env emits RC-edge attribute on
proc-proc edges) and Ours-NoRCEdge
(checkpoints/ours_no_rc_edge_N17/best.pt, env emits empty edges_p2p so
the encoder's p2p layers contribute no signal).

For every episode i in [seed_base, seed_base + n_episodes), both envs are
reset with the same seed: the env's thermal dynamics (RC matrix
simulation) are deterministic given the seed and produce the SAME
physical trajectory under the same policy actions. Different
trajectories can only come from policy disagreement on the placement
action sequence.

Output
------
eval_results/ours_no_rc_edge_paired/paired.csv with rows:
   episode_id, scheduler, num_nodes, action_mode, total_makespan_ms,
   peak_temp_episode, violations_total, dag_completion_rate,
   truncated, episode_return, is_unsafe (1 = peak >= T_pen)

Plus stdout summary: viol_rate per scheduler, paired McNemar exact test.

Usage
-----
   python -m cpo_thermal_v2.scripts.eval_ours_no_rc_edge_paired \\
       --n_episodes 500 --seed_base 100000

Mac CPU runtime: ~25-30 min for 500 episodes × 2 schedulers.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
from cpo_thermal_v2.envs.reward_shaping import RewardConfig
from cpo_thermal_v2.baselines import TrainedPPOScheduler


T_PEN = 80.0


def run_one_episode(
    env: CPOThermalDAGEnvV2,
    scheduler: TrainedPPOScheduler,
    seed: int,
) -> Dict:
    """Return a dict of per-episode metrics."""
    obs, info = env.reset(seed=seed)
    scheduler.reset(obs, info)

    total_makespan = 0.0
    peak_temp = -np.inf
    violations = 0
    n_dags_done = 0
    total_return = 0.0
    truncated_flag = False
    step_count = 0

    while True:
        action = scheduler.schedule(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        total_makespan += float(info.get("actual_workload_ms", 0.0))
        peak_temp = max(peak_temp, float(info.get("max_temp", 0.0)))
        violations += int(info.get("violations_step", 0))
        if info.get("dag_done", False):
            n_dags_done += 1
        if truncated:
            truncated_flag = True
        step_count += 1
        if terminated or truncated:
            break
        if step_count > 5000:    # paranoid loop guard
            break

    completion_rate = float(n_dags_done) / max(1, env.dags_per_episode)
    is_unsafe = int(peak_temp >= T_PEN)

    return {
        "total_makespan_ms":  round(total_makespan, 3),
        "peak_temp_episode":  round(peak_temp, 4),
        "violations_total":   int(violations),
        "dag_completion_rate": round(completion_rate, 4),
        "truncated":          int(truncated_flag),
        "episode_return":     round(total_return, 4),
        "is_unsafe":          is_unsafe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--seed_base", type=int, default=100000)
    parser.add_argument("--num_nodes", type=int, default=17)
    parser.add_argument("--dags_per_episode", type=int, default=20)
    parser.add_argument("--initial_temp_low",  type=float, default=60.0)
    parser.add_argument("--initial_temp_high", type=float, default=75.0)
    parser.add_argument("--ckpt_auto", type=str,
                        default="checkpoints/stage1_auto_only_N17/best.pt")
    parser.add_argument("--ckpt_norcedge", type=str,
                        default="checkpoints/ours_no_rc_edge_N17/best.pt")
    parser.add_argument("--dataset_path", type=str,
                        default="./data_pipeline/process/alibaba_dags_v2.json")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/ours_no_rc_edge_paired")
    parser.add_argument("--max_cooling_steps", type=int, default=100,
                        help="env auto-cool budget; set 0 to disable env safety net")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "paired.csv"

    # Common kwargs (env physics identical between the two envs; only
    # disable_rc_edge differs, which only affects graph_obs not dynamics).
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
        reward_config      = RewardConfig(),  # default thermal-aware reward
        dataset_path       = args.dataset_path,
    )

    env_auto     = CPOThermalDAGEnvV2(**common_env_kwargs, disable_rc_edge=False)
    env_norcedge = CPOThermalDAGEnvV2(**common_env_kwargs, disable_rc_edge=True)

    sched_auto = TrainedPPOScheduler(
        ckpt_path       = args.ckpt_auto,
        action_mode     = "auto_only",
        deterministic   = True,
        device          = "cpu",
        scheduler_label = "Ours-auto_only",
    )
    sched_norcedge = TrainedPPOScheduler(
        ckpt_path       = args.ckpt_norcedge,
        action_mode     = "auto_only",
        deterministic   = True,
        device          = "cpu",
        scheduler_label = "Ours-NoRCEdge",
    )

    rows: List[Dict] = []
    print(f"[eval] paired {args.n_episodes} eps, seed_base={args.seed_base}, "
          f"hot N={args.num_nodes} T_amb∈[{args.initial_temp_low},{args.initial_temp_high}]")

    for i in range(args.n_episodes):
        seed = args.seed_base + i

        m_auto = run_one_episode(env_auto, sched_auto, seed)
        m_auto.update(episode_id=i, scheduler="Ours-auto_only")
        rows.append(m_auto)

        m_nor = run_one_episode(env_norcedge, sched_norcedge, seed)
        m_nor.update(episode_id=i, scheduler="Ours-NoRCEdge")
        rows.append(m_nor)

        if (i + 1) % 25 == 0:
            done = i + 1
            # Running viol rate per scheduler
            ao = [r for r in rows if r["scheduler"] == "Ours-auto_only"]
            nr = [r for r in rows if r["scheduler"] == "Ours-NoRCEdge"]
            ao_viol = sum(r["is_unsafe"] for r in ao) / max(1, len(ao))
            nr_viol = sum(r["is_unsafe"] for r in nr) / max(1, len(nr))
            print(f"  [{done:4d}/{args.n_episodes}] "
                  f"auto viol={ao_viol:.4f}  NoRCEdge viol={nr_viol:.4f}")

    # Write CSV
    fields = ["episode_id", "scheduler", "total_makespan_ms",
              "peak_temp_episode", "violations_total", "dag_completion_rate",
              "truncated", "episode_return", "is_unsafe"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    # ----- paired McNemar exact test -----
    auto_unsafe = {r["episode_id"]: r["is_unsafe"]
                   for r in rows if r["scheduler"] == "Ours-auto_only"}
    nor_unsafe  = {r["episode_id"]: r["is_unsafe"]
                   for r in rows if r["scheduler"] == "Ours-NoRCEdge"}
    n01 = sum(1 for i in auto_unsafe if auto_unsafe[i] == 0 and nor_unsafe[i] == 1)
    n10 = sum(1 for i in auto_unsafe if auto_unsafe[i] == 1 and nor_unsafe[i] == 0)
    n11 = sum(1 for i in auto_unsafe if auto_unsafe[i] == 1 and nor_unsafe[i] == 1)
    n00 = sum(1 for i in auto_unsafe if auto_unsafe[i] == 0 and nor_unsafe[i] == 0)

    # exact 2-sided binomial test on min(n01, n10) under Bin(n01+n10, 0.5)
    from scipy.stats import binomtest
    if n01 + n10 == 0:
        p_value = 1.0
    else:
        bt = binomtest(min(n01, n10), n=n01 + n10, p=0.5, alternative="two-sided")
        p_value = float(bt.pvalue)

    auto_total_viol = sum(auto_unsafe.values()) / len(auto_unsafe)
    nor_total_viol  = sum(nor_unsafe.values())  / len(nor_unsafe)
    auto_total_mk = np.mean([r["total_makespan_ms"]
                              for r in rows if r["scheduler"] == "Ours-auto_only"])
    nor_total_mk  = np.mean([r["total_makespan_ms"]
                              for r in rows if r["scheduler"] == "Ours-NoRCEdge"])
    auto_total_peak = np.mean([r["peak_temp_episode"]
                                for r in rows if r["scheduler"] == "Ours-auto_only"])
    nor_total_peak  = np.mean([r["peak_temp_episode"]
                                for r in rows if r["scheduler"] == "Ours-NoRCEdge"])

    print()
    print("=" * 64)
    print(f" Paired hot N={args.num_nodes} | n = {args.n_episodes} | seed_base = {args.seed_base}")
    print("=" * 64)
    print(f" Ours-auto_only  : viol={auto_total_viol:.4f}  makespan={auto_total_mk:8.1f}  peak={auto_total_peak:.2f}")
    print(f" Ours-NoRCEdge   : viol={nor_total_viol:.4f}  makespan={nor_total_mk:8.1f}  peak={nor_total_peak:.2f}")
    print()
    print(f" McNemar 2x2 (auto x NoRCEdge on is_unsafe):")
    print(f"   n00 (both safe)               = {n00}")
    print(f"   n01 (auto safe, NoRC unsafe)  = {n01}")
    print(f"   n10 (auto unsafe, NoRC safe)  = {n10}")
    print(f"   n11 (both unsafe)             = {n11}")
    print(f" Discordant pairs: n01 + n10 = {n01 + n10}")
    print(f" McNemar exact p-value (2-sided binomial on min): {p_value:.6f}")
    if p_value < 0.001:
        sig = "*** (p<0.001)"
    elif p_value < 0.01:
        sig = "**  (p<0.01)"
    elif p_value < 0.05:
        sig = "*   (p<0.05)"
    else:
        sig = "ns  (not significant)"
    print(f" Significance: {sig}")
    print()
    if nor_total_viol > auto_total_viol and p_value < 0.05:
        print(" Reading: Ours-NoRCEdge is paired-significantly WORSE on safety.")
        print(" → RC-edge attribute IS contributing to safety; §6.2 claim STANDS.")
    elif abs(nor_total_viol - auto_total_viol) < 1e-6 or p_value >= 0.05:
        print(" Reading: paired-not-significant difference.")
        print(" → RC-edge attribute NOT significantly contributing.")
        print(" → §6.2 inductive-bias claim may need WEAKENING in HK-paper-8.")
    print("=" * 64)
    print(f" CSV: {csv_path}")


if __name__ == "__main__":
    main()
