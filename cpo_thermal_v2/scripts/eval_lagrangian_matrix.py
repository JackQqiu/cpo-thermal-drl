"""
scripts/eval_lagrangian_matrix.py — R2.2 Lagrangian/RCPO head-to-head eval
==========================================================================

Paper revision R2.2.  Evaluates the constrained-PPO (Lagrangian / RCPO)
baseline head-to-head with Ours-hybrid on the STANDARD eval matrix, using
the SAME grid axes and the SAME paired seeds so the two policies are
compared apples-to-apples on every episode.

Both schedulers run at ``action_mode='hybrid'`` (the Lagrangian checkpoint
is a hybrid-mode ``PPOActorCritic`` with an extra ``value_cost`` head that
:class:`TrainedPPOScheduler` ignores — it loads with ``strict=False`` and
``act()`` never touches the cost head, so the placement+delay behaviour is
exactly the trained policy's).

Grid (mirrors eval_scaling.yaml over default.yaml):
    scheduler   ∈ {Ours-hybrid, Ours-Lagrangian}
    num_nodes   ∈ {9, 13, 17, 24, 33}      (CLI --nodes, for parallel split)
    action_mode ∈ {hybrid}
    episodes    = CLI --episodes (default 500), seed_base=100000

Env regime (eval_scaling.yaml per-cell overrides over default.yaml):
    initial_temp_range [50, 65], dags_per_episode 20, max_dag_size None,
    truncate_mode hard.

Aggregation — per (scheduler, N) row, written incrementally to CSV:
    [scheduler, num_nodes, n_ep, violation_rate, peak_T_mean,
     makespan_mean, completion_rate]

where
    violation_rate  = mean(violations_total > 0)         (any-violation episode)
    peak_T_mean     = mean(peak_temp_episode)
    makespan_mean   = mean(total_makespan_ms)
    completion_rate = mean(dag_completion_rate)           (DAGs completed /
                                                           DAGs available,
                                                           the paper metric)

Usage
-----
    # full grid (one process, all N)
    PYTHONPATH=. conda run -n cpo_rl python -m \
        cpo_thermal_v2.scripts.eval_lagrangian_matrix --episodes 500

    # single-N slice (for the parallel launcher)
    PYTHONPATH=. conda run -n cpo_rl python -m \
        cpo_thermal_v2.scripts.eval_lagrangian_matrix \
        --nodes 17 --episodes 500 --out repro_outputs/eval_cells/N17.csv

    # tiny end-to-end smoke
    OMP_NUM_THREADS=1 PYTHONPATH=. conda run -n cpo_rl python -m \
        cpo_thermal_v2.scripts.eval_lagrangian_matrix \
        --nodes 17 --episodes 8 --device cpu --out repro_outputs/eval_lag_smoke.csv
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cpo_thermal_v2.evaluation.runner import run_grid


# =====================================================================
# Constants (mirror default.yaml + eval_scaling.yaml)
# =====================================================================
DT = 0.001                              # 1 ms RC step (default.yaml)
SEED_BASE = 100_000                     # paired seeds (eval_scaling.yaml)

# eval_scaling.yaml per-cell overrides over default.yaml
INITIAL_TEMP_RANGE = (50.0, 65.0)       # eval_scaling.yaml moderate-hot regime
DAGS_PER_EPISODE = 20                   # eval_scaling.yaml
DEFAULT_NODES = [9, 13, 17, 24, 33]     # eval_scaling.yaml num_nodes_list
DEFAULT_EPISODES = 500                  # eval_scaling.yaml num_episodes

OURS_HYBRID_CKPT = "checkpoints/stage2_hybrid_N17/best.pt"
LAGRANGIAN_CKPT = "checkpoints/lagrangian_constrained_N17/best.pt"


# =====================================================================
# base env kwargs (mirror default.yaml env defaults + eval overrides)
# =====================================================================
def make_base_env_kwargs() -> Dict:
    """Construct the env kwargs shared by every cell.

    Mirrors default.yaml env defaults with eval_scaling.yaml per-cell
    overrides (initial_temp_range [50,65], dags_per_episode 20,
    max_dag_size None, hard truncation).  ``num_nodes`` and
    ``action_mode`` are overlaid per-cell by ``run_grid``; we still set
    ``num_nodes`` here so the dict is a complete, valid env spec.
    rc_matrix_dir=None → env autodiscovers data/thermal_matrics/N{N}/
    per topology size.
    """
    return dict(
        # topology / step
        num_nodes=17,                   # overlaid per-cell by run_grid
        dt=DT,
        # action mode overlaid per-cell by run_grid (we pass hybrid)
        K_delay=5,
        delay_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        max_agent_delay_ms=50.0,
        # thermal thresholds (default.yaml)
        thermal_target=75.0,
        thermal_guardband=80.0,
        thermal_critical=85.0,
        mask_temp=82.0,
        # auto-cool (default.yaml)
        max_cooling_steps=100,
        precool_target_temp=None,
        # power model (default.yaml)
        asic_active_power=150.0,
        oe_active_power=40.0,
        oe_serialization_power=20.0,
        cpo_bandwidth_gbps=800.0,
        oe_conversion_delay_ms=0.005,
        leakage_base_power=30.0,
        leakage_beta=0.015,
        # temp-rise heuristics (default.yaml)
        temp_rise_per_ms_asic=0.08,
        temp_rise_per_ms_oe=0.18,
        # evaluation regime (eval_scaling.yaml overrides)
        initial_temp_range=INITIAL_TEMP_RANGE,
        max_dag_size=None,
        dags_per_episode=DAGS_PER_EPISODE,
        truncate_mode="hard",
        soft_truncate_recovery_temp=75.0,
        max_soft_recovery_steps=200,
        # data
        dataset_path="./data_pipeline/process/alibaba_dags_v2.json",
        rc_matrix_dir=None,             # autodiscover per-N matrices
    )


# =====================================================================
# Scheduler factories — BOTH hybrid-mode trained policies
# =====================================================================
def build_scheduler_factories(device: str, deterministic: bool
                              ) -> List[Tuple[str, object]]:
    """Return ``[(label, factory), ...]`` for Ours-hybrid + Ours-Lagrangian.

    Both are built with :class:`TrainedPPOScheduler` exactly as
    evaluate.py does (deterministic, device, scheduler_label).  The
    Lagrangian checkpoint's extra ``value_cost`` head is silently
    dropped by the wrapper's ``strict=False`` load and never read at
    eval time.
    """
    from cpo_thermal_v2.baselines import TrainedPPOScheduler

    for label, ckpt in (("Ours-hybrid", OURS_HYBRID_CKPT),
                        ("Ours-Lagrangian", LAGRANGIAN_CKPT)):
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"[eval-lag] checkpoint missing for {label}: {ckpt}")

    def make_factory(ckpt_path: str, label: str):
        def _f(num_nodes: int, action_mode: str):
            if action_mode != "hybrid":
                raise ValueError(f"{label} only runs at action_mode=hybrid")
            return TrainedPPOScheduler(
                ckpt_path=ckpt_path,
                action_mode="hybrid",
                deterministic=deterministic,
                device=device,
                scheduler_label=label,
            )
        return _f

    return [
        ("Ours-hybrid", make_factory(OURS_HYBRID_CKPT, "Ours-hybrid")),
        ("Ours-Lagrangian", make_factory(LAGRANGIAN_CKPT, "Ours-Lagrangian")),
    ]


# =====================================================================
# Aggregation
# =====================================================================
def aggregate(df_eps: pd.DataFrame, num_nodes: int) -> List[Dict]:
    """Collapse this cell's df_episodes into per-scheduler summary rows.

    Uses the EpisodeRecord column names from evaluation/metrics.py:
        scheduler, violations_total, peak_temp_episode,
        total_makespan_ms, dag_completion_rate.

    completion_rate := mean(dag_completion_rate), where the per-episode
    dag_completion_rate = DAGs completed / DAGs available (the paper's
    completion metric).
    """
    rows: List[Dict] = []
    if df_eps.empty:
        return rows
    for sched, sub in df_eps.groupby("scheduler"):
        viol = sub["violations_total"].to_numpy(dtype=float)
        peak = sub["peak_temp_episode"].to_numpy(dtype=float)
        mk = sub["total_makespan_ms"].to_numpy(dtype=float)
        comp = sub["dag_completion_rate"].to_numpy(dtype=float)
        rows.append(dict(
            scheduler=str(sched),
            num_nodes=int(num_nodes),
            n_ep=int(len(sub)),
            violation_rate=float(np.mean(viol > 0)),
            peak_T_mean=float(np.mean(peak)),
            makespan_mean=float(np.mean(mk)),
            completion_rate=float(np.mean(comp)),
        ))
    return rows


# =====================================================================
# Main eval
# =====================================================================
def run_eval(nodes: List[int], num_episodes: int, device: str,
             deterministic: bool, out_csv: str) -> pd.DataFrame:
    base_kwargs = make_base_env_kwargs()
    factories = build_scheduler_factories(device, deterministic)

    work_dir = os.path.dirname(os.path.abspath(out_csv)) or "."
    os.makedirs(work_dir, exist_ok=True)
    grid_scratch = os.path.join(work_dir, "_grid_scratch_lag")

    cols = ["scheduler", "num_nodes", "n_ep", "violation_rate",
            "peak_T_mean", "makespan_mean", "completion_rate"]

    all_rows: List[Dict] = []
    for i, N in enumerate(nodes):
        print(f"\n========== N={N} ({i+1}/{len(nodes)}), "
              f"n_ep={num_episodes}, schedulers="
              f"{[f[0] for f in factories]} ==========")

        df_eps, _df_dags = run_grid(
            base_env_kwargs=base_kwargs,
            scheduler_factories=factories,
            num_nodes_list=[N],
            action_mode_list=["hybrid"],
            num_episodes=num_episodes,
            output_dir=grid_scratch,
            seed_base=SEED_BASE,            # SAME across all cells → paired
            verbose=True,
        )

        cell_rows = aggregate(df_eps, N)
        all_rows.extend(cell_rows)
        for r in cell_rows:
            print(f"   -> {r['scheduler']}: viol_rate={r['violation_rate']:.4f} "
                  f"peak_T_mean={r['peak_T_mean']:.2f} "
                  f"makespan_mean={r['makespan_mean']:.1f} "
                  f"completion_rate={r['completion_rate']:.4f}")

        # incremental save (in case the run is interrupted)
        pd.DataFrame(all_rows, columns=cols).to_csv(out_csv, index=False)

    df_out = pd.DataFrame(all_rows, columns=cols)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[eval-lag] wrote {out_csv}")
    return df_out


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    p = argparse.ArgumentParser(
        description="R2.2 Lagrangian/RCPO vs Ours-hybrid head-to-head eval "
                    "(standard eval matrix, paired seeds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nodes", nargs="+", type=int, default=DEFAULT_NODES,
                   help="Topology sizes to evaluate (space-separated subset "
                        "for parallel split).")
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                   help="Episodes per (scheduler, N) cell.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device for the trained policies.")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Sample actions instead of argmax (default argmax).")
    p.add_argument("--out", type=str,
                   default="repro_outputs/eval_lagrangian_matrix.csv",
                   help="Output CSV path.")
    args = p.parse_args()

    df = run_eval(
        nodes=args.nodes,
        num_episodes=args.episodes,
        device=args.device,
        deterministic=not args.no_deterministic,
        out_csv=args.out,
    )

    # Pretty final table
    print("\n================ RESULT TABLE ================")
    if not df.empty:
        with pd.option_context("display.width", 160,
                               "display.max_columns", None):
            print(df.to_string(index=False))
    else:
        print("(empty)")


if __name__ == "__main__":
    main()
