"""
evaluation/runner.py — Evaluation grid runner
=============================================

Drives the full evaluation matrix::

    schedulers × topology_sizes × episodes

For each cell, builds the env at that size, instantiates the scheduler,
runs N episodes, and returns the per-episode + per-DAG records.

Output structure
----------------
The runner produces three pandas DataFrames (returned and optionally
saved to CSV):

    df_episodes : one row per episode, all summary fields
    df_dags     : one row per DAG completed, with normalized makespan etc
    df_steps    : (optional, large) one row per step — usually skipped
                   except for a single representative episode per cell

These DataFrames feed into ``evaluation.plots`` and ``evaluation.tables``
for visualisation and LaTeX generation.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
from cpo_thermal_v2.envs.reward_shaping import RewardConfig

from cpo_thermal_v2.evaluation.metrics import (
    EpisodeRecorder, run_episode,
)


# =====================================================================
# Single-cell driver
# =====================================================================
def evaluate_cell(
    env_kwargs:    Dict[str, Any],
    scheduler,
    num_episodes:  int,
    seed_base:     int = 100_000,
    verbose:       bool = True,
) -> Tuple[List, List]:
    """Run ``num_episodes`` of ``scheduler`` against an env built from
    ``env_kwargs``.  Returns ``(episode_records, dag_records)``.
    """
    env = CPOThermalDAGEnvV2(**env_kwargs)
    episode_records = []
    dag_records     = []

    cell_start = time.time()
    for ep in range(num_episodes):
        recorder = EpisodeRecorder(
            scheduler_name = scheduler.name,
            num_nodes      = env_kwargs["num_nodes"],
            action_mode    = env_kwargs.get("action_mode", "auto_only"),
            episode_id     = ep,
        )
        ep_rec = run_episode(
            env       = env,
            scheduler = scheduler,
            recorder  = recorder,
            seed      = seed_base + ep,
        )
        episode_records.append(ep_rec)
        dag_records.extend(recorder.dags)

        if verbose and ((ep + 1) % max(1, num_episodes // 10) == 0):
            elapsed = time.time() - cell_start
            rate = (ep + 1) / max(1e-3, elapsed)
            eta  = (num_episodes - ep - 1) / max(1e-3, rate)
            print(f"      [{scheduler.name} N={env_kwargs['num_nodes']}] "
                  f"{ep+1}/{num_episodes} eps  "
                  f"({rate:.1f} ep/s, ETA {eta:.0f}s)  "
                  f"makespan={ep_rec.total_makespan_ms:7.1f} "
                  f"peak_T={ep_rec.peak_temp_episode:5.1f} "
                  f"viol={ep_rec.violations_total}")

    env.close() if hasattr(env, "close") else None
    return episode_records, dag_records


# =====================================================================
# Full grid
# =====================================================================
def run_grid(
    base_env_kwargs:   Dict[str, Any],
    scheduler_factories: List[Tuple[str, Any]],
    num_nodes_list:    List[int],
    action_mode_list:  List[str],
    num_episodes:      int,
    output_dir:        str,
    seed_base:         int = 100_000,
    verbose:           bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full evaluation grid.

    Parameters
    ----------
    base_env_kwargs
        Defaults for env construction (T_target, T_pen, ..., dt, etc).
        Per-cell overrides for ``num_nodes`` and ``action_mode`` are
        applied automatically.
    scheduler_factories
        List of ``(scheduler_label, factory_fn)`` pairs.  Each
        ``factory_fn(num_nodes, action_mode) -> BaseScheduler`` returns
        a fresh scheduler for each cell.  We use factories rather than
        pre-built instances because some schedulers (HEFT,
        Thermal-HEFT) have num_nodes-dependent state.
    num_nodes_list
        List of topology sizes to sweep.  Default: [9, 13, 17, 24, 33].
    action_mode_list
        List of env action modes.  ``auto_only`` always works; for
        ``agent_only`` and ``hybrid`` the scheduler must produce a
        2-element action.  Schedulers that only know placement (HEFT,
        ThermalHEFT, RoundRobin) are silently restricted to
        ``auto_only``.
    num_episodes
        Episodes per (scheduler, num_nodes, action_mode) cell.
    output_dir
        Directory for CSV dumps; created if needed.

    Returns
    -------
    (df_episodes, df_dags) : tuple of pandas DataFrames
    """
    os.makedirs(output_dir, exist_ok=True)

    all_eps  = []
    all_dags = []

    grid_start = time.time()
    cells_done = 0
    cells_total = sum(
        1 for _ in scheduler_factories
        for _ in num_nodes_list
        for _ in action_mode_list
    )

    for sched_label, factory in scheduler_factories:
        for N in num_nodes_list:
            for mode in action_mode_list:
                cell_label = f"{sched_label} | N={N} | mode={mode}"
                if verbose:
                    print(f"\n[{cells_done+1}/{cells_total}] {cell_label}")

                # Build env kwargs for this cell
                env_kwargs = dict(base_env_kwargs)
                env_kwargs["num_nodes"]   = N
                env_kwargs["action_mode"] = mode

                # Build scheduler (some don't support factored actions)
                try:
                    scheduler = factory(num_nodes=N, action_mode=mode)
                except (NotImplementedError, ValueError) as e:
                    print(f"   skipping (factory: {e})")
                    cells_done += 1
                    continue

                eps, dags = evaluate_cell(
                    env_kwargs    = env_kwargs,
                    scheduler     = scheduler,
                    num_episodes  = num_episodes,
                    seed_base     = seed_base,
                    verbose       = verbose,
                )
                # Stamp the scheduler label on each record
                for r in eps:
                    r.scheduler = sched_label
                for d in dags:
                    d.scheduler_label = sched_label
                    d.num_nodes_cell  = N
                    d.action_mode_cell= mode

                all_eps.extend(eps)
                all_dags.extend(dags)
                cells_done += 1

                # Incremental save (in case the run is interrupted)
                _save_partial(all_eps, all_dags, output_dir)

    grid_elapsed = time.time() - grid_start
    if verbose:
        print(f"\n=== grid done in {grid_elapsed:.0f}s "
              f"({cells_total} cells, {len(all_eps)} eps total) ===")

    df_eps  = pd.DataFrame([asdict(r) for r in all_eps])
    df_dags = pd.DataFrame([_dag_to_dict(d) for d in all_dags])
    df_eps.to_csv(os.path.join(output_dir,  "episodes.csv"), index=False)
    df_dags.to_csv(os.path.join(output_dir, "dags.csv"),     index=False)
    return df_eps, df_dags


def _save_partial(eps_list, dags_list, output_dir: str) -> None:
    """Write incremental progress (overwrites every cell)."""
    if eps_list:
        pd.DataFrame([asdict(r) for r in eps_list]).to_csv(
            os.path.join(output_dir, "episodes_partial.csv"), index=False,
        )
    if dags_list:
        pd.DataFrame([_dag_to_dict(d) for d in dags_list]).to_csv(
            os.path.join(output_dir, "dags_partial.csv"), index=False,
        )


def _dag_to_dict(d) -> dict:
    """Convert a DAGRecord (with extra fields stamped on) to a flat dict."""
    base = asdict(d) if hasattr(d, "__dataclass_fields__") else dict(vars(d))
    # Pull stamped attributes added by run_grid
    for extra in ("scheduler_label", "num_nodes_cell", "action_mode_cell"):
        if hasattr(d, extra):
            base[extra] = getattr(d, extra)
    return base
