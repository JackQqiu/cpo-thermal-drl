"""
evaluation/evaluate.py — Stage E main entrypoint
=================================================

Loads the eval config, builds factory functions for all schedulers,
runs the grid, and emits the publication artifacts (CSVs, plots, LaTeX
tables).

Invocation::

    PYTHONPATH=. python -m cpo_thermal_v2.evaluation.evaluate \\
        --config cpo_thermal_v2/configs/eval_scaling.yaml \\
        --override eval.num_episodes=100 \\
        --override eval.checkpoint_path=checkpoints/stage2_hybrid_N17/best.pt

Output structure (under ``eval.output_dir``)::

    <output_dir>/
    ├── episodes.csv          — all per-episode records
    ├── dags.csv              — all per-DAG records
    ├── plots/
    │   ├── scaling_*.pdf
    │   ├── box_*.pdf
    │   ├── rho_analysis_*.pdf
    │   └── proc_util_*.pdf
    └── tables/
        ├── tab1_main.tex
        ├── tab2_scaling_makespan.tex
        ├── tab2_scaling_violations.tex
        └── tab3_ablation.tex
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import pandas as pd

from cpo_thermal_v2.training.config_loader import (
    load_config, merge_cli_overrides,
)
from cpo_thermal_v2.evaluation.runner import run_grid


# =====================================================================
# Scheduler factory builders
# =====================================================================
def _build_scheduler_factories(eval_cfg: Dict[str, Any]):
    """Return list of (label, factory_fn) pairs.

    Factory signature: ``factory(num_nodes, action_mode) -> BaseScheduler``.
    Schedulers that don't support a given action_mode raise ValueError
    inside the factory; runner skips those cells.
    """
    from cpo_thermal_v2.baselines import (
        RoundRobinScheduler, HEFTScheduler, ThermalHEFTScheduler,
    )

    factories = []

    # --- Classical baselines (placement only) ---
    def rr_factory(num_nodes: int, action_mode: str):
        if action_mode != "auto_only":
            # RR doesn't know about delay; only meaningful in auto_only
            raise ValueError("RoundRobin only supports auto_only env")
        return RoundRobinScheduler(action_mode=action_mode)
    factories.append(("RoundRobin", rr_factory))

    def heft_factory(num_nodes: int, action_mode: str):
        if action_mode != "auto_only":
            raise ValueError("HEFT only supports auto_only env")
        return HEFTScheduler(num_nodes=num_nodes, action_mode=action_mode)
    factories.append(("HEFT", heft_factory))

    def th_factory(num_nodes: int, action_mode: str):
        if action_mode != "auto_only":
            raise ValueError("ThermalHEFT only supports auto_only env")
        return ThermalHEFTScheduler(
            num_nodes=num_nodes, action_mode=action_mode,
            T_target=eval_cfg.get("T_target", 75.0),
        )
    factories.append(("ThermalHEFT", th_factory))

    # --- Trained model variants ---
    ckpt = eval_cfg.get("checkpoint_path")
    if not ckpt:
        print("[evaluate] no checkpoint_path in config; skipping trained "
              "schedulers.")
        return factories

    if not os.path.exists(ckpt):
        print(f"[evaluate] checkpoint not found at {ckpt}; "
              "skipping trained schedulers.")
        return factories

    # Decima (loads ckpt with thermal features masked)
    try:
        from cpo_thermal_v2.baselines import DecimaScheduler

        def decima_factory(num_nodes: int, action_mode: str):
            if action_mode != "auto_only":
                raise ValueError("Decima baseline only runs in auto_only mode")
            return DecimaScheduler(
                ckpt_path     = ckpt,
                num_nodes     = num_nodes,
                action_mode   = action_mode,
                deterministic = bool(eval_cfg.get("deterministic", True)),
                device        = eval_cfg.get("device", "cpu"),
            )
        factories.append(("Decima", decima_factory))
    except ImportError as e:
        print(f"[evaluate] DecimaScheduler unavailable: {e}")

    # Trained PPO — three variants (auto_only / agent_only / hybrid),
    # all using the same checkpoint.  ``action_mode`` is honoured by
    # both the env and the model wrapper.
    try:
        from cpo_thermal_v2.baselines import TrainedPPOScheduler

        for ours_mode in eval_cfg.get("action_mode_list",
                                       ["auto_only", "agent_only", "hybrid"]):
            label = f"Ours-{ours_mode}"

            def make_ours_factory(_mode=ours_mode, _label=label):
                def _f(num_nodes: int, action_mode: str):
                    # Only build this scheduler if env's action_mode matches
                    if action_mode != _mode:
                        raise ValueError(
                            f"{_label} only runs at action_mode={_mode}"
                        )
                    return TrainedPPOScheduler(
                        ckpt_path       = ckpt,
                        action_mode     = _mode,
                        deterministic   = bool(eval_cfg.get("deterministic", True)),
                        device          = eval_cfg.get("device", "cpu"),
                        scheduler_label = _label,
                    )
                return _f
            factories.append((label, make_ours_factory()))
    except ImportError as e:
        print(f"[evaluate] TrainedPPOScheduler unavailable: {e}")

    return factories


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPO v2 Stage E — evaluation and figure generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to eval YAML config.")
    parser.add_argument("--override", action="append", default=[],
                        help="Override config (repeat); e.g. "
                             "--override eval.num_episodes=100")
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip running the grid; load existing CSVs "
                             "and just regenerate plots/tables.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)
    eval_cfg = cfg["eval"]

    output_dir = eval_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ---- Grid run ----
    if not args.skip_grid:
        # Build env defaults from main env config
        base_env_kwargs = dict(cfg["env"])
        # Filter to keys the env constructor actually accepts; reuse the
        # robust filtering from env_factory.
        from cpo_thermal_v2.training.env_factory import _make_env_kwargs
        # _make_env_kwargs expects a top-level cfg; we fake one:
        fake_cfg = {"env": dict(base_env_kwargs)}
        if "reward" in cfg:
            fake_cfg["reward"] = cfg["reward"]
        # Apply per-cell overrides from eval section
        if "initial_temp_range" in eval_cfg:
            fake_cfg["env"]["initial_temp_range"] = eval_cfg["initial_temp_range"]
        if "max_dag_size" in eval_cfg:
            fake_cfg["env"]["max_dag_size"] = eval_cfg["max_dag_size"]
        if "dags_per_episode" in eval_cfg:
            fake_cfg["env"]["dags_per_episode"] = eval_cfg["dags_per_episode"]
        env_kwargs_clean = _make_env_kwargs(fake_cfg)
        # The runner will overlay num_nodes / action_mode per-cell

        # Build scheduler factories
        factories = _build_scheduler_factories(eval_cfg)
        if not factories:
            print("[evaluate] no schedulers available — exiting.")
            sys.exit(1)
        print(f"[evaluate] schedulers: {[f[0] for f in factories]}")

        df_eps, df_dags = run_grid(
            base_env_kwargs    = env_kwargs_clean,
            scheduler_factories= factories,
            num_nodes_list     = eval_cfg.get("num_nodes_list",
                                               [9, 13, 17, 24, 33]),
            action_mode_list   = eval_cfg.get("action_mode_list",
                                               ["auto_only", "agent_only",
                                                "hybrid"]),
            num_episodes       = int(eval_cfg.get("num_episodes", 100)),
            output_dir         = output_dir,
            seed_base          = int(eval_cfg.get("seed_base", 100_000)),
            verbose            = True,
        )
    else:
        df_eps  = pd.read_csv(os.path.join(output_dir, "episodes.csv"))
        df_dags = pd.read_csv(os.path.join(output_dir, "dags.csv"))
        print(f"[evaluate] loaded existing CSVs: "
              f"{len(df_eps)} eps, {len(df_dags)} DAGs")

    # ---- Plots ----
    print(f"\n[evaluate] generating plots …")
    from cpo_thermal_v2.evaluation.plots import (
        plot_scaling_curves, plot_box_per_metric, plot_rho_analysis,
        plot_proc_utilisation,
    )
    plots_dir = os.path.join(output_dir, "plots")
    plot_scaling_curves(df_eps, out_dir=plots_dir)
    plot_box_per_metric(df_eps, out_dir=plots_dir, num_nodes_focus=17)
    plot_proc_utilisation(df_eps, out_dir=plots_dir, num_nodes_focus=17)
    if not df_dags.empty:
        plot_rho_analysis(df_dags, out_dir=plots_dir, num_nodes_focus=17)
    print(f"  → {plots_dir}/")

    # ---- LaTeX tables ----
    print(f"\n[evaluate] generating LaTeX tables …")
    from cpo_thermal_v2.evaluation.tables import emit_all_tables
    tables_dir = os.path.join(output_dir, "tables")
    emit_all_tables(df_eps, out_dir=tables_dir, num_nodes_main=17)
    print(f"  → {tables_dir}/")

    print(f"\n=== Stage E evaluation complete  →  {output_dir} ===")


if __name__ == "__main__":
    main()
