"""
cpo_thermal_v2.evaluation — Stage E evaluation, plotting, and tables
=====================================================================

Public exports
--------------
    EpisodeRecorder       — per-episode metric collection
    run_episode           — drive one episode under a given scheduler
    run_grid              — run the full schedulers × N × modes matrix
    evaluate_cell         — single (scheduler, N, mode) cell
    plot_scaling_curves   — Fig 1: metric vs N for each scheduler
    plot_box_per_metric   — Fig 2: per-metric distribution box plots
    plot_rho_analysis     — Fig 3: makespan vs DAG ρ scatter
    plot_proc_utilisation — Fig 4: per-proc usage heatmap
    main_results_table    — Tab 1: scheduler × metric (mean ± stderr)
    scaling_table         — Tab 2: scheduler × N
    ablation_table        — Tab 3: Ours-{auto_only,agent_only,hybrid}
    emit_all_tables       — convenience: all three tables to a directory

Used as a CLI through ``evaluation.evaluate``::

    PYTHONPATH=. python -m cpo_thermal_v2.evaluation.evaluate \\
        --config cpo_thermal_v2/configs/eval_scaling.yaml \\
        --override eval.checkpoint_path=checkpoints/stage2_hybrid_N17/best.pt
"""
from .metrics import (
    EpisodeRecorder, EpisodeRecord, DAGRecord, StepRecord, run_episode,
)
from .runner  import evaluate_cell, run_grid
from .plots   import (
    plot_scaling_curves, plot_box_per_metric,
    plot_rho_analysis,   plot_proc_utilisation,
)
from .tables  import (
    main_results_table, scaling_table, ablation_table, emit_all_tables,
)

__all__ = [
    "EpisodeRecorder", "EpisodeRecord", "DAGRecord", "StepRecord",
    "run_episode", "evaluate_cell", "run_grid",
    "plot_scaling_curves", "plot_box_per_metric",
    "plot_rho_analysis",   "plot_proc_utilisation",
    "main_results_table",  "scaling_table", "ablation_table",
    "emit_all_tables",
]
