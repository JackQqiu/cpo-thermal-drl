"""
cpo_thermal_v2 — v2 reconstruction package
==========================================

A clean, physics-aligned re-implementation of the CPO thermal-aware DAG
scheduling system, organised into subpackages by responsibility:

    cpo_thermal_v2/
    ├── envs/           — environment + reward + RC dynamics + DAG parser
    ├── models/         — Stage C GNN encoder, factored actor, dual critic
    ├── data_pipeline/  — offline DAG enrichment + RC matrix generation
    ├── training/       — Stage D training loop + config loader (WIP)
    └── configs/        — YAML configuration files

User-facing API (most common imports)
-------------------------------------
::

    # Stage A + B (numpy-only, always available):
    from cpo_thermal_v2 import (
        CPOThermalDAGEnvV2, MicroserviceDAG, RCThermalDynamics,
        compute_reward, compute_reward_channels, RewardConfig,
    )

    # Stage C (requires torch + torch_geometric):
    from cpo_thermal_v2 import (
        PPOActorCritic, HeteroEncoder, CrossAttentionActor, DualCritic,
        build_batch, graph_obs_to_hetero_data,
    )

    # Sub-package access (always works):
    from cpo_thermal_v2 import envs, models, data_pipeline, training

The flat re-exports are preserved for convenience and to ease migration
from earlier flat-layout snapshots.
"""
# -----------------------------------------------------------------
# Stage A + B (numpy-only)
# -----------------------------------------------------------------
from .envs import (
    CPOThermalDAGEnvV2, CPOThermalDAGEnv,
    RCThermalDynamics, MicroserviceDAG,
    compute_reward, compute_reward_channels, decompose_reward, RewardConfig,
)

__all__ = [
    "CPOThermalDAGEnvV2", "CPOThermalDAGEnv",
    "RCThermalDynamics", "MicroserviceDAG",
    "compute_reward", "compute_reward_channels",
    "decompose_reward", "RewardConfig",
]

# -----------------------------------------------------------------
# Stage C (torch + torch_geometric — soft import)
# -----------------------------------------------------------------
try:
    from .models import (
        HeteroEncoder, graph_obs_to_hetero_data,
        CrossAttentionActor, DualCritic,
        PPOActorCritic, build_batch,
    )
    __all__ += [
        "HeteroEncoder", "graph_obs_to_hetero_data",
        "CrossAttentionActor", "DualCritic",
        "PPOActorCritic", "build_batch",
    ]
except ImportError:                            # pragma: no cover
    # torch/torch_geometric not present.  Stage C symbols simply aren't
    # re-exported; users on the GPU server still get the full package.
    pass

# -----------------------------------------------------------------
# Stage E (baselines + evaluation; require torch only for some baselines)
# -----------------------------------------------------------------
try:
    from .baselines import (
        BaseScheduler,
        RoundRobinScheduler, HEFTScheduler, ThermalHEFTScheduler,
    )
    __all__ += [
        "BaseScheduler",
        "RoundRobinScheduler", "HEFTScheduler", "ThermalHEFTScheduler",
    ]
    # Trained-model baselines depend on torch — soft-import them
    try:
        from .baselines import DecimaScheduler, TrainedPPOScheduler
        __all__ += ["DecimaScheduler", "TrainedPPOScheduler"]
    except ImportError:
        pass
except ImportError:                            # pragma: no cover
    pass

try:
    from .evaluation import (
        EpisodeRecorder, run_episode, evaluate_cell, run_grid,
        plot_scaling_curves, plot_box_per_metric,
        plot_rho_analysis,   plot_proc_utilisation,
        main_results_table, scaling_table, ablation_table, emit_all_tables,
    )
    __all__ += [
        "EpisodeRecorder", "run_episode", "evaluate_cell", "run_grid",
        "plot_scaling_curves", "plot_box_per_metric",
        "plot_rho_analysis",   "plot_proc_utilisation",
        "main_results_table", "scaling_table", "ablation_table",
        "emit_all_tables",
    ]
except ImportError:                            # pragma: no cover
    pass
