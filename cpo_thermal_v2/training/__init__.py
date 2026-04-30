"""
cpo_thermal_v2.training — Stage D PPO training infrastructure
=============================================================

Submodules:
    config_loader   — YAML inherit + CLI override
    gae             — dual-channel GAE
    rollout_buffer  — on-policy buffer + per-channel reward normaliser
    curriculum      — 3-stage cold/warm/hot scheduler
    env_factory     — AsyncVectorEnv builder + curriculum broadcast helper
    ppo_trainer     — PPO update loop (dual-channel clip + value losses)
    train           — main entrypoint, CLI

Most things are accessible via:

    >>> from cpo_thermal_v2.training import (
    ...     load_config, merge_cli_overrides,
    ...     CurriculumScheduler,
    ...     RolloutBuffer, RewardNormaliser,
    ...     compute_gae, normalise_advantages,
    ...     PPOTrainer, PPOUpdateMetrics,
    ...     make_optimizer, make_lr_scheduler,
    ...     make_vector_env, broadcast_curriculum_stage,
    ... )

The :mod:`config_loader` and :mod:`curriculum` submodules are torch-free
and importable on the simulation-only sandbox.  The rest require torch
(``rollout_buffer``, ``gae``) or torch + gymnasium + torch_geometric
(``env_factory``, ``ppo_trainer``, ``train``) and are soft-imported.
"""
# Always-available (numpy + PyYAML only)
from .config_loader import (
    load_config, merge_cli_overrides, save_resolved_config,
    DEFAULT_CONFIG_PATH,
)
from .curriculum    import CurriculumScheduler, CurriculumStage

__all__ = [
    "load_config", "merge_cli_overrides", "save_resolved_config",
    "DEFAULT_CONFIG_PATH",
    "CurriculumScheduler", "CurriculumStage",
]

# torch-required modules (soft-import)
try:
    from .gae            import compute_gae, normalise_advantages
    from .rollout_buffer import RolloutBuffer, RewardNormaliser
    __all__ += [
        "compute_gae", "normalise_advantages",
        "RolloutBuffer", "RewardNormaliser",
    ]
except ImportError:
    pass

# torch + gymnasium + PyG (soft-import)
try:
    from .env_factory import (
        make_vector_env, broadcast_curriculum_stage, smoke_test_vector_env,
        make_single_env,
    )
    from .ppo_trainer import (
        PPOTrainer, PPOUpdateMetrics, make_optimizer, make_lr_scheduler,
    )
    __all__ += [
        "make_vector_env", "broadcast_curriculum_stage",
        "smoke_test_vector_env", "make_single_env",
        "PPOTrainer", "PPOUpdateMetrics",
        "make_optimizer", "make_lr_scheduler",
    ]
except ImportError:
    pass
