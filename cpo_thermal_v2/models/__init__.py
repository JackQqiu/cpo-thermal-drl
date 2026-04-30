"""
cpo_thermal_v2.models — Stage C neural network modules
======================================================

Requires ``torch`` and ``torch_geometric`` at import time.  On a sandbox
without these, importing this subpackage fails — but the parent package
``cpo_thermal_v2`` soft-imports it, so Stage A+B usage still works.

Public re-exports
-----------------
    >>> from cpo_thermal_v2.models import (
    ...     PPOActorCritic, HeteroEncoder, CrossAttentionActor, DualCritic,
    ...     graph_obs_to_hetero_data, build_batch,
    ... )
"""
from .hetero_encoder        import HeteroEncoder, graph_obs_to_hetero_data
from .cross_attention_actor import CrossAttentionActor
from .value_critic          import DualCritic
from .ppo_actor_critic      import PPOActorCritic, build_batch

__all__ = [
    "HeteroEncoder", "graph_obs_to_hetero_data",
    "CrossAttentionActor", "DualCritic",
    "PPOActorCritic", "build_batch",
]
