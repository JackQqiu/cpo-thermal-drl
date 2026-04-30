"""
cpo_thermal_v2.envs — environment, RC dynamics, reward, DAG parser
==================================================================

This subpackage holds the simulation-side code (no torch dependency).

Public re-exports
-----------------
    >>> from cpo_thermal_v2.envs import (
    ...     CPOThermalDAGEnvV2, CPOThermalDAGEnv,
    ...     RCThermalDynamics, MicroserviceDAG,
    ...     compute_reward, compute_reward_channels, RewardConfig,
    ... )
"""
from .rc_dynamics    import RCThermalDynamics
from .dag_parser     import MicroserviceDAG
from .reward_shaping import (
    compute_reward, compute_reward_channels, decompose_reward, RewardConfig,
)
from .cpo_thermal_env import CPOThermalDAGEnvV2, CPOThermalDAGEnv

__all__ = [
    "CPOThermalDAGEnvV2", "CPOThermalDAGEnv",
    "RCThermalDynamics", "MicroserviceDAG",
    "compute_reward", "compute_reward_channels",
    "decompose_reward", "RewardConfig",
]
