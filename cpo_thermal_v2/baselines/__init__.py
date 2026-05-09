"""
cpo_thermal_v2.baselines — Evaluation baselines + trained-model wrapper
========================================================================

All schedulers implement :class:`BaseScheduler` for uniform driving
in :mod:`evaluation.runner`.

Public exports
--------------
    BaseScheduler         — abstract interface
    RoundRobinScheduler   — sanity-check baseline
    HEFTScheduler         — classic Heterogeneous Earliest Finish Time
    ThermalHEFTScheduler  — PROCHOT-aware HEFT (heat penalty in cost)
    DecimaFairScheduler   — fair Decima (separately trained, never
                            sees thermal features)
    TrainedPPOScheduler   — wraps a saved PPOActorCritic checkpoint
"""
from .base                 import BaseScheduler
from .round_robin          import RoundRobinScheduler
from .heft                 import HEFTScheduler
from .thermal_heft         import ThermalHEFTScheduler
from .throttled_heft       import ThrottledHEFTScheduler

__all__ = [
    "BaseScheduler",
    "RoundRobinScheduler",
    "HEFTScheduler",
    "ThermalHEFTScheduler",
    "ThrottledHEFTScheduler"
]

# torch-required schedulers (soft import)
try:
    from .decima_fair  import DecimaFairScheduler
    from .trained_ppo  import TrainedPPOScheduler
    __all__ += ["DecimaFairScheduler", "TrainedPPOScheduler"]
except ImportError:
    pass
