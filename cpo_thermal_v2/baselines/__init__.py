"""
cpo_thermal_v2.baselines — Evaluation baselines + trained-model wrapper
========================================================================

All schedulers implement :class:`BaseScheduler` for uniform driving
in :mod:`evaluation.runner`.

Public exports
--------------
    BaseScheduler       — abstract interface
    RoundRobinScheduler — sanity-check baseline
    HEFTScheduler       — classic Heterogeneous Earliest Finish Time
    ThermalHEFTScheduler— PROCHOT-aware HEFT (heat penalty in cost)
    DecimaScheduler     — Decima-style (no thermal awareness, uses
                          our trained ckpt with thermal features masked)
    TrainedPPOScheduler — wraps a saved PPOActorCritic checkpoint
"""
from .base                 import BaseScheduler
from .round_robin          import RoundRobinScheduler
from .heft                 import HEFTScheduler
from .thermal_heft         import ThermalHEFTScheduler

__all__ = [
    "BaseScheduler",
    "RoundRobinScheduler",
    "HEFTScheduler",
    "ThermalHEFTScheduler",
]

# torch-required schedulers (soft import)
try:
    from .decima       import DecimaScheduler
    from .trained_ppo  import TrainedPPOScheduler
    __all__ += ["DecimaScheduler", "TrainedPPOScheduler"]
except ImportError:
    pass
