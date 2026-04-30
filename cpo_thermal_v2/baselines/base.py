"""
baselines/base.py — Abstract scheduler interface for evaluation
================================================================

All baselines (HEFT, Round-Robin, Thermal-HEFT, Decima) AND our trained
PPO model implement this same interface so :mod:`evaluation.runner` can
treat them uniformly.

The interface is deliberately minimal — schedulers see one env at a
time (no batching), and :meth:`schedule` returns one action.  This is
much simpler than the AsyncVectorEnv setup used during training, but
it's sufficient for evaluation where throughput isn't critical.

Subclass contract
-----------------
::

    class MyScheduler(BaseScheduler):
        def reset(self, obs, info): ...      # Called once at episode start
        def schedule(self, obs, info) -> action: ...   # Called per step

The action shape must match the env's ``action_mode``:
    auto_only       → int (proc index)
    agent_only/hybrid → np.ndarray of shape (2,) [proc_idx, delay_idx]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np


Action = Union[int, np.ndarray]


class BaseScheduler(ABC):
    """Abstract base for all evaluation schedulers."""

    name: str = "BASE"   # subclass should override

    def __init__(self, action_mode: str = "auto_only", K_delay: int = 5):
        if action_mode not in ("auto_only", "agent_only", "hybrid"):
            raise ValueError(f"unknown action_mode: {action_mode!r}")
        self.action_mode = action_mode
        self.K_delay     = K_delay

    @abstractmethod
    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        """Called once at the start of each episode.  Use to clear caches,
        reset internal state, etc."""

    @abstractmethod
    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        """Return the action for this step."""

    # -----------------------------------------------------------------
    def _wrap_action(self, proc_idx: int, delay_idx: int = 0) -> Action:
        """Wrap a (proc_idx, delay_idx) decision in the right action shape.

        Concrete schedulers should call this rather than constructing
        the action manually — it handles the auto_only / factored-mode
        distinction transparently.
        """
        if self.action_mode == "auto_only":
            return int(proc_idx)
        return np.array([int(proc_idx), int(delay_idx)], dtype=np.int64)
