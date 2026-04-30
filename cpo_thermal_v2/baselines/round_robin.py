"""
baselines/round_robin.py — Sanity-check baseline
=================================================

Cycles through processors in order, respecting only the action mask.
Provides a lower-bound reference: any non-trivial scheduler should beat
this.  Useful for catching env bugs (if Round-Robin's stats look weird,
something's wrong with the simulator, not the policy).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Action, BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    """Cycles proc 0 → 1 → ... → N-1 → 0 → ...

    Skips masked-out (over-temp) procs.  Always picks ``delay_idx=0``
    (no agent delay) — matches the simplest possible policy.
    """

    name = "RoundRobin"

    def __init__(self, action_mode: str = "auto_only", K_delay: int = 5):
        super().__init__(action_mode=action_mode, K_delay=K_delay)
        self._next_idx = 0

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        self._next_idx = 0

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        mask = info["action_mask"]
        n = len(mask)
        # Find next valid proc starting from self._next_idx
        for offset in range(n):
            i = (self._next_idx + offset) % n
            if mask[i]:
                self._next_idx = (i + 1) % n
                return self._wrap_action(i, delay_idx=0)
        # Should never happen — env always keeps at least one valid proc
        return self._wrap_action(0, delay_idx=0)
