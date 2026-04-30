"""
baselines/trained_ppo.py — Wraps a trained PPO checkpoint as a scheduler
=========================================================================

The same model file (``ppo_actor_critic.PPOActorCritic``) is used during
training; here we wrap a loaded checkpoint in :class:`BaseScheduler` so
:mod:`evaluation.runner` can drive it through the same loop as HEFT,
Round-Robin, etc.

The same checkpoint is reused for the THREE evaluation conditions:

    Ours-auto_only:    action_mode='auto_only', delay always 0
    Ours-agent_only:   action_mode='agent_only', env-cool disabled (zero-shot)
    Ours-hybrid:       action_mode='hybrid', env-cool enabled

The ``Ours-agent_only`` evaluation is a *zero-shot generalization* test
— we never train with that mode, but we evaluate on it to demonstrate
that the agent has actually learned to anticipate (when env-cool is
removed, agent's own delay decisions must keep things stable).

Determinism
-----------
For reporting, evaluation always uses ``deterministic=True`` (argmax
instead of sampling).  This gives lower-variance numbers than
stochastic evaluation and matches what HEFT/Thermal-HEFT do (they're
deterministic).  The paper should note this in Section 6.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import Action, BaseScheduler


def _lazy_torch():
    import torch
    return torch


class TrainedPPOScheduler(BaseScheduler):
    """Loads a saved PPOActorCritic and runs it as a scheduler.

    Parameters
    ----------
    ckpt_path
        Path to ``best.pt`` or ``final.pt`` saved by training.
    action_mode
        Determines how the action is unpacked.  IMPORTANT: must match
        the env's action_mode at evaluation time.  See module docstring
        for the auto_only / agent_only / hybrid story.
    K_delay
        Must match training.  Defaults to 5.
    deterministic
        If True (default), use argmax over logits.  If False, sample
        from Categorical for stochastic evaluation.
    device
        Where to run the model.  ``'cpu'`` is fine for evaluation
        (low throughput requirements); use ``'cuda:0'`` to amortise
        CPU↔GPU transfer if running many episodes.
    """

    name = "TrainedPPO"

    def __init__(
        self,
        ckpt_path:     str,
        action_mode:   str = "hybrid",
        K_delay:       int = 5,
        deterministic: bool = True,
        device:        str  = "cpu",
        scheduler_label: Optional[str] = None,
    ):
        super().__init__(action_mode=action_mode, K_delay=K_delay)
        self.deterministic = bool(deterministic)
        self.device        = device
        if scheduler_label is not None:
            self.name = scheduler_label    # e.g. "Ours-hybrid", "Ours-agent_only"

        torch = _lazy_torch()
        from cpo_thermal_v2.models import PPOActorCritic, build_batch
        self._build_batch = build_batch

        # Construct a model in the requested action_mode.  We DELIBERATELY
        # do this with strict=False because the saved checkpoint was likely
        # trained with a different action_mode (e.g. ``hybrid`` ckpt loaded
        # for ``agent_only`` eval, or vice versa).  All shared layers
        # transfer; the action-mode-specific bits don't matter at eval
        # because we read the right output keys per mode.
        self._model = PPOActorCritic(
            action_mode = action_mode,
            K_delay     = K_delay,
        ).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        pass     # stateless

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        torch = _lazy_torch()
        graph_obs = info["graph_obs"][0] if isinstance(info["graph_obs"],
                                                       (list, np.ndarray)) \
                else info["graph_obs"]
        action_mask = np.asarray(info["action_mask"], dtype=bool)

        with torch.no_grad():
            batch = self._build_batch([graph_obs], [action_mask],
                                       device=self.device)
            out = self._model.act(batch, deterministic=self.deterministic)

        a = out["action"][0]
        if self.action_mode == "auto_only":
            return int(a)
        # factored modes
        if hasattr(a, "cpu"):
            a = a.cpu().numpy()
        return np.asarray(a, dtype=np.int64)
