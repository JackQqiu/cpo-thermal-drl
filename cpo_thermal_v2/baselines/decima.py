"""
baselines/decima.py — Decima-style RL baseline (no thermal awareness)
======================================================================

Decima (Mao et al., SIGCOMM 2019) is the reference RL scheduler for
DAG workloads.  It uses a GNN encoder + cross-attention placement head,
trained with PPO on makespan-only reward.  The original architecture
doesn't include thermal information.

For our paper, we report two flavours of "Decima":

1. **Decima-noThermal** (this file, default): mirrors the original
   paper — only DAG features and proc *availability* (no temperature),
   trained on a makespan-only reward.  This isolates the contribution
   of *thermal awareness* in our model.

2. **Decima-Hardcoded**: a stand-in that uses the trained model's GNN
   weights but ignores the thermal channel of the reward.  Implemented
   in :mod:`evaluation.runner` rather than here (it's a runtime
   reweighting, not a separate model class).

Implementation strategy
-----------------------
We DON'T re-train Decima from scratch (that's an entire separate
training run, ~25h on V100, would push our walltime over).  Instead:

* For evaluation, we use a "Decima-as-policy" wrapper that loads our
  ``stage1_auto_only`` checkpoint (which IS architecturally Decima:
  GNN + placement-only attention head, no agent delay) but runs it
  with the simplification ``proc_x[:, 0:4] *= 0`` — zeroing out the
  thermal channels of the proc features at inference time.
* This gives the trained GNN topology+DAG awareness but blinds it to
  temperature, mimicking what Decima would see.

This isn't a perfect Decima reimplementation — a real comparison
would need a separately-trained no-thermal model.  We document this
limitation in the paper (Section 6.1, "Baseline Implementation
Caveats").  In practice:
  * If Ours-hybrid > Decima-noThermal → thermal awareness helps.
  * If Decima-noThermal > Ours-hybrid → trouble (thermal awareness
    is hurting more than helping).

For a from-scratch Decima run (paper appendix), use a separate
training config with ``model.thermal_features_enabled=False`` once
that flag is added.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import Action, BaseScheduler


# Lazy torch import
def _lazy_torch():
    import torch
    return torch


class DecimaScheduler(BaseScheduler):
    """Trained-PPO wrapper that masks out thermal proc features.

    Loads a Stage-1 checkpoint (architecturally Decima-shaped) and at
    inference time replaces ``proc_x[:, 0:4]`` (the thermal-related
    features: T_norm, dT/dt, leakage, headroom) with zeros.  The
    encoder thus sees only ``busy / remaining / is_ASIC`` and DAG
    structure, mimicking what Decima sees.

    For full parity with the published Decima, a paper appendix should
    include a separately trained no-thermal model.  This stand-in is
    the practical compromise within walltime budget.
    """

    name = "Decima"

    def __init__(
        self,
        ckpt_path:    str,
        action_mode:  str = "auto_only",
        K_delay:      int = 5,
        num_nodes:    int = 17,
        deterministic: bool = True,
        device:       str  = "cpu",
    ):
        super().__init__(action_mode=action_mode, K_delay=K_delay)
        self.num_nodes     = num_nodes
        self.deterministic = bool(deterministic)
        self.device        = device

        torch = _lazy_torch()
        from cpo_thermal_v2.models import PPOActorCritic, build_batch
        self._build_batch = build_batch

        # Construct a fresh model and load the checkpoint
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

    def _strip_thermal_features(self, graph_obs: dict) -> dict:
        """Return a copy of graph_obs with proc thermal features zeroed."""
        proc_x = np.asarray(graph_obs.get("proc_x", []), dtype=np.float32)
        if proc_x.size > 0 and proc_x.ndim == 2:
            proc_x_blind = proc_x.copy()
            # Indices 0..3 are: T_norm, dT/dt, leakage, headroom
            # (see env._proc_features).  Zeroing them means the GNN sees
            # the same "structural-only" features Decima sees.
            proc_x_blind[:, 0:4] = 0.0
            new_obs = dict(graph_obs)
            new_obs["proc_x"] = proc_x_blind
            return new_obs
        return graph_obs

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        torch = _lazy_torch()
        graph_obs = info["graph_obs"][0] if isinstance(info["graph_obs"],
                                                       (list, np.ndarray)) \
                else info["graph_obs"]
        # Strip thermal features so the GNN sees what Decima would see
        blind_graph = self._strip_thermal_features(graph_obs)
        action_mask = np.asarray(info["action_mask"], dtype=bool)

        with torch.no_grad():
            batch = self._build_batch([blind_graph], [action_mask],
                                        device=self.device)
            out = self._model.act(batch, deterministic=self.deterministic)

        a = out["action"][0]
        if self.action_mode == "auto_only":
            return int(a)
        # factored modes: shape (2,)
        if hasattr(a, "cpu"):
            a = a.cpu().numpy()
        return np.asarray(a, dtype=np.int64)
