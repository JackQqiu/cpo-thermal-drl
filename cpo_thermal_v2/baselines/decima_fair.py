"""
baselines/decima_fair.py — Ours-NoThermal (paper-facing) / thermal-blind-input baseline
======================================================================================

Paper-facing name: **Ours-NoThermal**
Eval CSV scheduler label: **Decima** (renamed to Ours-NoThermal by
``cpo_thermal_v2/scripts/compose_paper_section5.py`` aggregator).
Disk checkpoint: ``checkpoints_v1/decima_fair_N17/best.pt``.

What this baseline isolates
---------------------------
The Ours architecture (heterogeneous-GAT encoder + RC-coupling edge
attribute + cross-attention placement actor) trained WITHOUT
per-processor thermal observation features and WITHOUT the
``est_temp_rise`` task-to-processor edge attribute. The thermal-aware
reward shaping (Section §3 of the paper, ``reward.thermal_aware``) is
RETAINED, identical to Ours-auto_only.

This baseline answers the paper §5 Component 5 question: "Does the
proposed architecture still deliver near-optimal safety when the
policy cannot observe per-processor temperatures, as long as the
training reward remains thermal-aware?" Empirical answer: yes
(viol_rate = 0.006 vs Ours-auto_only's 0.002; McNemar p = 0.625 ns).

Training-time configuration (ground truth: train_decima_fair.yaml)
------------------------------------------------------------------
  - proc_in_dim     = 3   (drops T_norm, dT/dt, leakage, headroom from proc node features)
  - edge_dim_t2p    = 1   (drops est_temp_rise from task→proc edges)
  - reward          = thermal-aware (unchanged from Ours; the fairness
                      constraint is on INPUT FEATURES only, not on the
                      reward signal)
  - env wrapper     = ThermalBlindWrapper at runtime (strips proc
                      thermal cols + edge thermal cols from
                      info['graph_obs'] at every reset/step)
  - total_steps     = 2_000_000  (4 h on 1×V100 + 16 envs)

Architectural parity with Ours-auto_only:
  - Same HeteroEncoder
  - Same cross-attention placement actor
  - Same value critic
The ONLY differences at the model level are ``proc_in_dim`` (3 vs 7)
and ``edge_dim_t2p`` (1 vs 2).

Naming history
--------------
The file name ``decima_fair`` predates the paper-facing name. The
``decima`` prefix comes from this baseline being the fair counterpart
to ``baselines/decima.py`` (which reuses our Stage-2 checkpoint with
inputs masked, leaking thermal-aware weights). Paper-facing prose
uses ``Ours-NoThermal``; CSV column reads ``Decima``; this module
keeps the historical name to avoid 5-file rename churn.

Usage in evaluation::

    from cpo_thermal_v2.baselines.decima_fair import DecimaFairScheduler
    decima = DecimaFairScheduler(
        ckpt_path="checkpoints_v1/decima_fair_N17/best.pt",
        num_nodes=17,
    )
    action = decima.schedule(obs, info)
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Action, BaseScheduler


def _lazy_torch():
    import torch
    return torch


class DecimaFairScheduler(BaseScheduler):
    """Decima trained from scratch without thermal features.

    Parameters
    ----------
    ckpt_path : str
        Path to a checkpoint trained by ``scripts/train_decima_fair.py``.
        The model inside MUST have ``proc_in_dim=3``; this is enforced
        at load time.
    num_nodes : int
        Number of processors (must match training-time N).
    deterministic : bool
        If True, use argmax over policy logits at inference.
    device : str
        torch device for inference.
    """

    name = "Decima"   # paper-facing name; appears in plots/tables

    def __init__(
        self,
        ckpt_path:    str,
        num_nodes:    int = 17,
        deterministic: bool = True,
        device:       str  = "cpu",
    ):
        # Decima is auto_only by design: it picks a processor per task,
        # no separate delay action.  Override action_mode regardless of
        # what the env runs in.
        super().__init__(action_mode="auto_only", K_delay=5)
        self.num_nodes     = int(num_nodes)
        self.deterministic = bool(deterministic)
        self.device        = device

        torch = _lazy_torch()
        from cpo_thermal_v2.models import PPOActorCritic, build_batch
        self._build_batch = build_batch

        # IMPORTANT: proc_in_dim=3 + edge_dim_t2p=1 (no thermal anywhere).
        # The encoder will only see [busy(0), remaining(0), is_ASIC] for
        # procs, and [est_exec_time/50] for task→proc edges.
        self._model = PPOActorCritic(
            action_mode  = "auto_only",
            K_delay      = 5,
            proc_in_dim  = 3,    # blind: 7 - 4 thermal cols
            edge_dim_t2p = 1,    # blind: 2 - 1 (drop est_temp_rise)
        ).to(device)

        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self._model.load_state_dict(state, strict=True)
        self._model.eval()

    def reset(self, obs: np.ndarray, info: Dict[str, Any]) -> None:
        pass    # stateless

    def _strip_thermal_features(self, graph_obs: dict) -> dict:
        """Return a copy of graph_obs with ALL thermal info removed.

        Three places thermal info appears in graph_obs:
          1. ``proc_x[:, 0:4]`` — T_norm, dT/dt, leakage, headroom
          2. ``edges_t2p_attr[:, 1]`` — est_temp_rise per (task,proc) edge
          3. None elsewhere (task_x is purely DAG-structural)

        Fair Decima must see NONE of this.  We slice both:
          - proc_x: 7 cols → 3 cols (drop thermal, keep busy/remaining/is_ASIC)
          - edges_t2p_attr: 2 cols → 1 col (drop est_temp_rise, keep est_exec_time)
        """
        new_obs = dict(graph_obs)

        # 1. Strip thermal cols from proc_x
        proc_x = np.asarray(graph_obs.get("proc_x", []), dtype=np.float32)
        if proc_x.size > 0 and proc_x.ndim == 2:
            if proc_x.shape[1] == 7:
                new_obs["proc_x"] = proc_x[:, 4:7].copy()
            elif proc_x.shape[1] == 3:
                pass    # already stripped
            else:
                raise ValueError(
                    f"Expected proc_x with 7 or 3 cols, got {proc_x.shape}"
                )

        # 2. Strip est_temp_rise from edges_t2p_attr
        # NOTE: env's key is "edges_t2p_attr", not "task_proc_edge_attr".
        edge_attr = np.asarray(graph_obs.get("edges_t2p_attr", []),
                               dtype=np.float32)
        if edge_attr.size > 0 and edge_attr.ndim == 2:
            if edge_attr.shape[1] == 2:
                new_obs["edges_t2p_attr"] = edge_attr[:, 0:1].copy()
            elif edge_attr.shape[1] == 1:
                pass    # already stripped

        return new_obs

    def schedule(self, obs: np.ndarray, info: Dict[str, Any]) -> Action:
        torch = _lazy_torch()
        graph_obs = info["graph_obs"][0] if isinstance(info["graph_obs"],
                                                       (list, np.ndarray)) \
                else info["graph_obs"]
        blind_graph = self._strip_thermal_features(graph_obs)
        action_mask = np.asarray(info["action_mask"], dtype=bool)

        with torch.no_grad():
            batch = self._build_batch([blind_graph], [action_mask],
                                       device=self.device,
                                       thermal_blind=True)
            out = self._model.act(batch, deterministic=self.deterministic)

        a = out["action"][0]
        return int(a)
