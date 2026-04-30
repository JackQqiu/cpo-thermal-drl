"""
value_critic.py
===============

Stage C — Dual-channel value head for the v2 PPO.

The critic predicts **two separate values** per state:

    V_placement(s)   ← predicts return on the placement-channel reward
    V_delay(s)       ← predicts return on the delay-channel reward

Why dual critics
----------------
PPO's advantage is ``A = R - V``.  With a single scalar critic, the
delay head would be updated using the ``A`` derived from ``placement +
delay`` rewards together, contaminating the credit-assignment signal.
Two critics let each head update on its own clean advantage:

    A_placement = R_placement - V_placement   → only placement head
    A_delay     = R_delay     - V_delay       → only delay head
    encoder gets gradient from the SUM of both losses

This is the implementation of the user's choice "完整实现 placement/delay
双 advantage" for hybrid mode.

For ``auto_only``, the trainer simply ignores the delay channel
(``A_delay`` is computed but no delay-head loss term enters backprop).

Input
-----
The critic takes the encoder output (``x_task``, ``x_proc``,
``batch_task``, ``batch_proc``) and produces two scalars per graph.

Pooling strategy: **(mean(task) ⊕ mean(proc) ⊕ max(proc))** — the proc
max-pool is what surfaces hot-spot signals (a single proc near T_pen
matters more than the average).  This was added vs the single-mean
v1 critic precisely to give the value head the same hot-spot view the
actor's delay head has.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_max_pool


class DualCritic(nn.Module):
    """Two-head value network sharing a pooled trunk."""

    def __init__(
        self,
        hidden:      int = 128,
        trunk_hidden: int = 256,
    ):
        super().__init__()
        # Pooled feature dim:
        #   mean(task) (h) + mean(proc) (h) + max(proc) (h) = 3h
        in_dim = hidden * 3
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden), nn.GELU(),
            nn.LayerNorm(trunk_hidden),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
            nn.LayerNorm(trunk_hidden),
        )
        self.value_placement = nn.Linear(trunk_hidden, 1)
        self.value_delay     = nn.Linear(trunk_hidden, 1)

    def forward(
        self,
        x_task:     torch.Tensor,    # (sum_tasks, h)
        x_proc:     torch.Tensor,    # (sum_proc, h)
        batch_task: torch.Tensor,    # (sum_tasks,)
        batch_proc: torch.Tensor,    # (sum_proc,)
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(V_placement, V_delay)`` each of shape ``(batch_size,)``."""
        # Robust pooling.  Edge case: a graph with zero tasks (shouldn't
        # happen with valid env output, but defensive: replace by zeros).
        if x_task.numel() > 0:
            mean_task = global_mean_pool(x_task, batch_task, size=batch_size)
        else:
            mean_task = x_task.new_zeros(batch_size, x_task.size(-1) if x_task.dim() == 2 else 1)

        mean_proc = global_mean_pool(x_proc, batch_proc, size=batch_size)
        max_proc  = global_max_pool(x_proc, batch_proc, size=batch_size)

        pooled = torch.cat([mean_task, mean_proc, max_proc], dim=-1)
        h = self.trunk(pooled)

        v_p = self.value_placement(h).squeeze(-1)   # (B,)
        v_d = self.value_delay(h).squeeze(-1)       # (B,)
        return v_p, v_d
