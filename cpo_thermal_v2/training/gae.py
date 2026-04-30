"""
training/gae.py — Generalised Advantage Estimation (dual channel)
=================================================================

Computes per-channel advantages and discounted returns for the v2
dual-critic PPO loop.

Why dual-channel
----------------
The env emits a per-step reward decomposition::

    info['reward_channels'] = {'placement': r_p, 'delay': r_d, 'total': r_p+r_d}

We run **two independent GAE passes**:

    A_placement, R_placement = gae(rewards=r_p, values=V_p, ...)
    A_delay,     R_delay     = gae(rewards=r_d, values=V_d, ...)

so that:
* the placement actor head is updated with ``A_placement`` only,
* the delay     actor head is updated with ``A_delay``    only,
* the encoder gets the **sum** of both losses (shared trunk).

In ``auto_only`` mode the trainer simply zeroes the delay-channel loss
coefficient — GAE itself is mode-agnostic.

Reference
---------
Schulman et al., "High-Dimensional Continuous Control Using
Generalized Advantage Estimation" (2016), eq. (16-18).

We implement the standard GAE-λ recurrence::

    δ_t = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
    A_t = δ_t + γλ·A_{t+1}·(1 - done_t)

with the convention that ``done_t = True`` zeroes the bootstrap from
``V(s_{t+1})`` AND the advantage propagation across the boundary —
i.e., terminal states act as absorbing reward sinks.  The
``truncated`` signal from gymnasium is treated identically to ``done``
for masking purposes (we still bootstrap from the saved tail value
``last_value`` in the truncated case; see ``compute_gae`` below).

API
---
``compute_gae(rewards, values, dones, last_values, gamma, lam)``
    Vectorised over a (T, N) buffer where T = rollout_length,
    N = num_envs.  Returns ``(advantages, returns)`` of shape (T, N).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def compute_gae(
    rewards:     torch.Tensor,    # (T, N) per-step reward (one channel)
    values:      torch.Tensor,    # (T, N) V(s_t) at each step
    dones:       torch.Tensor,    # (T, N) bool/float; 1 if episode ended at t
    last_values: torch.Tensor,    # (N,)   V(s_T) — the tail bootstrap value
    gamma:       float = 0.99,
    lam:         float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-λ advantages and discounted returns for ONE channel.

    Parameters
    ----------
    rewards
        Per-step rewards, shape ``(T, N)``.
    values
        Critic predictions ``V(s_t)``, shape ``(T, N)``.  These are the
        critic outputs at the **states** the agent acted on, NOT
        bootstrapped from ``s_{t+1}``.
    dones
        Episode-termination flags, shape ``(T, N)``.  ``True`` (or
        non-zero) means the episode ended at this step's *next* state —
        i.e., ``s_{t+1}`` is absorbing.
    last_values
        Tail-bootstrap critic values ``V(s_T)``, shape ``(N,)``.  Used
        only for the last GAE step (the recurrence requires
        ``V(s_{t+1})`` for the final ``t = T-1``).
    gamma
        Discount factor.  Default 0.99 — matches the v1 config.
    lam
        GAE λ.  Default 0.95 — matches the v1 config.

    Returns
    -------
    advantages : (T, N) — A_t for each (t, env) pair
    returns    : (T, N) — A_t + V_t (the targets for the value head)

    Notes
    -----
    * Dtype/device matching: all inputs must already be on the same
      device and dtype as ``values``.  We do NOT cast here, both because
      torch's broadcasting is finicky and because the trainer should
      have explicit control over precision (e.g. half-precision training).
    * No in-place mutation of inputs.
    """
    assert rewards.shape == values.shape == dones.shape, (
        f"shape mismatch: rewards={tuple(rewards.shape)}, "
        f"values={tuple(values.shape)}, dones={tuple(dones.shape)}"
    )
    T, N = rewards.shape
    assert last_values.shape == (N,), (
        f"last_values must be (N={N},), got {tuple(last_values.shape)}"
    )

    advantages = torch.zeros_like(rewards)
    # Roll forward: A_T (at t = T-1) bootstraps from last_values,
    # then A_{T-2}, ... back to A_0 via the recurrence.
    gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    # Convert dones to float once (saves repeated casts in the loop)
    not_done = (1.0 - dones.float())

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_values
        else:
            next_value = values[t + 1]
        # δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)
        delta = rewards[t] + gamma * next_value * not_done[t] - values[t]
        # A_t = δ_t + γλ A_{t+1} (1 - done_t)
        gae = delta + gamma * lam * not_done[t] * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def normalise_advantages(
    advantages: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-batch advantage normalisation (mean 0, std 1).

    Standard PPO trick.  The trainer can disable this via config —
    in that case it simply doesn't call this function.

    Operates on the FLATTENED (T*N,) view; returns the same shape.
    """
    flat = advantages.reshape(-1)
    mean = flat.mean()
    std  = flat.std(unbiased=False)
    return (advantages - mean) / (std + eps)


# =====================================================================
# Self-test
# =====================================================================
def _self_test():
    """Sanity-check GAE on hand-computable trivial cases."""
    print("Running gae.py self-tests...\n")

    # Case 1: gamma=0, lam=0  ⇒  A_t = r_t - V_t,  R_t = r_t
    T, N = 4, 2
    rewards = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    values  = torch.tensor([[0.5, 0.5]] * 4)
    dones   = torch.zeros(T, N)
    last_v  = torch.zeros(N)
    A, R = compute_gae(rewards, values, dones, last_v, gamma=0.0, lam=0.0)
    expected_A = rewards - values
    expected_R = rewards
    assert torch.allclose(A, expected_A), f"γ=λ=0 advantage wrong: {A} vs {expected_A}"
    assert torch.allclose(R, expected_R), f"γ=λ=0 return wrong: {R} vs {expected_R}"
    print(f"  ✅ γ=λ=0 case: A_t = r_t - V_t, R_t = r_t")

    # Case 2: gamma=1, lam=1, all-zero V, no dones
    #   ⇒  A_t = sum_{k>=t} r_k  (Monte Carlo return up to bootstrap)
    rewards = torch.tensor([[1.], [2.], [3.], [4.]])    # (T=4, N=1)
    values  = torch.zeros(T, 1)
    dones   = torch.zeros(T, 1)
    last_v  = torch.zeros(1)
    A, R = compute_gae(rewards, values, dones, last_v, gamma=1.0, lam=1.0)
    expected = torch.tensor([[10.], [9.], [7.], [4.]])  # 1+2+3+4, 2+3+4, ...
    assert torch.allclose(A, expected), f"MC case wrong: {A.flatten()} vs {expected.flatten()}"
    print(f"  ✅ γ=λ=1, V=0 case: A_t = Σ r_k (Monte Carlo)")

    # Case 3: done in middle truncates the future
    rewards = torch.tensor([[1.], [2.], [3.], [4.]])
    values  = torch.zeros(T, 1)
    dones   = torch.tensor([[0.], [1.], [0.], [0.]])    # done at t=1
    last_v  = torch.zeros(1)
    A, R = compute_gae(rewards, values, dones, last_v, gamma=1.0, lam=1.0)
    # t=3: A=4
    # t=2: A=3+4=7
    # t=1: done -> A=2 (no bootstrap from t=2)
    # t=0: A=1+2=3
    expected = torch.tensor([[3.], [2.], [7.], [4.]])
    assert torch.allclose(A, expected), f"done case wrong: {A.flatten()} vs {expected.flatten()}"
    print(f"  ✅ done at t=1 truncates: A = {A.flatten().tolist()}")

    # Case 4: last_value bootstrap
    rewards = torch.zeros(T, 1)
    values  = torch.zeros(T, 1)
    dones   = torch.zeros(T, 1)
    last_v  = torch.tensor([10.])
    A, R = compute_gae(rewards, values, dones, last_v, gamma=0.9, lam=1.0)
    # δ_t at t=3: 0 + 0.9 * 10 - 0 = 9; A_3 = 9
    # δ_t at t=2: 0 + 0.9 * 0  - 0 = 0; A_2 = 0 + 0.9*1.0*9 = 8.1
    # δ_t at t=1: 0; A_1 = 0 + 0.9*8.1 = 7.29
    # δ_t at t=0: 0; A_0 = 0 + 0.9*7.29 = 6.561
    expected = torch.tensor([[6.561], [7.29], [8.1], [9.0]])
    assert torch.allclose(A, expected, atol=1e-5), \
        f"bootstrap case wrong: {A.flatten()} vs {expected.flatten()}"
    print(f"  ✅ last_value bootstrap propagates")

    # Case 5: shape assertions
    try:
        compute_gae(rewards, values, dones, torch.zeros(2), 0.99, 0.95)
        assert False, "should have raised AssertionError"
    except AssertionError:
        print("  ✅ shape mismatch on last_values raises")

    # Case 6: normalise_advantages
    a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    n = normalise_advantages(a)
    assert abs(n.mean().item()) < 1e-6
    assert abs(n.std(unbiased=False).item() - 1.0) < 1e-5
    print(f"  ✅ normalise_advantages: mean={n.mean().item():.2e}, "
          f"std={n.std(unbiased=False).item():.4f}")

    print("\nAll gae.py tests passed ✓")


if __name__ == "__main__":
    _self_test()
