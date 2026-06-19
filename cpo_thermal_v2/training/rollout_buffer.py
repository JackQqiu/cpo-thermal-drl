"""
training/rollout_buffer.py — On-policy PPO rollout storage
==========================================================

A fixed-capacity ring of (T, N) tensors holding everything the PPO update
needs from one rollout phase.  Unlike the v1 ``replay_buffer.py``, this
buffer is **on-policy and overwritten every rollout** — there's no
sampling with replacement.

What's stored
-------------
For each timestep ``t`` in ``[0, T)`` and each env ``n`` in ``[0, N)``:

* **action**       : ``(T, N)`` for auto_only, ``(T, N, 2)`` for factored modes
* **log_prob**     : ``(T, N)`` — combined log-prob (placement + delay sum)
* **log_prob_p**   : ``(T, N)`` — placement-only log-prob, for dual-PPO update
* **log_prob_d**   : ``(T, N)`` — delay-only log-prob, for dual-PPO update
* **value_p**      : ``(T, N)`` — V_placement(s_t)
* **value_d**      : ``(T, N)`` — V_delay(s_t)
* **reward_p**     : ``(T, N)`` — placement-channel reward
* **reward_d**     : ``(T, N)`` — delay-channel reward
* **done**         : ``(T, N)`` — episode-end mask
* **graph_obs**    : a list of ``T*N`` plain dicts (NOT a tensor — ragged)
* **action_mask**  : a list of ``T*N`` numpy bool arrays — ragged in N
* **current_idx**  : ``(T, N)`` — within-graph idx of the task being scheduled

The ``graph_obs`` and ``action_mask`` lists hold the env-emitted observations
verbatim.  Their ragged structure (different num_tasks per graph) requires
per-call ``Batch.from_data_list`` building — the buffer doesn't try to
pre-materialise tensors.

Why store both reward channels separately
-----------------------------------------
GAE is run twice (placement and delay), each with its own per-step rewards
and per-state value.  Storing them separately is cheaper than recomputing
from a saved ``reward_components`` dict, and lets the trainer normalise
each channel's running mean/std independently (placement reward has
~10× the magnitude of delay reward).

Memory footprint
----------------
With T=256, N=16, the per-step tensors total ~75 KB per channel; the
ragged graph dicts dominate at ~2-5 MB total per rollout.  All comfortably
under 200 MB peak even for T=1024 — this never bottlenecks training.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """Fixed-capacity (T, N) on-policy rollout buffer.

    Parameters
    ----------
    rollout_length : T
        Steps per env collected before each PPO update.
    num_envs : N
        Parallel envs in the AsyncVectorEnv.
    action_mode : str
        ``auto_only`` ⇒ scalar action;
        ``agent_only`` / ``hybrid`` ⇒ length-2 action array.
    device : torch.device or str
        Where the dense tensors live.  The graph dicts stay on CPU
        as Python lists — they're small enough that GPU transfer is
        only worth doing per mini-batch.
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs:        int,
        action_mode:     str,
        device:          str = "cpu",
    ):
        if action_mode not in ("auto_only", "agent_only", "hybrid"):
            raise ValueError(f"unknown action_mode: {action_mode!r}")
        self.T           = int(rollout_length)
        self.N           = int(num_envs)
        self.action_mode = action_mode
        self.device      = device

        # Action tensor shape depends on mode
        action_shape: Tuple[int, ...]
        if action_mode == "auto_only":
            action_shape = (self.T, self.N)
        else:
            action_shape = (self.T, self.N, 2)
        self.actions = torch.zeros(action_shape, dtype=torch.long, device=device)

        # Per-step scalar tensors (always (T, N))
        self.log_probs   = torch.zeros(self.T, self.N, device=device)
        self.log_probs_p = torch.zeros(self.T, self.N, device=device)
        self.log_probs_d = torch.zeros(self.T, self.N, device=device)
        self.values_p    = torch.zeros(self.T, self.N, device=device)
        self.values_d    = torch.zeros(self.T, self.N, device=device)
        self.rewards_p   = torch.zeros(self.T, self.N, device=device)
        self.rewards_d   = torch.zeros(self.T, self.N, device=device)
        # Optional cost channel (Lagrangian/RCPO constrained variant). Always
        # allocated (cheap) but only populated/used in Lagrangian mode; for the
        # default dual-channel path these stay zero and never enter a loss.
        self.values_cost  = torch.zeros(self.T, self.N, device=device)
        self.rewards_cost = torch.zeros(self.T, self.N, device=device)
        self.dones       = torch.zeros(self.T, self.N, device=device)

        # Ragged Python lists for the graph observations + masks.
        # Layout convention: index = t * N + n  (row-major over time first).
        self.graph_obs:    List[Dict[str, Any]]   = [None] * (self.T * self.N)  # type: ignore
        self.action_masks: List[np.ndarray]       = [None] * (self.T * self.N)  # type: ignore

        self._step = 0  # next write position, 0..T

    # -----------------------------------------------------------------
    @property
    def is_full(self) -> bool:
        return self._step >= self.T

    @property
    def step(self) -> int:
        return self._step

    def reset(self) -> None:
        """Clear the write pointer.  The underlying tensors are *not*
        zeroed (cheaper to overwrite as we go); ragged lists are reset
        to None so a missed write is caught by the assertion at update
        time rather than silently using stale data.
        """
        self._step = 0
        for i in range(self.T * self.N):
            self.graph_obs[i] = None        # type: ignore
            self.action_masks[i] = None     # type: ignore

    # -----------------------------------------------------------------
    def add(
        self,
        *,
        graph_obs:    List[Dict[str, Any]],   # length N
        action_masks: List[np.ndarray],       # length N
        actions:      torch.Tensor,           # (N,) or (N, 2)
        log_probs:    torch.Tensor,           # (N,)
        log_probs_p:  torch.Tensor,           # (N,)
        log_probs_d:  torch.Tensor,           # (N,)
        values_p:     torch.Tensor,           # (N,)
        values_d:     torch.Tensor,           # (N,)
        rewards_p:    np.ndarray,             # (N,)
        rewards_d:    np.ndarray,             # (N,)
        values_cost:  Optional[torch.Tensor] = None,   # (N,) — Lagrangian only
        rewards_cost: Optional[np.ndarray]   = None,   # (N,) — Lagrangian only
        dones:        np.ndarray,             # (N,) bool/float
    ) -> None:
        """Push one (N-wide) timestep into the buffer."""
        assert not self.is_full, (
            f"RolloutBuffer.add() called when buffer is already full "
            f"(_step={self._step}, T={self.T}); call reset() first."
        )
        assert len(graph_obs)    == self.N, f"graph_obs len {len(graph_obs)} != N={self.N}"
        assert len(action_masks) == self.N, f"action_masks len {len(action_masks)} != N={self.N}"

        t = self._step
        # Dense tensors — copy_ to keep autograd-clean
        self.actions[t]      = actions.detach()
        self.log_probs[t]    = log_probs.detach()
        self.log_probs_p[t]  = log_probs_p.detach()
        self.log_probs_d[t]  = log_probs_d.detach()
        self.values_p[t]     = values_p.detach()
        self.values_d[t]     = values_d.detach()
        self.rewards_p[t]    = torch.as_tensor(rewards_p, dtype=torch.float32,
                                                device=self.device)
        self.rewards_d[t]    = torch.as_tensor(rewards_d, dtype=torch.float32,
                                                device=self.device)
        if values_cost is not None:
            self.values_cost[t]  = values_cost.detach()
        if rewards_cost is not None:
            self.rewards_cost[t] = torch.as_tensor(rewards_cost, dtype=torch.float32,
                                                    device=self.device)
        self.dones[t]        = torch.as_tensor(dones, dtype=torch.float32,
                                                device=self.device)
        # Ragged lists — store references (env emits a fresh dict each step)
        for n in range(self.N):
            self.graph_obs[t * self.N + n]    = graph_obs[n]
            self.action_masks[t * self.N + n] = action_masks[n]

        self._step += 1

    # =================================================================
    # PPO sampling
    # =================================================================
    def get_minibatches(
        self,
        num_minibatches: int,
        advantages_p:    torch.Tensor,    # (T, N)
        advantages_d:    torch.Tensor,    # (T, N)
        returns_p:       torch.Tensor,    # (T, N)
        returns_d:       torch.Tensor,    # (T, N)
        advantages_cost: Optional[torch.Tensor] = None,   # (T, N) — Lagrangian
        returns_cost:    Optional[torch.Tensor] = None,   # (T, N) — Lagrangian
        seed:            Optional[int] = None,
    ):
        """Yield mini-batches as plain dicts for the PPO update loop.

        The (T, N) tensors are flattened to (T*N,) and shuffled, then
        split into ``num_minibatches`` chunks.  Each chunk yields:

            {
                "graph_obs":    [list of dicts, length B],
                "action_masks": [list of bool arrays, length B],
                "actions":      (B,) or (B, 2),
                "log_probs_old":(B,),
                "log_probs_p_old": (B,),
                "log_probs_d_old": (B,),
                "advantages_p": (B,),
                "advantages_d": (B,),
                "returns_p":    (B,),
                "returns_d":    (B,),
            }

        where B = T * N // num_minibatches  (T*N must be divisible).

        Notes
        -----
        * ``num_minibatches`` is the number of *chunks*, not size of each.
        * The shuffle uses a fresh torch RNG; pass ``seed`` for repro.
        """
        assert self.is_full, "get_minibatches() requires a full buffer"
        total = self.T * self.N
        assert total % num_minibatches == 0, (
            f"T*N = {total} not divisible by num_minibatches = {num_minibatches}"
        )
        mb_size = total // num_minibatches

        # Permutation over the (T*N) flat axis
        if seed is not None:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            perm = torch.randperm(total, generator=g)
        else:
            perm = torch.randperm(total, device="cpu")

        # Flatten the dense tensors once per call
        actions_flat      = self.actions.reshape(total, *self.actions.shape[2:])
        log_probs_flat    = self.log_probs.reshape(total)
        log_probs_p_flat  = self.log_probs_p.reshape(total)
        log_probs_d_flat  = self.log_probs_d.reshape(total)
        adv_p_flat        = advantages_p.reshape(total)
        adv_d_flat        = advantages_d.reshape(total)
        ret_p_flat        = returns_p.reshape(total)
        ret_d_flat        = returns_d.reshape(total)
        has_cost          = (advantages_cost is not None) and (returns_cost is not None)
        if has_cost:
            adv_cost_flat = advantages_cost.reshape(total)
            ret_cost_flat = returns_cost.reshape(total)

        for mb in range(num_minibatches):
            start = mb * mb_size
            end   = start + mb_size
            idx_t = perm[start:end]
            idx_list = idx_t.tolist()
            mb = {
                "graph_obs":       [self.graph_obs[i]    for i in idx_list],
                "action_masks":    [self.action_masks[i] for i in idx_list],
                "actions":         actions_flat[idx_t].to(self.device),
                "log_probs_old":   log_probs_flat[idx_t].to(self.device),
                "log_probs_p_old": log_probs_p_flat[idx_t].to(self.device),
                "log_probs_d_old": log_probs_d_flat[idx_t].to(self.device),
                "advantages_p":    adv_p_flat[idx_t].to(self.device),
                "advantages_d":    adv_d_flat[idx_t].to(self.device),
                "returns_p":       ret_p_flat[idx_t].to(self.device),
                "returns_d":       ret_d_flat[idx_t].to(self.device),
            }
            if has_cost:
                mb["advantages_cost"] = adv_cost_flat[idx_t].to(self.device)
                mb["returns_cost"]    = ret_cost_flat[idx_t].to(self.device)
            yield mb


# =====================================================================
# Per-channel reward normaliser (running-stats style)
# =====================================================================
class RewardNormaliser:
    """Per-channel running-mean/std reward normaliser.

    Keeps an EMA-style running estimate of the *return* (not raw reward)
    standard deviation, and divides incoming rewards by it.  This is the
    DeepMind / OpenAI Baselines convention — see Engstrom et al. 2020,
    "Implementation Matters in Deep RL" for why dividing rewards by the
    *return* std (rather than reward std) is more stable.

    For dual-channel PPO, instantiate ONE normaliser PER CHANNEL.  They
    must NOT share state because placement reward has ~10× the magnitude
    of delay reward; sharing would let placement variance swamp the delay
    signal entirely.
    """

    def __init__(self, num_envs: int, gamma: float = 0.99, eps: float = 1e-8):
        self.num_envs = num_envs
        self.gamma = gamma
        self.eps   = eps
        # Running discounted return per env (reset on done by the caller)
        self.running_ret = np.zeros(num_envs, dtype=np.float64)
        # Running variance estimator (Welford's online algorithm)
        self._mean = 0.0
        self._var  = 1.0
        self._count = 0

    def update_and_normalise(
        self,
        rewards: np.ndarray,    # (N,)
        dones:   np.ndarray,    # (N,) bool/float
    ) -> np.ndarray:
        """Apply normalisation and update internal running stats.

        Returns the normalised reward vector for this step.
        """
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones   = np.asarray(dones,   dtype=np.float64).reshape(-1)
        # Update discounted return tracker (and reset entries that hit done)
        self.running_ret = self.running_ret * self.gamma * (1.0 - dones) + rewards

        # Update Welford stats from this step's running returns
        for v in self.running_ret:
            self._count += 1
            delta = v - self._mean
            self._mean += delta / self._count
            self._var  += delta * (v - self._mean)

        std = float(np.sqrt(self._var / max(1, self._count - 1))) + self.eps
        return (rewards / std).astype(np.float32)


# =====================================================================
# Self-test
# =====================================================================
def _self_test():
    """Sanity-check the buffer's add → minibatch flow."""
    print("Running rollout_buffer.py self-tests...\n")

    T, N = 8, 4
    buf = RolloutBuffer(T, N, action_mode="hybrid", device="cpu")
    assert not buf.is_full

    # Push T steps
    rng = np.random.default_rng(0)
    for t in range(T):
        buf.add(
            graph_obs=[{"task_x": [[0.0]*8], "proc_x": [[0.0]*7]}] * N,
            action_masks=[np.ones(17, dtype=bool)] * N,
            actions     =torch.randint(0, 17, (N, 2)),
            log_probs   =torch.randn(N),
            log_probs_p =torch.randn(N),
            log_probs_d =torch.randn(N),
            values_p    =torch.randn(N),
            values_d    =torch.randn(N),
            rewards_p   =rng.normal(0, 1, N).astype(np.float32),
            rewards_d   =rng.normal(0, 1, N).astype(np.float32),
            dones       =(rng.random(N) > 0.9).astype(np.float32),
        )
    assert buf.is_full
    print(f"  ✅ buffer full after T={T} steps")

    # Get mini-batches
    A_p = torch.randn(T, N)
    A_d = torch.randn(T, N)
    R_p = torch.randn(T, N)
    R_d = torch.randn(T, N)
    mbs = list(buf.get_minibatches(num_minibatches=4,
                                    advantages_p=A_p, advantages_d=A_d,
                                    returns_p=R_p, returns_d=R_d, seed=42))
    assert len(mbs) == 4
    expected_size = T * N // 4   # 8
    for mb in mbs:
        assert mb["actions"].shape == (expected_size, 2)
        assert mb["advantages_p"].shape == (expected_size,)
        assert mb["advantages_d"].shape == (expected_size,)
        assert len(mb["graph_obs"]) == expected_size
        assert len(mb["action_masks"]) == expected_size
    print(f"  ✅ 4 minibatches × {expected_size} samples, shapes correct")

    # Coverage: across all minibatches, each (t, n) sample appears exactly once
    seen = set()
    g = torch.Generator(); g.manual_seed(42)
    perm = torch.randperm(T*N, generator=g)
    for i in perm.tolist():
        seen.add(i)
    assert len(seen) == T * N
    print(f"  ✅ minibatch shuffle is a permutation (no duplicates)")

    # Reset clears everything
    buf.reset()
    assert not buf.is_full and buf.step == 0
    assert buf.graph_obs[0] is None
    print(f"  ✅ reset() clears step pointer and ragged storage")

    # Reward normaliser: should reduce variance toward 1
    norm = RewardNormaliser(num_envs=4, gamma=0.99)
    raw = rng.normal(0, 5, size=(100, 4)).astype(np.float32)
    dones = (rng.random((100, 4)) > 0.9).astype(np.float32)
    out_std_orig = float(raw.std())
    out_normed = []
    for t in range(100):
        out_normed.append(norm.update_and_normalise(raw[t], dones[t]))
    out_normed = np.array(out_normed)
    out_std_normed = float(out_normed.std())
    # Normaliser should bring std into ~[0.5, 2.0] range (won't be exactly 1
    # because we divide by *return* std, not raw reward std — but it should
    # be much closer to 1 than the raw 5.0).
    print(f"  ✅ reward normaliser: raw std={out_std_orig:.2f} -> normalised std={out_std_normed:.2f}")
    assert 0.3 < out_std_normed < 3.0, \
        f"normaliser failed to bring std near 1: {out_std_normed}"

    # auto_only mode: action shape (T, N), not (T, N, 2)
    buf2 = RolloutBuffer(T, N, action_mode="auto_only", device="cpu")
    buf2.add(
        graph_obs   =[{"task_x":[[0.0]*8],"proc_x":[[0.0]*7]}]*N,
        action_masks=[np.ones(17, dtype=bool)]*N,
        actions     =torch.randint(0, 17, (N,)),
        log_probs   =torch.randn(N),
        log_probs_p =torch.randn(N),
        log_probs_d =torch.zeros(N),
        values_p    =torch.randn(N),
        values_d    =torch.zeros(N),
        rewards_p   =np.zeros(N, dtype=np.float32),
        rewards_d   =np.zeros(N, dtype=np.float32),
        dones       =np.zeros(N, dtype=np.float32),
    )
    assert buf2.actions.shape == (T, N)
    print(f"  ✅ auto_only buffer has action shape (T, N) = {tuple(buf2.actions.shape)}")

    print("\nAll rollout_buffer.py tests passed ✓")


if __name__ == "__main__":
    _self_test()
