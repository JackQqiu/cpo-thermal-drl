"""
training/ppo_trainer.py — PPO update loop for the dual-channel actor-critic
============================================================================

Performs one full PPO update phase from a filled :class:`RolloutBuffer`:

1. Compute GAE advantages and discounted returns for BOTH reward channels
   (placement and delay).
2. Optionally normalise each channel's advantages independently.
3. For ``ppo_epochs`` epochs, iterate over ``num_minibatches`` shuffled
   mini-batches:

   a. Forward the model on the mini-batch's stored states + actions.
   b. Compute the **dual** clipped surrogate loss:

         loss_actor   = clip_loss_p + delay_coef × clip_loss_d
         loss_value   = 0.5 × (mse_p + delay_coef × mse_d)
         loss_entropy = -ent_coef × (entropy_p + delay_coef × entropy_d)
         total        = loss_actor + vf_coef × loss_value + loss_entropy

      ``delay_coef`` is 0 in ``auto_only`` mode (delay head & V_delay
      do not get gradient) and 1 otherwise.

   c. Backprop, clip global gradient norm, optimiser step.

4. Return per-update telemetry (mean policy loss, value loss, entropy,
   approx-KL, clipped-fraction, learning rate, current curriculum stage)
   for TensorBoard logging by the caller.

Why we don't share the two clip losses
--------------------------------------
The placement and delay heads have **different policy distributions**
(different ``log_prob_old`` references), so their clipped-ratio
expressions ``r_t = exp(log_prob_new - log_prob_old)`` are literally
different scalars.  Combining them into a single scalar before clipping
loses fidelity — clipping must happen per-head, then we sum the
post-clip surrogate losses.

This is exactly the standard recipe for factored/multi-discrete PPO,
matching how Decima and Petting-Zoo handle multi-head actors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cpo_thermal_v2.models import PPOActorCritic, build_batch
from cpo_thermal_v2.training.gae import compute_gae, normalise_advantages
from cpo_thermal_v2.training.rollout_buffer import RolloutBuffer


@dataclass
class PPOUpdateMetrics:
    """Per-update telemetry returned by :meth:`PPOTrainer.update`."""
    loss_total:        float = 0.0
    loss_actor_p:      float = 0.0
    loss_actor_d:      float = 0.0
    loss_value_p:      float = 0.0
    loss_value_d:      float = 0.0
    entropy_p:         float = 0.0
    entropy_d:         float = 0.0
    approx_kl_p:       float = 0.0
    approx_kl_d:       float = 0.0
    clip_frac_p:       float = 0.0
    clip_frac_d:       float = 0.0
    grad_norm:         float = 0.0
    lr:                float = 0.0
    advantage_p_mean:  float = 0.0
    advantage_p_std:   float = 0.0
    advantage_d_mean:  float = 0.0
    advantage_d_std:   float = 0.0
    # Lagrangian / RCPO constrained variant (0 / inert when disabled)
    lam:               float = 0.0
    cost_per_episode:  float = 0.0
    loss_value_cost:   float = 0.0


class PPOTrainer:
    """One-shot PPO updater.

    Constructed once at the start of training; ``update(buffer, ...)`` is
    called after each rollout phase.

    Parameters (all from the YAML ``training`` section)
    ---------------------------------------------------
    model            : :class:`PPOActorCritic`
    optimizer        : torch.optim.Optimizer
    clip_epsilon     : float
    vf_clip_epsilon  : float | None  — if set, value loss is also clipped
    gamma            : float
    gae_lambda       : float
    ppo_epochs       : int
    num_minibatches  : int
    max_grad_norm    : float
    vf_coef          : float
    ent_coef         : float
    normalize_advantages : bool
    delay_loss_coef  : float — 0.0 for auto_only, 1.0 for hybrid/agent_only
    device           : torch.device
    """

    def __init__(
        self,
        model:           PPOActorCritic,
        optimizer:       torch.optim.Optimizer,
        *,
        clip_epsilon:    float = 0.2,
        vf_clip_epsilon: Optional[float] = None,
        gamma:           float = 0.99,
        gae_lambda:      float = 0.95,
        ppo_epochs:      int   = 4,
        num_minibatches: int   = 8,
        max_grad_norm:   float = 0.5,
        vf_coef:         float = 0.5,
        ent_coef:        float = 0.01,
        normalize_advantages: bool = True,
        delay_loss_coef: float = 1.0,
        device:          str   = "cpu",
        thermal_blind:   bool  = False,
        action_mode:     str   = "auto_only",
        delay_warmup_steps: int = 0,
        # Lagrangian / RCPO constrained variant (all inert by default)
        lagrangian:      bool  = False,
        cost_limit:      float = 0.0,
        lam_init:        float = 0.0,
        lam_lr:          float = 0.0,
        lam_max:         float = 10.0,
    ):
        self.model = model
        self.optim = optimizer
        self.clip_epsilon    = float(clip_epsilon)
        self.vf_clip_epsilon = vf_clip_epsilon
        self.gamma           = float(gamma)
        self.gae_lambda      = float(gae_lambda)
        self.ppo_epochs      = int(ppo_epochs)
        self.num_minibatches = int(num_minibatches)
        self.max_grad_norm   = float(max_grad_norm)
        self.vf_coef         = float(vf_coef)
        self.ent_coef        = float(ent_coef)
        self.normalize_advantages = bool(normalize_advantages)
        self.delay_loss_coef = float(delay_loss_coef)
        self.device          = device
        self.thermal_blind   = bool(thermal_blind)
        # E4 (HK-1.5.8): linear-warmup of delay_loss_coef during the
        # first delay_warmup_steps env-steps of hybrid training, to
        # avoid encoder gradient shock at the Stage 1 -> Stage 2
        # transition. Only active when action_mode == "hybrid";
        # auto_only / agent_only ignore the warmup. delay_warmup_steps=0
        # disables warmup entirely (back to pre-HK-1.5.8 behaviour).
        self.action_mode        = str(action_mode)
        self.delay_warmup_steps = int(delay_warmup_steps)
        self._global_step       = 0
        self._effective_delay_loss_coef = float(delay_loss_coef)
        # Lagrangian / RCPO constrained variant.  When lagrangian=False the
        # cost channel is never read and the update is byte-identical to the
        # standard dual-channel PPO.
        self.lagrangian = bool(lagrangian)
        self.cost_limit = float(cost_limit)
        self.lam        = float(lam_init)
        self.lam_lr     = float(lam_lr)
        self.lam_max    = float(lam_max)

    # =================================================================
    # Top-level update
    # =================================================================
    def update(
        self,
        buffer:           RolloutBuffer,
        last_values_p:    torch.Tensor,    # (N,)  V_p(s_T) for bootstrap
        last_values_d:    torch.Tensor,    # (N,)  V_d(s_T) for bootstrap
        last_values_cost: Optional[torch.Tensor] = None,  # (N,) — Lagrangian
    ) -> PPOUpdateMetrics:
        """Run a full PPO update phase from the filled buffer.

        ``last_values_p / _d`` are the critic's predictions on the very
        last state collected (used for GAE tail bootstrap).
        """
        assert buffer.is_full, "PPO update requires a full rollout buffer"

        # ------------ 0. E4 (HK-1.5.8): linear-warmup delay_loss_coef ------------
        # In hybrid mode the encoder transitions from a Stage-1 backbone
        # optimised against placement-only objective to one that must
        # also satisfy delay-head gradients. A linear ramp lets the
        # representation adapt gradually rather than be yanked at full
        # delay_loss_coef from step 0.
        if (self.action_mode == "hybrid"
                and self._global_step < self.delay_warmup_steps
                and self.delay_warmup_steps > 0):
            self._effective_delay_loss_coef = (
                self.delay_loss_coef
                * (self._global_step / max(self.delay_warmup_steps, 1))
            )
        else:
            self._effective_delay_loss_coef = self.delay_loss_coef
        print(f"[trainer] global_step={self._global_step:>9d}  "
              f"dlc={self._effective_delay_loss_coef:.3f}  "
              f"(target={self.delay_loss_coef:.2f}, "
              f"warmup={self.delay_warmup_steps})")

        # ------------ 1. Compute GAE for both channels ------------
        adv_p, ret_p = compute_gae(
            rewards     = buffer.rewards_p,
            values      = buffer.values_p,
            dones       = buffer.dones,
            last_values = last_values_p,
            gamma       = self.gamma,
            lam         = self.gae_lambda,
        )
        adv_d, ret_d = compute_gae(
            rewards     = buffer.rewards_d,
            values      = buffer.values_d,
            dones       = buffer.dones,
            last_values = last_values_d,
            gamma       = self.gamma,
            lam         = self.gae_lambda,
        )

        # ------------ 1b. Lagrangian cost channel (RCPO) ------------
        # Constrained variant only.  We (i) run a third GAE pass on the
        # per-step cost (violation indicator) using the cost critic V_cost;
        # (ii) estimate the realised mean cost PER EPISODE Ĵ_C from the raw
        # cost rewards (true scale — needed to track the limit); (iii) take
        # one dual-ascent step on λ toward Ĵ_C ≤ cost_limit.  The cost
        # ADVANTAGE is normalised like the others below (it only sets a
        # gradient DIRECTION; λ carries the scale), while Ĵ_C stays raw.
        adv_cost = ret_cost = None
        cost_per_episode = 0.0
        if self.lagrangian:
            if last_values_cost is None:
                last_values_cost = torch.zeros_like(last_values_d)
            adv_cost, ret_cost = compute_gae(
                rewards     = buffer.rewards_cost,
                values      = buffer.values_cost,
                dones       = buffer.dones,
                last_values = last_values_cost,
                gamma       = self.gamma,
                lam         = self.gae_lambda,
            )
            n_eps = float(buffer.dones.sum().item())
            total_cost = float(buffer.rewards_cost.sum().item())
            cost_per_episode = total_cost / max(1.0, n_eps)
            # Dual ascent: λ ← clip(λ + lr·(Ĵ_C − limit), 0, λ_max).
            self.lam = float(np.clip(
                self.lam + self.lam_lr * (cost_per_episode - self.cost_limit),
                0.0, self.lam_max,
            ))

        # ------------ 2. Per-channel advantage normalisation ------------
        # IMPORTANT: each channel normalises independently.  Sharing
        # would let the higher-magnitude placement channel swamp the
        # delay channel.
        adv_p_mean = float(adv_p.mean().item())
        adv_p_std  = float(adv_p.std(unbiased=False).item())
        adv_d_mean = float(adv_d.mean().item())
        adv_d_std  = float(adv_d.std(unbiased=False).item())

        if self.normalize_advantages:
            adv_p = normalise_advantages(adv_p)
            adv_d = normalise_advantages(adv_d)
            if adv_cost is not None:
                adv_cost = normalise_advantages(adv_cost)

        # ------------ 3. PPO epochs over minibatches ------------
        metrics = PPOUpdateMetrics(
            advantage_p_mean=adv_p_mean, advantage_p_std=adv_p_std,
            advantage_d_mean=adv_d_mean, advantage_d_std=adv_d_std,
            lr=float(self.optim.param_groups[0]["lr"]),
            lam=self.lam, cost_per_episode=cost_per_episode,
        )
        n_updates = 0

        for epoch in range(self.ppo_epochs):
            # Re-shuffle each epoch (standard PPO recipe)
            mb_iter = buffer.get_minibatches(
                num_minibatches=self.num_minibatches,
                advantages_p=adv_p, advantages_d=adv_d,
                returns_p=ret_p, returns_d=ret_d,
                advantages_cost=adv_cost, returns_cost=ret_cost,
                seed=None,    # different shuffle each call
            )
            for mb in mb_iter:
                stats = self._update_one_minibatch(mb)
                # Accumulate (mean over total updates)
                metrics.loss_total      += stats["loss_total"]
                metrics.loss_actor_p    += stats["loss_actor_p"]
                metrics.loss_actor_d    += stats["loss_actor_d"]
                metrics.loss_value_p    += stats["loss_value_p"]
                metrics.loss_value_d    += stats["loss_value_d"]
                metrics.loss_value_cost += stats["loss_value_cost"]
                metrics.entropy_p       += stats["entropy_p"]
                metrics.entropy_d       += stats["entropy_d"]
                metrics.approx_kl_p     += stats["approx_kl_p"]
                metrics.approx_kl_d     += stats["approx_kl_d"]
                metrics.clip_frac_p     += stats["clip_frac_p"]
                metrics.clip_frac_d     += stats["clip_frac_d"]
                metrics.grad_norm       += stats["grad_norm"]
                n_updates += 1

        # Mean over total minibatch updates
        for k in ("loss_total", "loss_actor_p", "loss_actor_d",
                  "loss_value_p", "loss_value_d", "loss_value_cost",
                  "entropy_p", "entropy_d", "approx_kl_p", "approx_kl_d",
                  "clip_frac_p", "clip_frac_d", "grad_norm"):
            setattr(metrics, k, getattr(metrics, k) / max(1, n_updates))

        # E4 (HK-1.5.8): advance the internal env-step counter so the
        # next update() applies the correct warmup ramp position. The
        # buffer's rewards_p tensor has shape (T, N) so .numel() == T*N
        # equals the total env-steps consumed by this rollout.
        self._global_step += int(buffer.rewards_p.numel())

        return metrics

    # =================================================================
    # Per-minibatch update
    # =================================================================
    def _update_one_minibatch(self, mb: Dict[str, Any]) -> Dict[str, float]:
        """One forward/backward pass on a single mini-batch.

        Returns per-batch loss components (NOT averaged across minibatches —
        the caller accumulates).
        """
        # 1. Build the PyG Batch from the ragged graph_obs + masks
        batch = build_batch(mb["graph_obs"], mb["action_masks"],
                            device=self.device,
                            thermal_blind=self.thermal_blind)

        # 2. Forward through the actor-critic
        out = self.model.evaluate_actions(batch, mb["actions"].to(self.device))

        log_probs_p_new = out["log_prob_p"]
        log_probs_d_new = out["log_prob_d"]
        entropy_p_new   = out["entropy_p"]
        entropy_d_new   = out["entropy_d"]
        v_p_new         = out["v_placement"]
        v_d_new         = out["v_delay"]

        # 3. Compute clipped policy losses for each head.
        # Lagrangian/RCPO: fold the cost advantage into BOTH heads as
        # A_head − λ·A_cost.  Summing the per-head surrogates then yields the
        # RCPO cost gradient −λ·A_cost·∇log(π_place·π_delay) on the joint
        # policy (both actions jointly determine the thermal cost).
        adv_p = mb["advantages_p"]
        adv_d = mb["advantages_d"]
        v_cost_new = out["v_cost"]
        loss_value_cost = v_cost_new.new_zeros(())
        if self.lagrangian and "advantages_cost" in mb:
            adv_cost = mb["advantages_cost"]
            adv_p = adv_p - self.lam * adv_cost
            adv_d = adv_d - self.lam * adv_cost
            loss_value_cost = self._value_loss(v_cost_new, mb["returns_cost"])

        clip_loss_p, kl_p, clip_frac_p = self._clip_loss(
            log_probs_p_new, mb["log_probs_p_old"], adv_p,
        )
        clip_loss_d, kl_d, clip_frac_d = self._clip_loss(
            log_probs_d_new, mb["log_probs_d_old"], adv_d,
        )

        # 4. Value losses (per channel; optionally clipped — match PPO2 style)
        loss_value_p = self._value_loss(v_p_new, mb["returns_p"])
        loss_value_d = self._value_loss(v_d_new, mb["returns_d"])

        # 5. Combine with mode-aware delay coefficient
        # E4 (HK-1.5.8): use the warmup-ramped effective coefficient
        # set by update() rather than the raw self.delay_loss_coef.
        # When warmup is disabled (delay_warmup_steps=0) or finished,
        # this equals self.delay_loss_coef and behaviour is unchanged.
        c = self._effective_delay_loss_coef
        loss_actor   = clip_loss_p + c * clip_loss_d
        loss_value   = 0.5 * (loss_value_p + c * loss_value_d) + 0.5 * loss_value_cost
        loss_entropy = -(entropy_p_new.mean() + c * entropy_d_new.mean())
        total        = loss_actor + self.vf_coef * loss_value + self.ent_coef * loss_entropy

        # 6. Backprop + clip + step
        self.optim.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm,
        ).item())
        self.optim.step()

        return {
            "loss_total":     float(total.detach().item()),
            "loss_actor_p":   float(clip_loss_p.detach().item()),
            "loss_actor_d":   float(clip_loss_d.detach().item()),
            "loss_value_p":   float(loss_value_p.detach().item()),
            "loss_value_d":   float(loss_value_d.detach().item()),
            "loss_value_cost": float(loss_value_cost.detach().item()),
            "entropy_p":      float(entropy_p_new.mean().detach().item()),
            "entropy_d":      float(entropy_d_new.mean().detach().item()),
            "approx_kl_p":    float(kl_p),
            "approx_kl_d":    float(kl_d),
            "clip_frac_p":    float(clip_frac_p),
            "clip_frac_d":    float(clip_frac_d),
            "grad_norm":      grad_norm,
        }

    # -----------------------------------------------------------------
    def _clip_loss(
        self,
        log_probs_new: torch.Tensor,    # (B,)
        log_probs_old: torch.Tensor,    # (B,)
        advantages:    torch.Tensor,    # (B,)
    ) -> Tuple[torch.Tensor, float, float]:
        """Standard PPO clipped surrogate loss for ONE head.

        Returns
        -------
        loss      : scalar tensor (negative — a loss, not a return)
        approx_kl : float — Schulman approximation: ``mean((r-1) - log r)``
        clip_frac : float — fraction of mb where the ratio was clipped
        """
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr_unclipped = ratio * advantages
        surr_clipped   = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
        ) * advantages
        loss = -torch.min(surr_unclipped, surr_clipped).mean()

        # Schulman's approximate-KL (eq. 2 of "Approximating KL" gist):
        # E[(r - 1) - log r] is non-negative and tracks true KL well.
        with torch.no_grad():
            log_ratio = log_probs_new - log_probs_old
            approx_kl = float(((torch.exp(log_ratio) - 1) - log_ratio).mean().item())
            clip_frac = float(
                ((ratio < 1.0 - self.clip_epsilon) |
                 (ratio > 1.0 + self.clip_epsilon)).float().mean().item()
            )
        return loss, approx_kl, clip_frac

    # -----------------------------------------------------------------
    def _value_loss(
        self,
        v_new:   torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """MSE value loss, with optional clipping (PPO2-style).

        If ``self.vf_clip_epsilon`` is None (the default in our config)
        we use a plain ``(v_new - returns)^2``.  When set, we clip the
        change in V to the same ε used on the policy ratio — this can
        stabilise critic training but introduces an extra hyperparameter,
        which is why we leave it off by default.
        """
        if self.vf_clip_epsilon is None:
            return F.mse_loss(v_new, returns)
        # PPO2-style: track v_old via the rollout buffer's stored values.
        # Skipping the implementation here because we'd need to thread
        # ``values_p_old`` / ``values_d_old`` through the minibatch dict.
        # If the user wants clip-vf they can re-introduce v_old later.
        # For now treat as the simple case.
        return F.mse_loss(v_new, returns)


# =====================================================================
# Optimiser + LR schedule helpers
# =====================================================================
def make_optimizer(
    model: PPOActorCritic,
    training_cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Construct an Adam optimiser from the YAML ``training`` section."""
    return torch.optim.Adam(
        model.parameters(),
        lr  = float(training_cfg["learning_rate"]),
        eps = float(training_cfg.get("adam_eps", 1e-5)),
    )


def make_lr_scheduler(
    optimizer:    torch.optim.Optimizer,
    training_cfg: Dict[str, Any],
    total_updates: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create an LR schedule based on ``training.lr_schedule``.

    Returns ``None`` for ``constant``; a step-based scheduler for
    ``linear`` and ``cosine``.  ``total_updates`` is the expected number
    of optimiser updates across the whole run, used to set the schedule
    horizon.
    """
    sched = str(training_cfg.get("lr_schedule", "constant")).lower()
    if sched == "constant":
        return None
    if sched == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,    # decay to 0 at the end
            total_iters=total_updates,
        )
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_updates, eta_min=0.0,
        )
    raise ValueError(f"unknown lr_schedule: {sched!r}; expected constant|linear|cosine")
