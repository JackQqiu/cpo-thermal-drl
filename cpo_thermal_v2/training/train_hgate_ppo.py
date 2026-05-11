"""
training/train_hgate_ppo.py — PPO entrypoint for HGATE-PPO baseline
====================================================================

Multi-env, minibatched PPO trainer.  Invoked via train.py's
``model.architecture: hgate_ppo`` dispatch (parallel to HK-3.0's
``algorithm: reinforce`` dispatch routing to train_decima_true).

Why a separate file rather than reusing the existing PPO trainer
---------------------------------------------------------------
The existing PPO trainer at training/train.py + ppo_trainer.py is
deeply coupled to PPOActorCritic's dual-channel batched interface:
``model.act(batch)`` takes a PyG Batch and returns log_prob_p /
log_prob_d / v_placement / v_delay.  HGATEActorCritic operates on a
single graph_obs dict and is single-critic.  Adapting the existing
trainer to consume HGATEActorCritic would either require a
PyG-Batch -> dict converter (fragile) or a refactor of the trainer's
inner loop (invasive).  The cleaner separation is to write a
purpose-built PPO loop here.

HK-4.5 perf-fix layout
- Fix A: vector-env rollout (yaml ``training.num_envs`` honoured;
  default async).  Per env-step we forward the model once per env,
  then ``vec_env.step`` runs N envs in parallel — eliminating the
  single-env CPU bottleneck.
- Fix B: minibatched PPO update.  After rollout, transitions are
  flattened (T*N), shuffled per epoch, and updated in mini-batches of
  ``minibatch_size`` (default 64) with ONE backward+step per
  minibatch — mirroring Ours' ppo_trainer.py.
- Fix 2: log informative metrics — approx_kl (Schulman), clip_frac,
  entropy (mean of softmax entropies).  Replaces the loss_pg=0
  logging artifact that arises from summing signed per-transition
  losses over normalised advantages.

What it does NOT cover
- Reward channel decomposition (HGATE is single-critic; the env's
  reward_placement / reward_delay info entries are summed into a
  single scalar reward channel for the update)
- LR schedule (constant LR — Wu 2025 baseline is short-horizon
  enough that scheduling doesn't move the needle)
"""
from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cpo_thermal_v2.baselines.hgate_ppo import HGATEActorCritic
from cpo_thermal_v2.training.config_loader import save_resolved_config
from cpo_thermal_v2.training.env_factory   import (
    make_vector_env, smoke_test_vector_env,
    _extract_per_env,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:                                     # pragma: no cover
    _TB_AVAILABLE = False


# =====================================================================
# Helpers
# =====================================================================
def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_graph_obs_list(info: Dict[str, Any], num_envs: int) -> List[Dict[str, Any]]:
    """Pull per-env graph_obs dicts out of a vectorised info.

    Same unwrap pattern as training/train.py — copied locally rather
    than imported to keep this trainer self-contained.
    """
    raw_list = _extract_per_env(info, "graph_obs", num_envs)
    out: List[Dict[str, Any]] = []
    for n, item in enumerate(raw_list):
        if isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
            item = item[0]
        if not isinstance(item, dict):
            raise TypeError(
                f"graph_obs[{n}] has unexpected type {type(item).__name__}"
            )
        out.append(item)
    return out


def _extract_action_masks(info: Dict[str, Any], num_envs: int) -> List[np.ndarray]:
    raw_list = _extract_per_env(info, "action_mask", num_envs)
    out: List[np.ndarray] = []
    for n, m in enumerate(raw_list):
        if m is None:
            raise RuntimeError(f"env {n} did not emit an action_mask in info")
        out.append(np.asarray(m, dtype=bool))
    return out


def _compute_gae(
    rewards:    List[float],
    values:     List[float],
    dones:      List[bool],
    last_value: float,
    gamma:      float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages + returns for one rollout (single env).

    Returns (advantages, returns) both of length len(rewards).
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def _evaluate_action(
    model:       HGATEActorCritic,
    graph_obs:   Dict[str, Any],
    action_mask: np.ndarray,
    action:      int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (new_log_prob, entropy, value) for a STORED action under
    the current policy."""
    logits, value = model.forward(graph_obs, action_mask)
    device = logits.device
    mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
    masked = logits.masked_fill(~mask_t, float("-inf"))
    log_probs = F.log_softmax(masked, dim=-1)
    probs     = log_probs.exp()
    new_log_prob = log_probs[int(action)]
    entropy = -(probs[mask_t] * log_probs[mask_t]).sum()
    return new_log_prob, entropy, value


# =====================================================================
# Rollout container + per-phase helpers
# =====================================================================
@dataclasses.dataclass
class RolloutBatch:
    """Flat rollout snapshot, length M = T_actual * num_envs."""
    graph_obs:     List[Dict[str, Any]]
    masks:         List[np.ndarray]
    actions:       np.ndarray            # (M,) int64
    old_log_probs: torch.Tensor          # (M,)
    advantages:    torch.Tensor          # (M,)  normalised
    returns:       torch.Tensor          # (M,)


def _collect_rollout(
    model:                  HGATEActorCritic,
    vec_env:                Any,
    info_init:              Dict[str, Any],
    num_envs:               int,
    rollout_length:         int,
    gamma:                  float,
    gae_lambda:             float,
    device:                 torch.device,
    global_step:            int,
    total_steps:            int,
    episode_returns_buffer: np.ndarray,
    ep_returns_out:         List[float],
) -> Tuple[RolloutBatch, Dict[str, Any], int]:
    """Collect rollout_length env-steps across num_envs parallel envs.

    Per env-step:
      1. forward the model once per env (model is single-graph), assemble
         per-env action / log_prob / value
      2. vec_env.step(actions)  — runs N envs in parallel
      3. snapshot pre-step (graph_obs, mask, action, log_prob, value)
         and post-step (reward, done) into rollout buffers

    Returns
    -------
    rollout      : RolloutBatch with flat (M = T_actual * num_envs) lists
    info         : the most recent info dict, for next rollout's first step
    global_step  : updated global step counter (incremented by num_envs/iter)
    """
    info = info_init
    graph_obs_list = _extract_graph_obs_list(info, num_envs)
    action_masks   = _extract_action_masks  (info, num_envs)

    roll_graph_obs:    List[List[Dict[str, Any]]] = []
    roll_masks:        List[List[np.ndarray]]     = []
    roll_actions:      List[np.ndarray]           = []
    roll_log_probs:    List[List[torch.Tensor]]   = []
    roll_values:       List[np.ndarray]           = []
    roll_rewards:      List[np.ndarray]           = []
    roll_dones:        List[np.ndarray]           = []

    model.eval()
    with torch.no_grad():
        for _t in range(rollout_length):
            actions_arr = np.zeros(num_envs, dtype=np.int64)
            log_probs_t: List[torch.Tensor] = []
            values_t = np.zeros(num_envs, dtype=np.float32)
            for n in range(num_envs):
                out = model.act(graph_obs_list[n], action_masks[n],
                                 deterministic=False)
                a = out["action"]
                actions_arr[n] = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
                log_probs_t.append(out["log_prob"].detach())
                values_t[n] = float(out["value"].item())

            # snapshot PRE-step state (the (s_t, a_t) the model just acted on)
            roll_graph_obs.append(graph_obs_list)
            roll_masks.append(action_masks)
            roll_actions.append(actions_arr.copy())
            roll_log_probs.append(log_probs_t)
            roll_values.append(values_t)

            _obs, rewards_arr, term, trunc, info = vec_env.step(actions_arr)
            rewards_arr = np.asarray(rewards_arr, dtype=np.float32).reshape(-1)
            term  = np.asarray(term ).reshape(-1).astype(bool)
            trunc = np.asarray(trunc).reshape(-1).astype(bool)
            done  = term | trunc

            roll_rewards.append(rewards_arr)
            roll_dones  .append(done)

            episode_returns_buffer += rewards_arr
            for n in range(num_envs):
                if done[n]:
                    ep_returns_out.append(float(episode_returns_buffer[n]))
                    episode_returns_buffer[n] = 0.0

            global_step += num_envs
            graph_obs_list = _extract_graph_obs_list(info, num_envs)
            action_masks   = _extract_action_masks  (info, num_envs)

            if global_step >= total_steps:
                break

        # bootstrap tail values per env (for GAE)
        T_actual = len(roll_actions)
        last_values = np.zeros(num_envs, dtype=np.float32)
        for n in range(num_envs):
            last_values[n] = float(model.get_value(graph_obs_list[n]).item())

    # ----- per-env GAE -----
    advantages_arr = np.zeros((T_actual, num_envs), dtype=np.float32)
    returns_arr    = np.zeros((T_actual, num_envs), dtype=np.float32)
    for n in range(num_envs):
        rewards_n = [float(r[n]) for r in roll_rewards]
        values_n  = [float(v[n]) for v in roll_values]
        dones_n   = [bool (d[n]) for d in roll_dones ]
        adv_n, ret_n = _compute_gae(
            rewards_n, values_n, dones_n,
            last_value=float(last_values[n]),
            gamma=gamma, gae_lambda=gae_lambda,
        )
        for t in range(T_actual):
            advantages_arr[t, n] = adv_n[t]
            returns_arr   [t, n] = ret_n[t]

    # ----- flatten (T, N) -> M = T*N -----
    flat_graph_obs:     List[Dict[str, Any]] = []
    flat_masks:         List[np.ndarray]     = []
    flat_actions:       List[int]            = []
    flat_log_probs_old: List[torch.Tensor]   = []
    flat_advantages:    List[float]          = []
    flat_returns:       List[float]          = []
    for t in range(T_actual):
        for n in range(num_envs):
            flat_graph_obs.append(roll_graph_obs[t][n])
            flat_masks.append(roll_masks[t][n])
            flat_actions.append(int(roll_actions[t][n]))
            flat_log_probs_old.append(roll_log_probs[t][n])
            flat_advantages.append(float(advantages_arr[t, n]))
            flat_returns   .append(float(returns_arr   [t, n]))

    adv_t = torch.tensor(flat_advantages, dtype=torch.float32, device=device)
    ret_t = torch.tensor(flat_returns,    dtype=torch.float32, device=device)
    # Per-rollout advantage normalisation (standard PPO)
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    rollout = RolloutBatch(
        graph_obs     = flat_graph_obs,
        masks         = flat_masks,
        actions       = np.array(flat_actions, dtype=np.int64),
        old_log_probs = torch.stack(flat_log_probs_old).to(device),
        advantages    = adv_t,
        returns       = ret_t,
    )
    return rollout, info, global_step


def _ppo_update(
    model:          HGATEActorCritic,
    optimizer:      torch.optim.Optimizer,
    rollout:        RolloutBatch,
    ppo_epochs:     int,
    minibatch_size: int,
    clip_eps:       float,
    value_coef:     float,
    entropy_coef:   float,
    max_grad_norm:  float,
    device:         torch.device,
) -> Dict[str, float]:
    """Shuffled-minibatch PPO update.

    For each (ppo_epoch, minibatch) we:
      - re-evaluate the stored actions under the *current* policy
        (per-transition forward, since HGATEActorCritic is single-graph)
      - compute clipped surrogate loss + value MSE + entropy bonus on the
        full minibatch, then ONE backward + optimizer.step per minibatch
    Returns the mean-over-minibatches of each metric.
    """
    model.train()
    M = len(rollout.actions)
    mb_size = max(1, int(minibatch_size))
    num_minibatches = max(1, M // mb_size)

    sums = {
        "loss_pg":   0.0, "loss_v":   0.0, "loss_ent": 0.0,
        "approx_kl": 0.0, "clip_frac": 0.0, "entropy":  0.0,
    }
    n_updates = 0

    rollout_device = rollout.old_log_probs.device

    for _epoch in range(ppo_epochs):
        perm = np.random.permutation(M)
        for mb_idx in range(num_minibatches):
            mb_start = mb_idx * mb_size
            mb_end   = (mb_idx + 1) * mb_size if mb_idx < num_minibatches - 1 else M
            mb_indices = perm[mb_start:mb_end]

            mb_new_log_probs: List[torch.Tensor] = []
            mb_entropies:     List[torch.Tensor] = []
            mb_values:        List[torch.Tensor] = []
            for idx in mb_indices:
                new_lp, ent, v = _evaluate_action(
                    model,
                    rollout.graph_obs[int(idx)],
                    rollout.masks    [int(idx)],
                    int(rollout.actions[int(idx)]),
                )
                mb_new_log_probs.append(new_lp.view(-1))
                mb_entropies    .append(ent.view(-1))
                mb_values       .append(v.view(-1))

            new_log_probs = torch.cat(mb_new_log_probs)   # (B,)
            entropies     = torch.cat(mb_entropies)       # (B,)
            values        = torch.cat(mb_values)          # (B,)

            mb_idx_t = torch.as_tensor(
                mb_indices, dtype=torch.long, device=rollout_device)
            old_log_probs = rollout.old_log_probs[mb_idx_t]
            advs          = rollout.advantages   [mb_idx_t]
            rets          = rollout.returns      [mb_idx_t]

            log_ratio = new_log_probs - old_log_probs
            ratio     = log_ratio.exp()
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advs
            loss_pg  = -torch.min(surr1, surr2).mean()
            loss_v   =  F.mse_loss(values, rets)
            loss_ent = -entropies.mean() * entropy_coef
            loss     =  loss_pg + value_coef * loss_v + loss_ent

            with torch.no_grad():
                approx_kl  = float(((ratio - 1.0) - log_ratio).mean().item())
                clip_frac  = float(
                    ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps))
                    .float().mean().item()
                )
                ent_mean   = float(entropies.mean().item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            sums["loss_pg"]   += float(loss_pg .item())
            sums["loss_v"]    += float(loss_v  .item())
            sums["loss_ent"]  += float(loss_ent.item())
            sums["approx_kl"] += approx_kl
            sums["clip_frac"] += clip_frac
            sums["entropy"]   += ent_mean
            n_updates += 1

    n_div = max(1, n_updates)
    return {k: v / n_div for k, v in sums.items()}


# =====================================================================
# Main entrypoint
# =====================================================================
def train_hgate_ppo(cfg: Dict[str, Any]) -> None:
    """Multi-env minibatched PPO loop for HGATEActorCritic.

    Reads the same yaml schema as the dual-channel PPO trainer; only
    the keys actually used here are described below.

      training:
        algorithm:       ppo                   # filter, default
        device:          auto | cuda | cpu
        total_steps:     int
        num_envs:        int                   # default 16  (HK-4.5 Fix A)
        rollout_length:  int                   # default 256
        minibatch_size:  int                   # default 64  (HK-4.5 Fix B)
        learning_rate:   float                 # default 3e-4
        clip_eps:        float                 # default 0.2
        value_coef:      float                 # default 0.5
        entropy_coef:    float                 # default 0.01
        gae_lambda:      float                 # default 0.95
        gamma:           float                 # default 0.99
        ppo_epochs:      int                   # default 4
        max_grad_norm:   float                 # default 0.5
        save_every_episodes: int               # default 200
        seed_base:       int                   # default 42
        vector_env_mode: 'async' | 'sync'      # default 'async'

      model:
        architecture:    hgate_ppo
        hidden_dim:      int                   # default 128
        num_gat_layers:  int                   # default 2
        num_heads:       int                   # default 4
    """
    train_cfg = cfg.get("training", {}) or {}
    log_cfg   = cfg.get("logging",  {}) or {}
    model_cfg = cfg.get("model",    {}) or {}
    env_cfg   = cfg.get("env",      {}) or {}

    # ----- hyper-params -----
    seed_base       = int  (train_cfg.get("seed_base",       42))
    total_steps     = int  (train_cfg.get("total_steps",     5_000_000))
    num_envs        = int  (train_cfg.get("num_envs",        16))
    rollout_length  = int  (train_cfg.get("rollout_length",  256))
    minibatch_size  = int  (train_cfg.get("minibatch_size",  64))
    learning_rate   = float(train_cfg.get("learning_rate",   3.0e-4))
    clip_eps        = float(train_cfg.get("clip_eps",        0.2))
    value_coef      = float(train_cfg.get("value_coef",      0.5))
    entropy_coef    = float(train_cfg.get("entropy_coef",    0.01))
    gae_lambda      = float(train_cfg.get("gae_lambda",      0.95))
    gamma           = float(train_cfg.get("gamma",           0.99))
    ppo_epochs      = int  (train_cfg.get("ppo_epochs",      4))
    max_grad_norm   = float(train_cfg.get("max_grad_norm",   0.5))
    save_every_eps  = int  (train_cfg.get("save_every_episodes", 200))
    vec_mode        = str  (train_cfg.get("vector_env_mode", "async"))

    hidden_dim      = int(model_cfg.get("hidden_dim",     128))
    num_gat_layers  = int(model_cfg.get("num_gat_layers", 2))
    num_heads       = int(model_cfg.get("num_heads",      4))
    num_procs       = int(env_cfg  .get("num_nodes",      17))

    run_name        = log_cfg.get("run_name", "hgate_ppo_run")
    ckpt_dir        = Path(log_cfg.get(
        "checkpoint_dir", f"checkpoints/{run_name}"))
    tb_dir          = log_cfg.get("tb_log_dir", None)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, str(ckpt_dir / "resolved_config.yaml"))

    _seed_everything(seed_base)

    # ----- device -----
    device_str = str(train_cfg.get("device", "auto")).lower()
    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"⚠️  config requests {device_str!r} but CUDA is unavailable. "
                  f"Falling back to cpu.")
            device = torch.device("cpu")
        else:
            device = torch.device(device_str)

    print(f"[train_hgate_ppo] run_name={run_name!r}")
    print(f"[train_hgate_ppo] ckpt_dir={ckpt_dir}")
    print(f"[train_hgate_ppo] total_steps={total_steps:,}  num_envs={num_envs}  "
          f"rollout={rollout_length}  minibatch={minibatch_size}  "
          f"vec_mode={vec_mode}")
    print(f"[train_hgate_ppo] lr={learning_rate}  ppo_epochs={ppo_epochs}  "
          f"clip_eps={clip_eps}  gamma={gamma}  gae_lambda={gae_lambda}  "
          f"max_grad_norm={max_grad_norm}")
    print(f"[train_hgate_ppo] model: hidden={hidden_dim} gat_layers={num_gat_layers} "
          f"heads={num_heads}  num_procs={num_procs}")
    print(f"[train_hgate_ppo] device={device}")
    _reward_mode = (cfg.get("reward", {}) or {}).get("reward_mode",
                                                       "thermal_aware")
    print(f"[train_hgate_ppo] env reward_mode={_reward_mode!r}, "
          f"observation_keys={env_cfg.get('observation_keys', None)}")
    print(f"[train_hgate_ppo] algorithm={train_cfg.get('algorithm', 'ppo')!r}  "
          f"architecture={model_cfg.get('architecture', '?')!r}")

    # ----- env (multi-env per HK-4.5 Fix A) -----
    vec_env = make_vector_env(cfg, num_envs=num_envs,
                               seed_base=seed_base, mode=vec_mode)
    diag = smoke_test_vector_env(vec_env, num_envs=num_envs)
    print(f"[train_hgate_ppo] env smoke ok  "
          f"graph_obs_keys={sorted(diag['graph_obs_keys'])}")

    # ----- model + optimiser -----
    model = HGATEActorCritic(
        hidden_dim     = hidden_dim,
        num_procs      = num_procs,
        num_heads      = num_heads,
        num_gat_layers = num_gat_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ----- tensorboard -----
    writer = None
    if _TB_AVAILABLE and tb_dir:
        writer = SummaryWriter(log_dir=str(tb_dir))

    # ----- training state -----
    global_step  = 0
    episode_idx  = 0
    best_ep_ret  = -float("inf")
    t_start      = time.time()
    _obs, info   = vec_env.reset(seed=seed_base)
    episode_returns_buffer = np.zeros(num_envs, dtype=np.float64)

    # ----- main loop: collect rollout -> PPO update -> repeat -----
    while global_step < total_steps:
        ep_returns_collected: List[float] = []

        rollout, info, global_step = _collect_rollout(
            model=model,
            vec_env=vec_env,
            info_init=info,
            num_envs=num_envs,
            rollout_length=rollout_length,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
            global_step=global_step,
            total_steps=total_steps,
            episode_returns_buffer=episode_returns_buffer,
            ep_returns_out=ep_returns_collected,
        )
        episode_idx += len(ep_returns_collected)

        metrics = _ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=rollout,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        # ----- logging -----
        if writer is not None:
            writer.add_scalar("loss/policy",   metrics["loss_pg"],   global_step)
            writer.add_scalar("loss/value",    metrics["loss_v"],    global_step)
            writer.add_scalar("loss/entropy",  metrics["loss_ent"],  global_step)
            writer.add_scalar("policy/approx_kl", metrics["approx_kl"], global_step)
            writer.add_scalar("policy/clip_frac", metrics["clip_frac"], global_step)
            writer.add_scalar("policy/entropy",   metrics["entropy"],   global_step)
            if ep_returns_collected:
                writer.add_scalar("train/ep_return",
                                   float(np.mean(ep_returns_collected)),
                                   global_step)

        elapsed = time.time() - t_start
        sps = global_step / max(elapsed, 1e-6)
        mean_ret = (float(np.mean(ep_returns_collected))
                    if ep_returns_collected else float("nan"))
        print(f"[step {global_step:>10,}/{total_steps:,}]  "
              f"ep_done={len(ep_returns_collected)}  "
              f"mean_ep_ret={mean_ret:+8.2f}  "
              f"loss_pg={metrics['loss_pg']:+8.4f}  "
              f"loss_v={metrics['loss_v']:+8.4f}  "
              f"H={metrics['entropy']:6.3f}  "
              f"KL={metrics['approx_kl']:6.4f}  "
              f"clipfrac={metrics['clip_frac']:5.3f}  "
              f"{sps:6.0f} step/s  elapsed={elapsed/60:5.1f}min")

        # ----- checkpointing -----
        if ep_returns_collected:
            ep_ret = max(ep_returns_collected)
            if ep_ret > best_ep_ret:
                best_ep_ret = ep_ret
                _save_ckpt(ckpt_dir / "best.pt", model, optimizer,
                            global_step, ep_ret)
            if episode_idx % save_every_eps == 0:
                _save_ckpt(ckpt_dir / f"ckpt_ep_{episode_idx:06d}.pt",
                            model, optimizer, global_step, ep_ret)

    # final
    _save_ckpt(ckpt_dir / "final.pt", model, optimizer,
                global_step, best_ep_ret)
    if writer is not None:
        writer.close()
    print(f"[train_hgate_ppo] done — episodes={episode_idx}  "
          f"global_step={global_step:,}  best_ep_return={best_ep_ret:+.2f}")


def _save_ckpt(
    path:        Path,
    model:       HGATEActorCritic,
    optimizer:   torch.optim.Optimizer,
    global_step: int,
    ep_return:   float,
) -> None:
    torch.save({
        "model":           model.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "global_step":     int(global_step),
        "metrics_summary": {
            "ep_ret_mean":  float(ep_return),
        },
    }, str(path))


# =====================================================================
# CLI — usually invoked via train.py dispatch; standalone for testing
# =====================================================================
def main() -> None:                                     # pragma: no cover
    import argparse
    from cpo_thermal_v2.training.config_loader import (
        load_config, merge_cli_overrides,
    )
    parser = argparse.ArgumentParser(
        description="CPO v2 — HGATE-PPO trainer (standalone)",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)
    train_hgate_ppo(cfg)


if __name__ == "__main__":                              # pragma: no cover
    main()
