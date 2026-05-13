"""
training/train_decima_xattn.py — PPO entrypoint for D2 baseline
================================================================

D2 = Decima encoder (homog GCN) + cross-attention actor + PPO.  This
trainer is the HK-4.5 / HK-4.6 ``train_hgate_ppo`` loop with the model
import swapped to ``DecimaXAttnActorCritic`` — the rollout, GAE,
minibatched PPO update, and rolling-50 best.pt gate are byte-for-byte
identical (HGATE-PPO and D2 share a PPO recipe; only the model differs).

Why the explicit copy rather than parameter-passing into a shared
trainer
-----------------------------------------------------------------
``train_hgate_ppo.py`` type-hints HGATEActorCritic throughout for
clarity.  Generalising to a "ModelABC PPO trainer" would force the
HGATE / D2 / future-baseline interfaces to converge on an exact
shared shape (act_batched signature, evaluate_actions_batched
signature, etc.).  Maintaining that conformance is friction the
project hasn't earned yet — we have two baselines.  When we add a
third (or when the trainer logic itself starts drifting), the right
refactor is to extract the shared loop into ``ppo_loop_single_critic.py``.
For now: explicit copy, identical signatures, separate import.
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

from cpo_thermal_v2.baselines.decima_xattn import DecimaXAttnActorCritic
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


_D2_PROFILE = bool(int(os.environ.get("D2_PROFILE", "0")))


# =====================================================================
# Helpers
# =====================================================================
def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_graph_obs_list(info: Dict[str, Any], num_envs: int) -> List[Dict[str, Any]]:
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


# =====================================================================
# Rollout container + per-phase helpers
# =====================================================================
@dataclasses.dataclass
class RolloutBatch:
    """Flat rollout snapshot, length M = T_actual * num_envs."""
    graph_obs:     List[Dict[str, Any]]
    masks:         List[np.ndarray]
    actions:       np.ndarray
    old_log_probs: torch.Tensor
    advantages:    torch.Tensor
    returns:       torch.Tensor


def _collect_rollout(
    model:                  DecimaXAttnActorCritic,
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
    profile:                Optional[Dict[str, float]] = None,
) -> Tuple[RolloutBatch, Dict[str, Any], int]:
    info = info_init
    graph_obs_list = _extract_graph_obs_list(info, num_envs)
    action_masks   = _extract_action_masks  (info, num_envs)

    roll_graph_obs:    List[List[Dict[str, Any]]] = []
    roll_masks:        List[List[np.ndarray]]     = []
    roll_actions:      List[np.ndarray]           = []
    roll_log_probs:    List[torch.Tensor]         = []
    roll_values:       List[np.ndarray]           = []
    roll_rewards:      List[np.ndarray]           = []
    roll_dones:        List[np.ndarray]           = []

    model.eval()
    with torch.no_grad():
        for _t in range(rollout_length):
            if profile is not None:
                _t0 = time.perf_counter()
            batched = model.act_batched(
                graph_obs_list, action_masks, deterministic=False)
            actions_arr = batched["actions"].detach().cpu().numpy().astype(np.int64)
            log_probs_t = batched["log_probs"].detach()
            values_t    = batched["values"   ].detach().cpu().numpy().astype(np.float32)
            if profile is not None:
                profile["model_act_s"] += time.perf_counter() - _t0

            roll_graph_obs.append(graph_obs_list)
            roll_masks    .append(action_masks)
            roll_actions  .append(actions_arr.copy())
            roll_log_probs.append(log_probs_t)
            roll_values   .append(values_t)

            if profile is not None:
                _t0 = time.perf_counter()
            _obs, rewards_arr, term, trunc, info = vec_env.step(actions_arr)
            if profile is not None:
                profile["env_step_s"] += time.perf_counter() - _t0
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
            if profile is not None:
                _t0 = time.perf_counter()
            graph_obs_list = _extract_graph_obs_list(info, num_envs)
            action_masks   = _extract_action_masks  (info, num_envs)
            if profile is not None:
                profile["info_extract_s"] += time.perf_counter() - _t0

            if global_step >= total_steps:
                break

        T_actual = len(roll_actions)
        last_values = model.get_value_batched(
            graph_obs_list).detach().cpu().numpy().astype(np.float32)

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

    flat_graph_obs: List[Dict[str, Any]] = []
    flat_masks:     List[np.ndarray]     = []
    for t in range(T_actual):
        for n in range(num_envs):
            flat_graph_obs.append(roll_graph_obs[t][n])
            flat_masks    .append(roll_masks   [t][n])

    flat_actions    = np.stack(roll_actions, axis=0).reshape(-1)
    flat_log_probs  = torch.stack(roll_log_probs, dim=0).reshape(-1).to(device)
    adv_t = torch.from_numpy(advantages_arr).reshape(-1).to(device)
    ret_t = torch.from_numpy(returns_arr   ).reshape(-1).to(device)
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    rollout = RolloutBatch(
        graph_obs     = flat_graph_obs,
        masks         = flat_masks,
        actions       = flat_actions.astype(np.int64),
        old_log_probs = flat_log_probs,
        advantages    = adv_t,
        returns       = ret_t,
    )
    return rollout, info, global_step


def _ppo_update(
    model:          DecimaXAttnActorCritic,
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

            mb_graph_obs = [rollout.graph_obs[int(i)] for i in mb_indices]
            mb_masks     = [rollout.masks    [int(i)] for i in mb_indices]
            mb_actions   = rollout.actions[mb_indices]
            new_log_probs, entropies, values = model.evaluate_actions_batched(
                mb_graph_obs, mb_masks, mb_actions,
            )

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
def train_decima_xattn(cfg: Dict[str, Any]) -> None:
    """Multi-env minibatched PPO loop for DecimaXAttnActorCritic.

    Reads the same yaml schema as train_hgate_ppo with one rename:
      model.num_gcn_layers (D2)      <->   model.num_gat_layers (HGATE)
    Other yaml keys are identical so configs can be cross-compared.
    """
    train_cfg = cfg.get("training", {}) or {}
    log_cfg   = cfg.get("logging",  {}) or {}
    model_cfg = cfg.get("model",    {}) or {}
    env_cfg   = cfg.get("env",      {}) or {}

    seed_base       = int  (train_cfg.get("seed_base",       42))
    total_steps     = int  (train_cfg.get("total_steps",     5_000_000))
    num_envs        = int  (train_cfg.get("num_envs",        16))
    rollout_length  = int  (train_cfg.get("rollout_length",  256))
    minibatch_size  = int  (train_cfg.get("minibatch_size",  64))
    learning_rate   = float(train_cfg.get("learning_rate",   2.0e-4))
    clip_eps        = float(train_cfg.get("clip_eps",        0.2))
    value_coef      = float(train_cfg.get("value_coef",      0.5))
    entropy_coef    = float(train_cfg.get("entropy_coef",    0.01))
    gae_lambda      = float(train_cfg.get("gae_lambda",      0.98))
    gamma           = float(train_cfg.get("gamma",           0.99))
    ppo_epochs      = int  (train_cfg.get("ppo_epochs",      4))
    max_grad_norm   = float(train_cfg.get("max_grad_norm",   0.5))
    save_every_eps  = int  (train_cfg.get("save_every_episodes", 200))
    vec_mode        = str  (train_cfg.get("vector_env_mode", "async"))

    hidden_dim      = int(model_cfg.get("hidden_dim",     128))
    num_gcn_layers  = int(model_cfg.get("num_gcn_layers", 4))
    num_heads       = int(model_cfg.get("num_heads",      4))
    num_procs       = int(env_cfg  .get("num_nodes",      17))

    run_name        = log_cfg.get("run_name", "decima_xattn_run")
    ckpt_dir        = Path(log_cfg.get(
        "checkpoint_dir", f"checkpoints/{run_name}"))
    tb_dir          = log_cfg.get("tb_log_dir", None)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, str(ckpt_dir / "resolved_config.yaml"))

    _seed_everything(seed_base)

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

    print(f"[train_decima_xattn] run_name={run_name!r}")
    print(f"[train_decima_xattn] ckpt_dir={ckpt_dir}")
    print(f"[train_decima_xattn] total_steps={total_steps:,}  num_envs={num_envs}  "
          f"rollout={rollout_length}  minibatch={minibatch_size}  "
          f"vec_mode={vec_mode}")
    print(f"[train_decima_xattn] lr={learning_rate}  ppo_epochs={ppo_epochs}  "
          f"clip_eps={clip_eps}  gamma={gamma}  gae_lambda={gae_lambda}  "
          f"max_grad_norm={max_grad_norm}")
    print(f"[train_decima_xattn] model: hidden={hidden_dim} gcn_layers={num_gcn_layers} "
          f"heads={num_heads}  num_procs={num_procs}")
    print(f"[train_decima_xattn] device={device}")
    _reward_mode = (cfg.get("reward", {}) or {}).get("reward_mode",
                                                       "thermal_aware")
    print(f"[train_decima_xattn] env reward_mode={_reward_mode!r}, "
          f"observation_keys={env_cfg.get('observation_keys', None)}")

    vec_env = make_vector_env(cfg, num_envs=num_envs,
                               seed_base=seed_base, mode=vec_mode)
    diag = smoke_test_vector_env(vec_env, num_envs=num_envs)
    print(f"[train_decima_xattn] env smoke ok  "
          f"graph_obs_keys={sorted(diag['graph_obs_keys'])}")

    model = DecimaXAttnActorCritic(
        hidden_dim     = hidden_dim,
        num_procs      = num_procs,
        num_heads      = num_heads,
        num_gcn_layers = num_gcn_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = None
    if _TB_AVAILABLE and tb_dir:
        writer = SummaryWriter(log_dir=str(tb_dir))

    global_step  = 0
    episode_idx  = 0
    best_ep_ret  = -float("inf")
    t_start      = time.time()
    _obs, info   = vec_env.reset(seed=seed_base)
    episode_returns_buffer = np.zeros(num_envs, dtype=np.float64)

    rolling_ep_returns: List[float] = []
    _BEST_CKPT_WINDOW = 50

    if _D2_PROFILE:
        print("[train_decima_xattn] D2_PROFILE=1 — per-phase timing enabled")

    while global_step < total_steps:
        ep_returns_collected: List[float] = []

        profile: Optional[Dict[str, float]] = None
        if _D2_PROFILE:
            profile = {"model_act_s": 0.0, "env_step_s": 0.0,
                        "info_extract_s": 0.0}
            _t_roll_start = time.perf_counter()

        rollout, info, global_step = _collect_rollout(
            model=model, vec_env=vec_env, info_init=info,
            num_envs=num_envs, rollout_length=rollout_length,
            gamma=gamma, gae_lambda=gae_lambda, device=device,
            global_step=global_step, total_steps=total_steps,
            episode_returns_buffer=episode_returns_buffer,
            ep_returns_out=ep_returns_collected,
            profile=profile,
        )
        episode_idx += len(ep_returns_collected)

        if _D2_PROFILE:
            _rollout_total_s = time.perf_counter() - _t_roll_start
            _t_upd_start = time.perf_counter()

        metrics = _ppo_update(
            model=model, optimizer=optimizer, rollout=rollout,
            ppo_epochs=ppo_epochs, minibatch_size=minibatch_size,
            clip_eps=clip_eps, value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm, device=device,
        )

        if _D2_PROFILE:
            _update_total_s = time.perf_counter() - _t_upd_start

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
        if _D2_PROFILE and profile is not None:
            _other_roll_s = max(0.0, _rollout_total_s - (
                profile["model_act_s"] + profile["env_step_s"]
                + profile["info_extract_s"]))
            _phase_total = _rollout_total_s + _update_total_s
            def _pct(x):
                return (100.0 * x / _phase_total) if _phase_total > 0 else 0.0
            print(f"  [profile] rollout={_rollout_total_s*1000:7.0f}ms "
                  f"(model_act={profile['model_act_s']*1000:5.0f}ms / "
                  f"{_pct(profile['model_act_s']):4.1f}%, "
                  f"env_step={profile['env_step_s']*1000:5.0f}ms / "
                  f"{_pct(profile['env_step_s']):4.1f}%, "
                  f"info_extract={profile['info_extract_s']*1000:5.0f}ms / "
                  f"{_pct(profile['info_extract_s']):4.1f}%, "
                  f"other={_other_roll_s*1000:4.0f}ms)  "
                  f"update={_update_total_s*1000:6.0f}ms / "
                  f"{_pct(_update_total_s):4.1f}%")

        if ep_returns_collected:
            rolling_ep_returns.extend(ep_returns_collected)
            new_best, should_save = _update_best_ckpt(
                rolling_returns=rolling_ep_returns,
                best_so_far=best_ep_ret,
                window=_BEST_CKPT_WINDOW,
            )
            if should_save:
                best_ep_ret = new_best  # type: ignore[assignment]
                _save_ckpt(ckpt_dir / "best.pt", model, optimizer,
                            global_step, best_ep_ret)
                print(f"  [save] best.pt updated: "
                      f"moving_avg({_BEST_CKPT_WINDOW} eps)={best_ep_ret:+.2f} "
                      f"at episode {episode_idx}, step {global_step:,}")

            if episode_idx % save_every_eps == 0:
                _save_ckpt(ckpt_dir / f"ckpt_ep_{episode_idx:06d}.pt",
                            model, optimizer, global_step,
                            float(np.mean(ep_returns_collected)))

    _save_ckpt(ckpt_dir / "final.pt", model, optimizer,
                global_step, best_ep_ret)
    if writer is not None:
        writer.close()
    print(f"[train_decima_xattn] done — episodes={episode_idx}  "
          f"global_step={global_step:,}  best_ep_return={best_ep_ret:+.2f}")


def _update_best_ckpt(
    rolling_returns: List[float],
    best_so_far:     float,
    window:          int = 50,
) -> Tuple[Optional[float], bool]:
    """Rolling-mean best.pt gate (HK-4.6 fix, see train_hgate_ppo.py)."""
    if len(rolling_returns) < window:
        return None, False
    moving_avg = float(np.mean(rolling_returns[-window:]))
    if moving_avg > best_so_far:
        return moving_avg, True
    return None, False


def _save_ckpt(
    path:        Path,
    model:       DecimaXAttnActorCritic,
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


def main() -> None:                                     # pragma: no cover
    import argparse
    from cpo_thermal_v2.training.config_loader import (
        load_config, merge_cli_overrides,
    )
    parser = argparse.ArgumentParser(
        description="CPO v2 — D2 (Decima encoder + cross-attention actor) trainer",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)
    train_decima_xattn(cfg)


if __name__ == "__main__":                              # pragma: no cover
    main()
