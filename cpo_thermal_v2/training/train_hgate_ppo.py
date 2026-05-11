"""
training/train_hgate_ppo.py — PPO entrypoint for HGATE-PPO baseline
====================================================================

Standalone single-env PPO trainer.  Invoked via train.py's
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
inner loop (invasive).  The cleaner separation is to write a small
single-env PPO loop here.

What this trainer covers
- 1-env SyncVectorEnv rollouts (no batching headache; suitable for
  the smoke + initial V100 pilot)
- Standard PPO update: clipped surrogate loss + value MSE +
  entropy bonus, GAE(λ) advantages
- Per-episode logging, best/final ckpt persistence, tb scalars

What it does NOT cover (deferred to a future HK-4.x):
- Multi-env asynchronous rollouts (HGATE has no env-side mutable
  state shared between envs, so this is mostly a perf concern)
- Reward channel decomposition (HGATE is single-critic; the
  env's reward_placement / reward_delay info entries are summed
  into a single scalar reward channel for the update)
"""
from __future__ import annotations

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


def _extract_graph_obs(info: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the per-env graph_obs dict out of a vectorised info."""
    go_list = _extract_per_env(info, "graph_obs", num_envs=1)
    g = go_list[0]
    if isinstance(g, np.ndarray) and g.dtype == object and g.size == 1:
        g = g[0]
    if not isinstance(g, dict):
        raise RuntimeError(
            f"_extract_graph_obs: expected dict, got {type(g).__name__}"
        )
    return g


def _extract_mask(info: Dict[str, Any]) -> np.ndarray:
    masks = _extract_per_env(info, "action_mask", num_envs=1)
    return np.asarray(masks[0], dtype=bool)


def _compute_gae(
    rewards:    List[float],
    values:     List[float],
    dones:      List[bool],
    last_value: float,
    gamma:      float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages + returns for one rollout.

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
    the current policy.  Used by PPO update to compute the importance
    ratio without sampling a new action."""
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
# Main entrypoint
# =====================================================================
def train_hgate_ppo(cfg: Dict[str, Any]) -> None:
    """Single-env PPO loop for HGATEActorCritic.

    Reads the same yaml schema as the dual-channel PPO trainer; only
    the keys actually used here are described below.

      training:
        algorithm:       ppo                   # filter, default
        device:          auto | cuda | cpu
        total_steps:     int
        rollout_length:  int                   # default 256
        learning_rate:   float                 # default 3e-4
        clip_eps:        float                 # default 0.2
        value_coef:      float                 # default 0.5
        entropy_coef:    float                 # default 0.01
        gae_lambda:      float                 # default 0.95
        gamma:           float                 # default 0.99
        ppo_epochs:      int                   # default 4
        save_every_episodes: int               # default 200
        seed_base:       int                   # default 42

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
    seed_base       = int(train_cfg.get("seed_base", 42))
    total_steps     = int(train_cfg.get("total_steps", 5_000_000))
    rollout_length  = int(train_cfg.get("rollout_length", 256))
    learning_rate   = float(train_cfg.get("learning_rate", 3.0e-4))
    clip_eps        = float(train_cfg.get("clip_eps", 0.2))
    value_coef      = float(train_cfg.get("value_coef", 0.5))
    entropy_coef    = float(train_cfg.get("entropy_coef", 0.01))
    gae_lambda      = float(train_cfg.get("gae_lambda", 0.95))
    gamma           = float(train_cfg.get("gamma", 0.99))
    ppo_epochs      = int(train_cfg.get("ppo_epochs", 4))
    save_every_eps  = int(train_cfg.get("save_every_episodes", 200))

    hidden_dim      = int(model_cfg.get("hidden_dim", 128))
    num_gat_layers  = int(model_cfg.get("num_gat_layers", 2))
    num_heads       = int(model_cfg.get("num_heads", 4))
    num_procs       = int(env_cfg.get("num_nodes", 17))

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
    print(f"[train_hgate_ppo] total_steps={total_steps:,}  rollout={rollout_length}  "
          f"lr={learning_rate}  ppo_epochs={ppo_epochs}  clip_eps={clip_eps}  "
          f"gamma={gamma} gae_lambda={gae_lambda}")
    print(f"[train_hgate_ppo] model: hidden={hidden_dim} gat_layers={num_gat_layers} "
          f"heads={num_heads}  num_procs={num_procs}")
    print(f"[train_hgate_ppo] device={device}")
    _reward_mode = (cfg.get("reward", {}) or {}).get("reward_mode",
                                                       "thermal_aware")
    print(f"[train_hgate_ppo] env reward_mode={_reward_mode!r}, "
          f"observation_keys={env_cfg.get('observation_keys', None)}")
    print(f"[train_hgate_ppo] algorithm={train_cfg.get('algorithm', 'ppo')!r}  "
          f"architecture={model_cfg.get('architecture', '?')!r}")

    # ----- env (single sync worker for the smoke; multi-env left for
    # a future HK if perf becomes the bottleneck) -----
    vec_env = make_vector_env(cfg, num_envs=1, seed_base=seed_base, mode="sync")
    diag = smoke_test_vector_env(vec_env, num_envs=1)
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
    obs, info    = vec_env.reset(seed=seed_base)
    ep_ret_accum = 0.0

    # ----- main loop: collect rollout -> PPO update -> repeat -----
    while global_step < total_steps:
        # ----- collect a rollout -----
        roll_graph_obs:   List[Dict[str, Any]]  = []
        roll_masks:       List[np.ndarray]      = []
        roll_actions:     List[int]             = []
        roll_log_probs:   List[torch.Tensor]    = []
        roll_values:      List[float]           = []
        roll_rewards:     List[float]           = []
        roll_dones:       List[bool]            = []
        ep_returns_collected: List[float]       = []

        model.eval()    # rollout under no-grad sampling
        with torch.no_grad():
            for _ in range(rollout_length):
                g_obs = _extract_graph_obs(info)
                mask  = _extract_mask(info)

                out = model.act(g_obs, mask, deterministic=False)
                action_int = int(out["action"])
                actions = np.asarray([action_int], dtype=np.int64)

                roll_graph_obs.append(g_obs)
                roll_masks.append(mask)
                roll_actions.append(action_int)
                roll_log_probs.append(out["log_prob"].detach())
                roll_values.append(float(out["value"].item()))

                obs, reward, term, trunc, info = vec_env.step(actions)
                r = float(np.asarray(reward).reshape(-1)[0])
                done = bool(
                    np.asarray(term).reshape(-1)[0] or
                    np.asarray(trunc).reshape(-1)[0]
                )
                roll_rewards.append(r)
                roll_dones.append(done)

                global_step  += 1
                ep_ret_accum += r
                if done:
                    episode_idx += 1
                    ep_returns_collected.append(ep_ret_accum)
                    ep_ret_accum = 0.0
                    obs, info = vec_env.reset()

                if global_step >= total_steps:
                    break

            # bootstrap V for the final state (handles partial episode at
            # rollout end)
            last_g_obs = _extract_graph_obs(info)
            last_value = float(model.get_value(last_g_obs).item())

        # ----- compute GAE advantages + returns -----
        advantages, returns = _compute_gae(
            roll_rewards, roll_values, roll_dones,
            last_value=last_value, gamma=gamma, gae_lambda=gae_lambda,
        )
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(returns,    dtype=torch.float32, device=device)
        # Standard PPO normalisation of advantages (per rollout)
        if adv_tensor.numel() > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        old_log_probs = torch.stack(roll_log_probs)        # (T,)

        # ----- PPO update -----
        model.train()
        loss_pg_acc, loss_v_acc, loss_ent_acc = 0.0, 0.0, 0.0
        for _epoch in range(ppo_epochs):
            for t in range(len(roll_actions)):
                new_log_prob, entropy, value = _evaluate_action(
                    model,
                    roll_graph_obs[t],
                    roll_masks[t],
                    roll_actions[t],
                )
                ratio = (new_log_prob - old_log_probs[t]).exp()
                pg1 = ratio * adv_tensor[t]
                pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) \
                          * adv_tensor[t]
                loss_pg  = -torch.min(pg1, pg2)
                loss_v   = F.mse_loss(value, ret_tensor[t])
                loss_ent = -entropy * entropy_coef
                loss = loss_pg + value_coef * loss_v + loss_ent

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                loss_pg_acc  += float(loss_pg.item())
                loss_v_acc   += float(loss_v.item())
                loss_ent_acc += float(loss_ent.item())

        T_steps = max(1, ppo_epochs * len(roll_actions))
        loss_pg_mean  = loss_pg_acc  / T_steps
        loss_v_mean   = loss_v_acc   / T_steps
        loss_ent_mean = loss_ent_acc / T_steps

        # ----- logging -----
        if writer is not None:
            writer.add_scalar("loss/policy",  loss_pg_mean,  global_step)
            writer.add_scalar("loss/value",   loss_v_mean,   global_step)
            writer.add_scalar("loss/entropy", loss_ent_mean, global_step)
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
              f"loss_pg={loss_pg_mean:+8.4f}  loss_v={loss_v_mean:+8.4f}  "
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
