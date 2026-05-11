"""
training/train_decima_true.py — REINFORCE entrypoint for Decima true
====================================================================

Standalone Mao-style trainer.  Invoked via
``train.py``'s ``algorithm: reinforce`` dispatch.

Design notes
------------
* Single-env (SyncVectorEnv, num_envs=1) rollout — REINFORCE updates
  per-episode anyway, so multi-env asynchrony adds no throughput
  benefit and complicates per-episode log-prob bookkeeping.
* One gradient step per episode (no minibatching, no GAE, no clipping).
* Moving-average baseline of length ``baseline_window``.
* Best-by-validation-ep_return checkpoint at ``checkpoints/<run>/best.pt``;
  full final ckpt at ``checkpoints/<run>/final.pt``; periodic snapshots
  every ``save_every_episodes`` episodes.
* TensorBoard scalars under ``logging.tb_log_dir`` (if set).

This module deliberately does not share code with ppo_trainer.py; the
two algorithms have incompatible update semantics and the duplication
keeps each implementation auditable in isolation.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cpo_thermal_v2.baselines.decima_true import (
    DecimaTruePolicy, DecimaTrueAgent,
)
from cpo_thermal_v2.training.config_loader import save_resolved_config
from cpo_thermal_v2.training.env_factory   import (
    make_vector_env, smoke_test_vector_env,
    _extract_per_env,                 # robust gymnasium info unwrap
)

# tensorboard import is lazy — sandbox systems may lack it
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
    """Pull the per-env graph_obs dict out of a vectorised info.

    Matches the unwrap pattern used by env_factory.smoke_test_vector_env:
    _extract_per_env handles all known gymnasium info-broadcast formats,
    then the env's 1-element object-array wrapper is unwrapped to the
    underlying dict.
    """
    go_list = _extract_per_env(info, "graph_obs", num_envs=1)
    g = go_list[0]
    if isinstance(g, np.ndarray) and g.dtype == object and g.size == 1:
        g = g[0]
    if not isinstance(g, dict):
        raise RuntimeError(
            f"_extract_graph_obs: expected dict, got {type(g).__name__}"
        )
    return g


def _extract_action_mask(info: Dict[str, Any]) -> np.ndarray:
    masks = _extract_per_env(info, "action_mask", num_envs=1)
    m = masks[0]
    return np.asarray(m, dtype=bool)


# =====================================================================
# Main entrypoint
# =====================================================================
def train_reinforce(cfg: Dict[str, Any]) -> None:
    """REINFORCE training loop for Decima true.

    Reads the same yaml schema as the PPO trainer, plus these training/*
    keys:

      algorithm:        "reinforce"  (dispatch from train.py)
      baseline_window:  int   (default 50)
      learning_rate:    float (default 5e-4)
      total_steps:      int   (default 5_000_000)
      entropy_coef:     float (default 0.0)
      save_every_episodes: int (default 200)
      seed_base:        int   (default 42)

    And under model/:
      hidden_dim:       int (default 256)
      num_gcn_layers:   int (default 4)
    """
    train_cfg = cfg.get("training", {}) or {}
    log_cfg   = cfg.get("logging",  {}) or {}
    model_cfg = cfg.get("model",    {}) or {}
    env_cfg   = cfg.get("env",      {}) or {}

    # ----- hyper-params -----
    seed_base       = int(train_cfg.get("seed_base", 42))
    total_steps     = int(train_cfg.get("total_steps", 5_000_000))
    learning_rate   = float(train_cfg.get("learning_rate", 5.0e-4))
    baseline_window = int(train_cfg.get("baseline_window", 50))
    entropy_coef    = float(train_cfg.get("entropy_coef", 0.0))
    gamma           = float(train_cfg.get("gamma", 0.99))
    save_every_eps  = int(train_cfg.get("save_every_episodes", 200))

    hidden_dim      = int(model_cfg.get("hidden_dim", 256))
    num_gcn_layers  = int(model_cfg.get("num_gcn_layers", 4))

    # Resolve device — mirrors train.py:226-240 pattern.  "auto" picks
    # cuda if available, else cpu.  Explicit values ("cuda:0" / "cpu" /
    # "cuda:N") are honoured as-is.  Without this resolution the policy
    # would stay on cpu forever (HK-3.1.1 bug: V100 ran with 0%
    # GPU-Util because train_reinforce had zero device handling).
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

    run_name        = log_cfg.get("run_name", "decima_true_run")
    ckpt_dir        = Path(log_cfg.get(
        "checkpoint_dir", f"checkpoints/{run_name}"))
    tb_dir          = log_cfg.get("tb_log_dir", None)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config alongside ckpts
    save_resolved_config(cfg, str(ckpt_dir / "resolved_config.yaml"))

    _seed_everything(seed_base)

    print(f"[train_decima_true] run_name={run_name!r}")
    print(f"[train_decima_true] ckpt_dir={ckpt_dir}")
    print(f"[train_decima_true] total_steps={total_steps:,}  lr={learning_rate}  "
          f"baseline_window={baseline_window}  gamma={gamma}  "
          f"entropy_coef={entropy_coef}")
    print(f"[train_decima_true] model: hidden={hidden_dim} gcn_layers={num_gcn_layers}")
    print(f"[train_decima_true] device={device}")
    # Resolve reward_mode: env_factory builds RewardConfig from the top-
    # level `reward:` section, so we look there (not under env:).
    _reward_mode = (cfg.get("reward", {}) or {}).get(
        "reward_mode", "thermal_aware")
    print(f"[train_decima_true] env reward_mode={_reward_mode!r}, "
          f"observation_keys={env_cfg.get('observation_keys', None)}")
    print(f"[train_decima_true] algorithm={train_cfg.get('algorithm', 'ppo')!r}")

    # ----- env (single sync worker — REINFORCE updates per-episode) -----
    vec_env = make_vector_env(cfg, num_envs=1, seed_base=seed_base, mode="sync")
    diag = smoke_test_vector_env(vec_env, num_envs=1)
    print(f"[train_decima_true] env smoke ok  "
          f"graph_obs_keys={sorted(diag['graph_obs_keys'])}")

    # ----- policy + agent -----
    policy = DecimaTruePolicy(
        hidden_dim     = hidden_dim,
        num_gcn_layers = num_gcn_layers,
    ).to(device)
    agent = DecimaTrueAgent(
        policy,
        lr              = learning_rate,
        gamma           = gamma,
        baseline_window = baseline_window,
        entropy_coef    = entropy_coef,
    )

    # ----- tensorboard -----
    writer = None
    if _TB_AVAILABLE and tb_dir:
        writer = SummaryWriter(log_dir=str(tb_dir))

    # ----- main loop -----
    global_step  = 0
    episode_idx  = 0
    best_ep_ret  = -float("inf")
    t_start      = time.time()
    obs, info    = vec_env.reset(seed=seed_base)

    while global_step < total_steps:
        # ----- collect one episode -----
        episode_log_probs: List[torch.Tensor] = []
        episode_entropies: List[torch.Tensor] = []
        episode_rewards:   List[float]        = []
        done = False
        ep_steps = 0
        while not done:
            g_obs = _extract_graph_obs(info)
            a_mask = _extract_action_mask(info)
            action, log_p, ent = policy.select_action(
                g_obs, a_mask, deterministic=False)
            actions = np.asarray([action], dtype=np.int64)

            obs, reward, terminated, truncated, info = vec_env.step(actions)
            r = float(np.asarray(reward).reshape(-1)[0])
            done = bool(
                np.asarray(terminated).reshape(-1)[0] or
                np.asarray(truncated).reshape(-1)[0]
            )

            episode_log_probs.append(log_p)
            episode_entropies.append(ent)
            episode_rewards.append(r)

            global_step += 1
            ep_steps    += 1
            if global_step >= total_steps:
                break

        # ----- one REINFORCE step at end of episode -----
        metrics = agent.update(
            episode_log_probs, episode_rewards,
            entropies=episode_entropies if entropy_coef > 0 else None,
        )
        episode_idx += 1
        ep_ret = float(np.sum(episode_rewards))

        if writer is not None:
            writer.add_scalar("train/loss",       metrics["loss"], global_step)
            writer.add_scalar("train/baseline",   metrics["baseline"], global_step)
            writer.add_scalar("train/ep_return",  ep_ret,           global_step)
            writer.add_scalar("train/ep_steps",   ep_steps,         global_step)

        # Print every episode for the first 5 (so short smoke runs see
        # something), then every 10th episode for steady-state.
        if episode_idx <= 5 or episode_idx % 10 == 0:
            elapsed = time.time() - t_start
            sps = global_step / max(elapsed, 1e-6)
            print(f"[ep {episode_idx:>6d}] step {global_step:>10,}/{total_steps:,}  "
                  f"ep_ret={ep_ret:+8.2f}  baseline={metrics['baseline']:+8.2f}  "
                  f"loss={metrics['loss']:+8.4f}  ep_steps={ep_steps:>4d}  "
                  f"{sps:6.0f} step/s  elapsed={elapsed/60:5.1f}min")

        # ----- checkpointing -----
        if ep_ret > best_ep_ret:
            best_ep_ret = ep_ret
            _save_ckpt(ckpt_dir / "best.pt", policy, agent,
                       global_step, ep_ret)
        if episode_idx % save_every_eps == 0:
            _save_ckpt(ckpt_dir / f"ckpt_ep_{episode_idx:06d}.pt",
                       policy, agent, global_step, ep_ret)

        # ----- reset env for next episode -----
        if not done:
            # exited inner loop due to total_steps cap, don't reset
            break
        obs, info = vec_env.reset()

    # final ckpt
    _save_ckpt(ckpt_dir / "final.pt", policy, agent,
               global_step, best_ep_ret)
    if writer is not None:
        writer.close()
    print(f"[train_decima_true] done — episodes={episode_idx}  "
          f"global_step={global_step:,}  best_ep_return={best_ep_ret:+.2f}")


def _save_ckpt(
    path:        Path,
    policy:      DecimaTruePolicy,
    agent:       DecimaTrueAgent,
    global_step: int,
    ep_return:   float,
) -> None:
    torch.save({
        "model":           policy.state_dict(),
        "optimizer":       agent.optimizer.state_dict(),
        "global_step":     int(global_step),
        "metrics_summary": {
            "ep_ret_mean": float(ep_return),
            "baseline":    float(agent.baseline),
        },
    }, str(path))


# =====================================================================
# CLI — usually invoked via train.py's dispatch; standalone for
# direct testing.
# =====================================================================
def main() -> None:                                     # pragma: no cover
    import argparse
    from cpo_thermal_v2.training.config_loader import (
        load_config, merge_cli_overrides,
    )
    parser = argparse.ArgumentParser(
        description="CPO v2 — Decima true REINFORCE trainer (standalone)",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)
    train_reinforce(cfg)


if __name__ == "__main__":                              # pragma: no cover
    main()
