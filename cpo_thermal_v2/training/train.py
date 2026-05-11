"""
training/train.py — Main PPO training entrypoint
================================================

Reads a YAML config (with optional CLI overrides), constructs an
AsyncVectorEnv + PPOActorCritic, and runs the rollout/update loop until
``training.total_steps`` env-steps are collected.

Invocation
----------
::

    python -m cpo_thermal_v2.training.train --config configs/stage1_auto_only.yaml

    # Stage 2 (warm-start path is read from the yaml):
    python -m cpo_thermal_v2.training.train --config configs/stage2_hybrid.yaml

    # CLI override examples:
    python -m cpo_thermal_v2.training.train \
        --config configs/stage1_auto_only.yaml \
        --override training.learning_rate=1e-4 \
        --override env.num_nodes=33

Output layout (one run)
-----------------------
::

    <checkpoint_dir>/
    ├── resolved_config.yaml      # the fully-resolved config (inherit + CLI)
    ├── ckpt_step_50000.pt        # periodic checkpoints
    ├── ckpt_step_100000.pt
    ├── ...
    ├── best.pt                   # symlink-or-copy to the best-eval checkpoint
    └── tb_logs/                  # tensorboard event files (under tb_log_dir)

Defensive startup smoke
-----------------------
Before entering the main loop, ``train()`` runs a 1-step rollout + 1-step
PPO update.  Any shape/dtype/PyG bug that would silently crash hours later
crashes here within seconds.  This is the user's "don't lose 24h of A100
time on a one-line bug" insurance policy.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cpo_thermal_v2.models import PPOActorCritic, build_batch
from cpo_thermal_v2.training.config_loader   import (
    load_config, merge_cli_overrides, save_resolved_config,
)
from cpo_thermal_v2.training.curriculum      import CurriculumScheduler
from cpo_thermal_v2.training.env_factory     import (
    make_vector_env, broadcast_curriculum_stage, smoke_test_vector_env,
)
from cpo_thermal_v2.training.gae             import compute_gae
from cpo_thermal_v2.training.ppo_trainer     import (
    PPOTrainer, make_optimizer, make_lr_scheduler, PPOUpdateMetrics,
)
from cpo_thermal_v2.training.rollout_buffer  import (
    RolloutBuffer, RewardNormaliser,
)


# =====================================================================
# Helpers
# =====================================================================
def _seed_everything(seed: int) -> None:
    """Set RNG seeds for numpy / torch / torch.cuda for reproducibility.

    Note: AsyncVectorEnv workers seed themselves via ``seed_base + worker_idx``
    in ``env_factory.make_vector_env``; their seeds are independent of this.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_warm_start(
    model:    PPOActorCritic,
    ckpt_path: str,
    strict:   bool,
) -> Tuple[List[str], List[str]]:
    """Load a checkpoint into ``model`` with logging of mismatches.

    Returns ``(missing, unexpected)`` from ``load_state_dict``.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"warm-start checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Accept either a raw state_dict or a wrapped dict containing 'model'
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[warm-start] loaded {ckpt_path}")
    if missing:
        print(f"  missing keys (will start fresh): {len(missing)}")
        for k in missing[:10]:
            print(f"    - {k}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    if unexpected:
        print(f"  unexpected keys (ignored): {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"    - {k}")
    return list(missing), list(unexpected)


def _save_checkpoint(
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    metrics_summary: Dict[str, float],
    out_path: str,
) -> None:
    """Save a self-contained checkpoint."""
    torch.save({
        "model":           model.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "global_step":     global_step,
        "metrics_summary": metrics_summary,
    }, out_path)


def _gather_info_array(info: Dict[str, Any], key: str, num_envs: int) -> np.ndarray:
    """Gymnasium AsyncVectorEnv broadcasts info as ``info[key]`` arrays of
    length ``num_envs``.  This helper extracts a (N,) numpy view.
    """
    raw = info.get(key)
    if raw is None:
        return np.zeros(num_envs)
    if isinstance(raw, np.ndarray):
        return raw
    return np.asarray(raw)


def _extract_reward_channels(
    info: Dict[str, Any],
    num_envs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pull placement and delay reward channels out of the vector-env info.

    Returns ``(rewards_p, rewards_d)`` each shape ``(N,)``.

    The env emits these as **flat keys** ``reward_placement`` /
    ``reward_delay`` (rather than a nested ``reward_channels`` dict)
    specifically to dodge a gymnasium 1.x vector-info bug — see the
    comment in ``cpo_thermal_env.py:_make_info``.
    """
    from cpo_thermal_v2.training.env_factory import _extract_per_env

    rp_list = _extract_per_env(info, "reward_placement", num_envs)
    rd_list = _extract_per_env(info, "reward_delay",     num_envs)

    rp = np.zeros(num_envs, dtype=np.float32)
    rd = np.zeros(num_envs, dtype=np.float32)
    for n in range(num_envs):
        if rp_list[n] is not None:
            rp[n] = float(rp_list[n])
        if rd_list[n] is not None:
            rd[n] = float(rd_list[n])
    return rp, rd


def _extract_graph_obs_list(info: Dict[str, Any], num_envs: int) -> List[Dict]:
    """Pull per-env ``graph_obs`` dicts out of vector-env info, unwrapping
    the 1-element object-array container the env uses for pickle safety.
    """
    from cpo_thermal_v2.training.env_factory import _extract_per_env
    raw_list = _extract_per_env(info, "graph_obs", num_envs)

    out: List[Dict] = []
    for n, item in enumerate(raw_list):
        # Unwrap 1-element object array if present
        if isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
            item = item[0]
        if not isinstance(item, dict):
            raise TypeError(
                f"graph_obs[{n}] has unexpected type "
                f"{type(item).__name__}; the env emits a dict wrapped in a "
                f"1-element object array.  Got: {item!r}"
            )
        out.append(item)
    return out


def _extract_action_masks(info: Dict[str, Any], num_envs: int) -> List[np.ndarray]:
    """Pull per-env action_masks out of the vector-env info."""
    from cpo_thermal_v2.training.env_factory import _extract_per_env
    raw_list = _extract_per_env(info, "action_mask", num_envs)
    out: List[np.ndarray] = []
    for n, m in enumerate(raw_list):
        if m is None:
            raise RuntimeError(
                f"env {n} did not emit an action_mask in info; this should "
                f"never happen for CPOThermalDAGEnvV2.  Check the env wiring."
            )
        out.append(np.asarray(m, dtype=bool))
    return out


# =====================================================================
# Main training function
# =====================================================================
def train(config: Dict[str, Any]) -> None:
    """Run one full training session per the resolved config."""
    # ---------------- Setup ----------------
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    train_cfg  = config["training"]
    log_cfg    = config["logging"]
    num_envs   = int(train_cfg["num_envs"])
    rollout_T  = int(train_cfg["rollout_length"])
    total_steps = int(train_cfg["total_steps"])

    # Resolve device: "auto" picks cuda if available, else cpu.
    # Explicit "cuda:0" / "cpu" / "cuda:N" are honoured as-is.
    device_str = str(train_cfg.get("device", "auto")).lower()
    if device_str == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"⚠️  config requests {device!r} but CUDA is unavailable. "
                  f"Falling back to cpu.")
            device = "cpu"

    print(f"=== CPO v2 Training ===")
    print(f"  config.run_name      = {log_cfg.get('run_name', '?')}")
    print(f"  device               = {device}")
    print(f"  num_envs             = {num_envs}")
    print(f"  rollout_length       = {rollout_T}")
    print(f"  total_steps          = {total_steps:,}")
    print(f"  action_mode          = {config['env']['action_mode']}")
    print(f"  num_nodes            = {config['env']['num_nodes']}")

    # ---------------- Output directories ----------------
    ckpt_dir = log_cfg["checkpoint_dir"]
    tb_dir   = log_cfg["tb_log_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    # Snapshot the resolved config alongside the checkpoint for reproducibility
    save_resolved_config(config, os.path.join(ckpt_dir, "resolved_config.yaml"))
    print(f"  config snapshot → {ckpt_dir}/resolved_config.yaml")

    writer = SummaryWriter(log_dir=tb_dir)

    # Optional wandb (silent if not configured / installed)
    wandb_run = None
    if log_cfg.get("use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=log_cfg.get("wandb_project", "cpo-v2"),
                name=log_cfg.get("run_name"),
                config=config,
            )
            print(f"  wandb run → {wandb_run.url}")
        except ImportError:
            print("  wandb not installed; skipping (this is fine).")

    # ---------------- Vector env ----------------
    vec_mode = train_cfg.get("vector_env_mode", "async")
    print(f"  vector env mode      = {vec_mode}")
    vec_env = make_vector_env(config, num_envs=num_envs,
                               seed_base=seed, mode=vec_mode)

    # ---------------- Curriculum ----------------
    curriculum = CurriculumScheduler(config["curriculum"])
    if curriculum.enabled:
        broadcast_curriculum_stage(vec_env, **curriculum.current_kwargs)
        print(f"  curriculum start: {curriculum.current.name!r} "
              f"({curriculum.current.initial_temp_range}, "
              f"max_dag_size={curriculum.current.max_dag_size})")

    # ---------------- Model + optimiser ----------------
    model_cfg = config["model"]
    model = PPOActorCritic(
        action_mode = config["env"]["action_mode"],
        K_delay     = config["env"]["K_delay"],
        # Encoder input dims — defaulting to the full thermal-aware
        # feature set; can be reduced in train_decima_fair config to
        # remove thermal info.
        task_in_dim  = model_cfg.get("task_in_dim",  8),
        proc_in_dim  = model_cfg.get("proc_in_dim",  7),
        edge_dim_t2p = model_cfg.get("edge_dim_t2p", 2),
        hidden      = model_cfg["hidden"],
        num_layers  = model_cfg["num_layers"],
        num_heads   = model_cfg["num_heads"],
        dropout     = model_cfg["dropout"],
        critic_hidden = model_cfg["critic_hidden"],
    ).to(device)

    # Thermal-blind flag: if True, build_batch strips proc_x cols 0:4
    # and edges_t2p_attr col 1 before constructing the PyG Batch.  This
    # is THE strip path for fair Decima training — env-side wrapping
    # was unreliable across AsyncVectorEnv subprocess boundaries.
    thermal_blind = bool(config.get("env", {}).get("thermal_blind", False))
    if thermal_blind:
        print(f"[train] thermal_blind=True — build_batch will strip "
              f"thermal features from every graph_obs (proc_x: 7→3, "
              f"edges_t2p_attr: 2→1)")

    optimizer = make_optimizer(model, train_cfg)

    # Warm-start (Stage 2 path)
    warm_start_path = train_cfg.get("warm_start_path")
    if warm_start_path:
        _load_warm_start(
            model, warm_start_path,
            strict=bool(train_cfg.get("warm_start_strict", False)),
        )

    # ---------------- LR schedule ----------------
    # Total updates ≈ total_steps / (rollout_T * num_envs), times ppo_epochs
    # × num_minibatches.  The LR scheduler steps once per *PPO update phase*,
    # not per minibatch — match Schulman PPO conventions.
    total_update_phases = max(1, total_steps // (rollout_T * num_envs))
    lr_scheduler = make_lr_scheduler(optimizer, train_cfg, total_update_phases)

    # ---------------- PPO trainer ----------------
    delay_loss_coef = 0.0 if config["env"]["action_mode"] == "auto_only" else 1.0
    print(f"  delay_loss_coef      = {delay_loss_coef} "
          f"(0.0 disables delay-head gradient)")
    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        clip_epsilon    = train_cfg["clip_epsilon"],
        vf_clip_epsilon = train_cfg.get("vf_clip_epsilon"),
        gamma           = train_cfg["gamma"],
        gae_lambda      = train_cfg["gae_lambda"],
        ppo_epochs      = train_cfg["ppo_epochs"],
        num_minibatches = train_cfg["num_minibatches"],
        max_grad_norm   = train_cfg["max_grad_norm"],
        vf_coef         = train_cfg["vf_coef"],
        ent_coef        = train_cfg["ent_coef"],
        normalize_advantages = train_cfg.get("normalize_advantages", True),
        delay_loss_coef = delay_loss_coef,
        device          = device,
        thermal_blind   = thermal_blind,
        # E4 (HK-1.5.8): warmup the delay-channel loss coefficient over
        # the first delay_warmup_steps env-steps of hybrid training.
        # Default 0 = no warmup (pre-HK-1.5.8 behaviour).
        action_mode        = config["env"]["action_mode"],
        delay_warmup_steps = int(train_cfg.get("delay_warmup_steps", 0)),
    )

    # ---------------- Rollout buffer + reward normaliser ----------------
    buffer = RolloutBuffer(
        rollout_length = rollout_T,
        num_envs       = num_envs,
        action_mode    = config["env"]["action_mode"],
        device         = device,
    )
    use_reward_norm = bool(train_cfg.get("normalize_rewards", True))
    norm_p = RewardNormaliser(num_envs=num_envs, gamma=train_cfg["gamma"]) if use_reward_norm else None
    norm_d = RewardNormaliser(num_envs=num_envs, gamma=train_cfg["gamma"]) if use_reward_norm else None

    # ---------------- Defensive startup smoke ----------------
    print("\n[smoke] running 1 reset + 1 step on the vector env ...")
    diag = smoke_test_vector_env(vec_env, num_envs)
    print(f"  obs.shape          = {diag['first_obs_shape']}")
    print(f"  info_format        = {diag['info_format']}")
    print(f"  reward_channels    = {diag['reward_channels_keys']}")
    print(f"  graph_obs_keys     = {diag['graph_obs_keys']}")
    assert diag["reward_channels_keys"] == {"placement", "delay", "total"}, (
        f"reward_channels malformed: got {diag['reward_channels_keys']}.  "
        f"info_format was {diag['info_format']}."
    )
    print(f"[smoke] OK\n")

    # =================================================================
    # Main loop
    # =================================================================
    obs, info = vec_env.reset(seed=seed)
    graph_obs_list = _extract_graph_obs_list(info, num_envs)
    action_masks   = _extract_action_masks  (info, num_envs)
    global_step = 0
    update_phase = 0
    last_log_time = time.time()
    last_log_step = 0
    train_start_time = last_log_time
    best_eval_score = -float("inf")
    best_eval_stage_rank = -1     # int rank of the stage best was set in
                                   # (cold=0, warm=1, hot=2, static=99)
                                   # We only allow `best` to be set in the
                                   # current stage if its rank ≥ saved.
                                   # Otherwise an early easy-curriculum
                                   # peak (cold ep_ret=30) would lock-in
                                   # forever even though a hot-stage
                                   # ckpt is what we actually want.
    _stage_rank_map = {"cold": 0, "warm": 1, "hot": 2, "static": 99}
    log_every  = int(log_cfg.get("log_every_steps", 1000))
    save_every = int(log_cfg.get("save_every_steps", 50000))

    # Track per-update-phase summary stats (for eventual best-checkpoint
    # selection — we use mean episode return as proxy here)
    rolling_returns: List[float] = []
    episode_returns_buffer = np.zeros(num_envs, dtype=np.float64)

    # Env-physics rolling trackers.  These are the metrics the paper
    # actually cares about (makespan, peak temp, violations, cooling
    # overhead), independent of the PPO reward shaping.  We collect
    # per-step values and emit rolling means to TensorBoard at log time.
    # Buffers are cleared after every TB write to keep memory bounded.
    rolling_peak_temp:   List[float] = []      # max chip temp during a step
    rolling_idle_temp:   List[float] = []      # max chip temp at end of step
    rolling_step_ms:     List[float] = []      # exec_time + cooling_overhead
    rolling_cool_ms:     List[float] = []      # cooling_overhead alone
    rolling_violations:  List[int]   = []      # 1 if would_violate_without_delay
    rolling_dag_done:    List[int]   = []      # 1 if a DAG completed this step

    print(f"=== entering main loop  (target: {total_steps:,} steps) ===\n")

    while global_step < total_steps:
        # ============ COLLECT PHASE ============
        buffer.reset()
        for t in range(rollout_T):
            # 1. Forward (no_grad) for action selection.
            # Sanity check: graph_obs must not contain NaN/Inf BEFORE
            # the encoder sees them.  A single NaN in proc_x or task_x
            # propagates through the GNN softmax and silently produces
            # NaN logits, after which Categorical(...) crashes with a
            # confusing tensor dump.  Trapping it here gives us a
            # specific (env, key) pointer.
            for n_env in range(num_envs):
                go = graph_obs_list[n_env]
                for k in ("task_x", "proc_x"):
                    arr = np.asarray(go.get(k, []), dtype=np.float32)
                    if arr.size > 0 and not np.all(np.isfinite(arr)):
                        bad = np.argwhere(~np.isfinite(arr))
                        raise RuntimeError(
                            f"\n[ROLLOUT] env {n_env}, step {t}, "
                            f"global_step {global_step}: graph_obs[{k!r}] "
                            f"contains non-finite values.\n"
                            f"  shape: {arr.shape}\n"
                            f"  first {min(5, len(bad))} bad indices: "
                            f"{bad[:5].tolist()}\n"
                            f"  values at those indices: "
                            f"{arr[tuple(bad[:5].T)].tolist()}\n"
                            f"This usually means the simulator's temperature "
                            f"runaway poisoned a proc feature.  Check the "
                            f"reward shaping (especially soft/hard wall "
                            f"penalties) and the cooling logic."
                        )

            batch = build_batch(graph_obs_list, action_masks,
                                device=device, thermal_blind=thermal_blind)
            with torch.no_grad():
                act_out = model.act(batch)
            actions_t = act_out["action"]

            # 2. Step the env
            actions_np = actions_t.detach().cpu().numpy()
            next_obs, rewards, term, trunc, next_info = vec_env.step(actions_np)
            done = np.logical_or(term, trunc).astype(np.float32)
            global_step += num_envs

            # 3. Extract dual-channel rewards
            rp, rd = _extract_reward_channels(next_info, num_envs)

            # 4. (Optional) reward normalisation per channel
            if norm_p is not None:
                rp_n = norm_p.update_and_normalise(rp, done)
                rd_n = norm_d.update_and_normalise(rd, done)
            else:
                rp_n, rd_n = rp.astype(np.float32), rd.astype(np.float32)

            # 5. Track episode returns (raw, NOT normalised, for logging)
            episode_returns_buffer += (rp + rd)
            for n in range(num_envs):
                if done[n]:
                    rolling_returns.append(float(episode_returns_buffer[n]))
                    episode_returns_buffer[n] = 0.0

            # 5b. Track env-physics metrics for TB logging.  These are the
            # publication-relevant numbers (peak temp, makespan, violation
            # rate, cooling overhead).  Pulling them from info is cheap.
            from cpo_thermal_v2.training.env_factory import _extract_per_env
            for key, buf, dtype in (
                ("max_temp",                     rolling_peak_temp,  float),
                ("idle_max_temp",                rolling_idle_temp,  float),
                ("actual_workload_ms",           rolling_step_ms,    float),
                ("cooling_overhead_ms",          rolling_cool_ms,    float),
                ("would_violate_without_delay",  rolling_violations, int),
                ("dag_done",                     rolling_dag_done,   int),
            ):
                vals = _extract_per_env(next_info, key, num_envs)
                for v in vals:
                    if v is not None:
                        try:
                            buf.append(dtype(v))
                        except (TypeError, ValueError):
                            pass

            # 6. Push transition into buffer
            buffer.add(
                graph_obs   = graph_obs_list,
                action_masks= action_masks,
                actions     = actions_t,
                log_probs   = act_out["log_prob"],
                log_probs_p = act_out["log_prob_p"],
                log_probs_d = act_out["log_prob_d"],
                values_p    = act_out["v_placement"],
                values_d    = act_out["v_delay"],
                rewards_p   = rp_n,
                rewards_d   = rd_n,
                dones       = done,
            )

            # 7. Carry next state into the next iteration
            obs            = next_obs
            graph_obs_list = _extract_graph_obs_list(next_info, num_envs)
            action_masks   = _extract_action_masks  (next_info, num_envs)

            # 8. Curriculum stage transition?
            if curriculum.update(global_step):
                broadcast_curriculum_stage(vec_env, **curriculum.current_kwargs)
                print(f"\n[curriculum] step {global_step:,} → {curriculum.current.name!r} "
                      f"({curriculum.current.initial_temp_range}, "
                      f"max_dag_size={curriculum.current.max_dag_size})")
                writer.add_text("curriculum/transition",
                                f"step={global_step} stage={curriculum.current.name}",
                                global_step)

        # ============ TAIL VALUES (for GAE bootstrap) ============
        with torch.no_grad():
            tail_batch = build_batch(graph_obs_list, action_masks,
                                     device=device, thermal_blind=thermal_blind)
            v_p_tail, v_d_tail = model.get_value(tail_batch)

        # ============ PPO UPDATE ============
        metrics: PPOUpdateMetrics = trainer.update(buffer, v_p_tail, v_d_tail)
        update_phase += 1
        if lr_scheduler is not None:
            lr_scheduler.step()

        # ============ LOGGING ============
        # Per-update scalars
        writer.add_scalar("loss/total",         metrics.loss_total,     global_step)
        writer.add_scalar("loss/actor_p",       metrics.loss_actor_p,   global_step)
        writer.add_scalar("loss/actor_d",       metrics.loss_actor_d,   global_step)
        writer.add_scalar("loss/value_p",       metrics.loss_value_p,   global_step)
        writer.add_scalar("loss/value_d",       metrics.loss_value_d,   global_step)
        writer.add_scalar("policy/entropy_p",   metrics.entropy_p,      global_step)
        writer.add_scalar("policy/entropy_d",   metrics.entropy_d,      global_step)
        writer.add_scalar("policy/approx_kl_p", metrics.approx_kl_p,    global_step)
        writer.add_scalar("policy/approx_kl_d", metrics.approx_kl_d,    global_step)
        writer.add_scalar("policy/clip_frac_p", metrics.clip_frac_p,    global_step)
        writer.add_scalar("policy/clip_frac_d", metrics.clip_frac_d,    global_step)
        writer.add_scalar("optim/grad_norm",    metrics.grad_norm,      global_step)
        writer.add_scalar("optim/lr",           metrics.lr,             global_step)
        writer.add_scalar("advantage/p_mean",   metrics.advantage_p_mean, global_step)
        writer.add_scalar("advantage/p_std",    metrics.advantage_p_std,  global_step)
        writer.add_scalar("advantage/d_mean",   metrics.advantage_d_mean, global_step)
        writer.add_scalar("advantage/d_std",    metrics.advantage_d_std,  global_step)

        if rolling_returns:
            recent = rolling_returns[-100:]
            writer.add_scalar("episode/return_mean",  float(np.mean(recent)), global_step)
            writer.add_scalar("episode/return_std",   float(np.std(recent)),  global_step)

        # ============ ENV-PHYSICS SCALARS ============
        # These are the publication-relevant metrics. We log rolling
        # means since the last update phase (typically T*N = 2048 samples
        # per phase, plenty for a low-variance estimate), then clear the
        # buffers so memory stays bounded over a 5M-step run.
        if rolling_peak_temp:
            writer.add_scalar("env/peak_temp_during",
                              float(np.mean(rolling_peak_temp)), global_step)
            writer.add_scalar("env/peak_temp_max",
                              float(np.max(rolling_peak_temp)), global_step)
        if rolling_idle_temp:
            writer.add_scalar("env/idle_temp_step_end",
                              float(np.mean(rolling_idle_temp)), global_step)
        if rolling_step_ms:
            writer.add_scalar("env/step_ms_mean",
                              float(np.mean(rolling_step_ms)), global_step)
        if rolling_cool_ms:
            writer.add_scalar("env/cooling_ms_mean",
                              float(np.mean(rolling_cool_ms)), global_step)
            writer.add_scalar("env/cooling_fraction",
                              float(np.mean(np.asarray(rolling_cool_ms) > 0)),
                              global_step)
        if rolling_violations:
            # Rate of "would have violated without env intervention" — a key
            # safety metric.  In auto_only this includes auto-cool kicking
            # in; in agent_only this is true violations.
            writer.add_scalar("env/violation_rate",
                              float(np.mean(rolling_violations)), global_step)
        if rolling_dag_done:
            writer.add_scalar("env/dag_completion_rate",
                              float(np.mean(rolling_dag_done)), global_step)

        # Clear buffers so memory stays bounded
        rolling_peak_temp.clear()
        rolling_idle_temp.clear()
        rolling_step_ms.clear()
        rolling_cool_ms.clear()
        rolling_violations.clear()
        rolling_dag_done.clear()

        # Curriculum stage tag (lets you visually correlate stage transitions
        # with the curves).  Only emit when a stage is set.
        if curriculum.enabled and curriculum.current.name:
            stage_idx = {"cold": 0, "warm": 1, "hot": 2, "static": -1}.get(
                curriculum.current.name, 99
            )
            writer.add_scalar("env/curriculum_stage", stage_idx, global_step)

        # Console heartbeat: print one line per ``log_every_steps`` env-steps
        # crossed.  We compute steps/sec from the wall-clock elapsed since
        # the previous heartbeat, which gives a rolling estimate that's
        # both accurate on short runs and responsive to mid-training drift.
        prev_step_bucket = (global_step - rollout_T * num_envs) // log_every
        curr_step_bucket = global_step // log_every
        crossed_log_boundary = (update_phase == 1) or (curr_step_bucket > prev_step_bucket)
        if crossed_log_boundary:
            now = time.time()
            elapsed = max(1e-3, now - last_log_time)
            steps_since_last = global_step - last_log_step
            steps_per_sec = steps_since_last / elapsed
            last_log_time = now
            last_log_step = global_step

            ep_ret_str = f"{np.mean(rolling_returns[-100:]):.2f}" if rolling_returns else "n/a"
            print(f"step {global_step:>9,} | "
                  f"phase {update_phase:>4} | "
                  f"ep_ret(100) {ep_ret_str:>8} | "
                  f"loss {metrics.loss_total:+.3f} | "
                  f"KL_p {metrics.approx_kl_p:.4f} | "
                  f"clipfrac_p {metrics.clip_frac_p:.2f} | "
                  f"H_p {metrics.entropy_p:.2f} | "
                  f"H_d {metrics.entropy_d:.2f} | "
                  f"lr {metrics.lr:.1e} | "
                  f"steps/s {steps_per_sec:5.0f}")

        # Mirror to wandb if configured
        if wandb_run is not None:
            wandb_run.log({
                "loss/total":     metrics.loss_total,
                "loss/actor_p":   metrics.loss_actor_p,
                "loss/actor_d":   metrics.loss_actor_d,
                "loss/value_p":   metrics.loss_value_p,
                "loss/value_d":   metrics.loss_value_d,
                "policy/approx_kl_p": metrics.approx_kl_p,
                "policy/clip_frac_p": metrics.clip_frac_p,
                "ep_ret_100": (float(np.mean(rolling_returns[-100:]))
                                if rolling_returns else 0.0),
                "global_step":  global_step,
            })

        # ============ CHECKPOINTING ============
        if (global_step // save_every) > ((global_step - rollout_T * num_envs) // save_every):
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
            _save_checkpoint(model, optimizer, global_step, {
                "ep_ret_mean": float(np.mean(rolling_returns[-100:]))
                                if rolling_returns else 0.0,
                "loss_total":  metrics.loss_total,
            }, ckpt_path)
            print(f"  💾 saved {ckpt_path}")

            # Best checkpoint (by recent ep return, curriculum-aware).
            #
            # IMPORTANT: comparing raw episode return across curriculum
            # stages is meaningless — cold is much easier than hot and
            # will always show higher return.  We therefore require any
            # new "best" to come from a stage of EQUAL OR HIGHER rank
            # than the current best.  Once we've entered hot, only hot
            # ckpts can beat the current best; anything saved during
            # cold gets retired the first time we save a hot ckpt that
            # beats its own stage's ckpts.
            if rolling_returns:
                recent_score = float(np.mean(rolling_returns[-100:]))
                cur_stage = (curriculum.current.name
                             if curriculum.enabled else "static")
                cur_rank  = _stage_rank_map.get(cur_stage, 0)

                # Two cases that allow a new best:
                #  (a) we've entered a *harder* stage than the saved best
                #      → unconditionally seed a new best from the first
                #      checkpoint at that stage (raw score isn't directly
                #      comparable, so we reset)
                #  (b) we're at the *same* stage and beat the saved score
                if cur_rank > best_eval_stage_rank:
                    best_eval_score      = recent_score
                    best_eval_stage_rank = cur_rank
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    _save_checkpoint(model, optimizer, global_step, {
                        "ep_ret_mean":   recent_score,
                        "stage":         cur_stage,
                    }, best_path)
                    print(f"  🏆 new best  (entered stage='{cur_stage}', "
                          f"ep_ret={recent_score:.2f})  → {best_path}")
                elif cur_rank == best_eval_stage_rank \
                        and recent_score > best_eval_score:
                    best_eval_score = recent_score
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    _save_checkpoint(model, optimizer, global_step, {
                        "ep_ret_mean":   recent_score,
                        "stage":         cur_stage,
                    }, best_path)
                    print(f"  🏆 new best  (stage='{cur_stage}', "
                          f"ep_ret={recent_score:.2f})  → {best_path}")

    # ============ FINAL SAVE ============
    final_path = os.path.join(ckpt_dir, "final.pt")
    _save_checkpoint(model, optimizer, global_step, {
        "ep_ret_mean": float(np.mean(rolling_returns[-100:]))
                        if rolling_returns else 0.0,
    }, final_path)
    print(f"\n=== training complete  step={global_step:,}  → {final_path} ===")

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    vec_env.close()


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPO v2 PPO trainer (dual-channel actor-critic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a YAML config (relative to package root or absolute).",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override config (repeat as needed). "
             "Example: --override training.learning_rate=5e-5",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)

    # Algorithm dispatch (Decima true Step 5).  Default is PPO (the
    # cfg-less branch); 'reinforce' defers to the Mao-style trainer in
    # train_decima_true.  Add further algorithms here as needed.
    algo = (cfg.get("training", {}) or {}).get("algorithm", "ppo").lower()
    if algo == "reinforce":
        from cpo_thermal_v2.training.train_decima_true import train_reinforce
        train_reinforce(cfg)
        return
    if algo not in ("ppo", "default"):
        raise ValueError(
            f"Unknown training.algorithm={algo!r}. "
            f"Supported: 'ppo' (default), 'reinforce'."
        )
    train(cfg)


if __name__ == "__main__":
    main()
