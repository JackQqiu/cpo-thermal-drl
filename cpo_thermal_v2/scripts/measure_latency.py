#!/usr/bin/env python3
"""
measure_latency.py  (v2)

Measure inference latency of the trained hetero-GNN PPO scheduler.

Usage (from project root, activated cpo_drl conda env):
    cd <REPO_ROOT>
    python cpo_thermal_v2/scripts/measure_latency.py \
        --ckpt   checkpoints/stage2_hybrid_N17/best.pt \
        --N      17 \
        --mode   hybrid \
        --warmup 200 \
        --trials 1000

        python cpo_thermal_v2/scripts/measure_latency.py \
    --ckpt checkpoints/stage1_auto_only_N17/best.pt \
    --N 17 --mode auto_only

    python cpo_thermal_v2/scripts/measure_latency.py \
    --ckpt checkpoints/stage2_hybrid_N17/best.pt \
    --N 17 --mode agent_only

Run for all three modes by repeating with --mode auto_only / agent_only / hybrid.

What it does:
  1. Constructs a CPOThermalDAGEnvV2 at the requested N
  2. Constructs PPOActorCritic with hetero-GNN hyperparameters
  3. Loads checkpoint weights (handles {"model_state_dict": ...} or raw state dict)
  4. Resets env + steps a few times to reach a realistic mid-episode state
  5. Times `--trials` forward passes after `--warmup` warmups
  6. Prints median / p50 / p95 / p99 / max latency in ms

Output:
    median:    3.42  ms
    mean:      3.51  ms
    p50:       3.40  ms
    p95:       4.78  ms
    p99:       6.12  ms
    max:       9.41  ms
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import statistics

import numpy as np
import torch


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure per-decision inference latency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to .pt checkpoint (Stage 1 or Stage 2).")
    p.add_argument("--N",    type=int, default=17,
                   help="Number of processors (1 ASIC + N-1 OEs).")
    p.add_argument("--mode", type=str, default="hybrid",
                   choices=["auto_only", "hybrid", "agent_only"],
                   help="Inference mode used for timing.")
    p.add_argument("--warmup",   type=int, default=200,
                   help="Number of warm-up forward passes (untimed).")
    p.add_argument("--trials",   type=int, default=1000,
                   help="Number of timed forward passes.")
    p.add_argument("--seed",     type=int, default=42,
                   help="Random seed.")
    p.add_argument("--torch_threads", type=int, default=1,
                   help="torch.set_num_threads(); 1 = single-threaded.")
    p.add_argument("--K_delay",   type=int, default=5)
    p.add_argument("--hidden",    type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--steps_into_episode", type=int, default=20,
                   help="Step the env this many times before timing "
                        "(to reach a realistic mid-episode state).")
    p.add_argument("--dataset_path", type=str, default=None,
                   help="Path to Alibaba DAG dataset; if not given, env "
                        "uses its built-in default path.")
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Force CPU + single-thread reproducibility
    device = torch.device("cpu")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Locate project root and add to path
    # ------------------------------------------------------------------
    here       = os.path.dirname(os.path.abspath(__file__))
    proj_root  = os.path.abspath(os.path.join(here, ".."))     # cpo_thermal_v2/
    src_root   = os.path.abspath(os.path.join(proj_root, ".."))  # cpo_project/
    for p in (src_root, proj_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    # ------------------------------------------------------------------
    # Imports — match project layout
    # ------------------------------------------------------------------
    from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
    from cpo_thermal_v2.models import PPOActorCritic, build_batch

    # ------------------------------------------------------------------
    # Build env at the requested N (parameter is num_nodes, NOT num_processors)
    # ------------------------------------------------------------------
    env_kwargs = dict(
        num_nodes         = args.N,
        thermal_guardband = 80.0,
        thermal_critical  = 85.0,
        action_mode       = args.mode,
        K_delay           = args.K_delay if args.mode != "auto_only" else 5,
        initial_temp_range= (60.0, 75.0),  # extreme ambient = hardest decision
    )
    if args.dataset_path is not None:
        env_kwargs["dataset_path"] = args.dataset_path

    env = CPOThermalDAGEnvV2(**env_kwargs)
    obs, info = env.reset(seed=args.seed)

    # Step into the episode a bit so we have a realistic graph state
    for _ in range(args.steps_into_episode):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            obs, info = env.reset(seed=args.seed + 1)

    # graph_obs is wrapped in a numpy array of dtype=object; index 0 is the dict
    graph_obs   = info["graph_obs"][0]
    action_mask = info["action_mask"]

    # ------------------------------------------------------------------
    # Build model with hetero-GNN hyperparameters
    # ------------------------------------------------------------------
    model = PPOActorCritic(
        action_mode  = args.mode,
        K_delay      = args.K_delay,
        hidden       = args.hidden,
        num_layers   = args.num_layers,
        num_heads    = args.num_heads,
    ).to(device)

    # Load checkpoint — handle multiple wrapping conventions
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys ({len(missing)}): {missing[:3]} ...",
              file=sys.stderr)
    if unexpected:
        print(f"[WARN] unexpected keys ({len(unexpected)}): {unexpected[:3]} ...",
              file=sys.stderr)
    if not missing and not unexpected:
        print("[OK] checkpoint loaded with 0 missing / 0 unexpected keys")

    # Sanity check: verify weights are non-trivial (not all zeros / not random)
    # by printing a few weight statistics.
    with torch.no_grad():
        sample_weight = next(model.parameters())
        print(f"[sanity] first weight: mean={sample_weight.mean().item():.4f} "
              f"std={sample_weight.std().item():.4f} "
              f"abs_max={sample_weight.abs().max().item():.4f}")
        print(f"[sanity] (random init typically has std ~0.01-0.1; "
              f"trained weights often have wider distribution)")

    model.eval()

    # ------------------------------------------------------------------
    # Build the PyG Batch we'll re-time
    # NOTE: keep the SAME state across trials so the latency excludes
    # env-step cost — that's the per-decision model latency the paper
    # reports.
    # ------------------------------------------------------------------
    batch = build_batch([graph_obs], [action_mask], device=device)
    n_tasks = int(batch["task"].x.size(0)) if batch["task"].x.numel() > 0 else 0
    n_proc  = int(batch["proc"].x.size(0))

    print(f"[info] benchmarked state: {n_tasks} task nodes, {n_proc} proc nodes")
    print(f"[info] mode={args.mode}  N={args.N}  device={device}  "
          f"torch_threads={args.torch_threads}")
    print(f"[info] warmup={args.warmup}  trials={args.trials}")
    print()

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.act(batch, deterministic=True)

    # ------------------------------------------------------------------
    # Timed trials
    # ------------------------------------------------------------------
    times_ms = []
    with torch.no_grad():
        for _ in range(args.trials):
            t0 = time.perf_counter()
            _ = model.act(batch, deterministic=True)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    times_ms_sorted = sorted(times_ms)
    median = statistics.median(times_ms)
    mean   = statistics.mean(times_ms)
    p50    = times_ms_sorted[int(0.50 * len(times_ms_sorted))]
    p95    = times_ms_sorted[int(0.95 * len(times_ms_sorted))]
    p99    = times_ms_sorted[int(0.99 * len(times_ms_sorted))]
    mx     = times_ms_sorted[-1]

    print("=" * 50)
    print("Inference latency (one decision, single batch):")
    print("=" * 50)
    print(f"  median:    {median:7.3f}  ms")
    print(f"  mean:      {mean:7.3f}  ms")
    print(f"  p50:       {p50:7.3f}  ms")
    print(f"  p95:       {p95:7.3f}  ms")
    print(f"  p99:       {p99:7.3f}  ms")
    print(f"  max:       {mx:7.3f}  ms")
    print()
    print(f"  n_trials:  {args.trials}")
    print(f"  mode:      {args.mode}")
    print(f"  N:         {args.N}")
    print(f"  device:    {device}")
    print(f"  threads:   {args.torch_threads}")


if __name__ == "__main__":
    main()