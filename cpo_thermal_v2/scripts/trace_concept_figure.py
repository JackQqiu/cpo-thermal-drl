"""
trace_concept_figure.py — capture trace for the §3 concept figure.

Runs ONE hot N=17 episode under both HEFT and Ours-hybrid using the
same env seed (so the DAG topology and initial temperatures are
identical between the two traces), then writes a single JSON payload
that the figure-generation script consumes.

The payload contains:
  - dag: structure (nodes, edges, workload) of the first few DAGs in
    the episode, enough to render a sample DAG panel
  - schedule[scheduler]: per-step (step, t_start, t_end, processor_id,
    task_id, dag_id) timeline used for the Gantt panel
  - thermal[scheduler]: per-step (t, peak_T, ASIC_T) used for the
    thermal-trace panel

Output: eval_results/concept_figure_trace.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from cpo_thermal_v2.envs import CPOThermalDAGEnvV2
from cpo_thermal_v2.envs.reward_shaping import RewardConfig
from cpo_thermal_v2.baselines.heft import HEFTScheduler
from cpo_thermal_v2.baselines import TrainedPPOScheduler


def trace_one_episode(
    env: CPOThermalDAGEnvV2,
    scheduler,
    seed: int,
    max_steps: int = 200,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    scheduler.reset(obs, info)

    schedule = []     # list of dict: step, t_start, t_end, proc, task_id, dag_id
    thermal  = []     # list of dict: t, peak_T, ASIC_T, OE_max_T
    dag_records = []  # list of dict: per DAG, structure

    t_now = 0.0
    recorded_dags = set()

    for step in range(max_steps):
        # Record current DAG structure once
        dag_id = env.current_dag_count
        if dag_id not in recorded_dags:
            cdag = env.current_dag
            nodes = list(cdag.graph.nodes)
            edges = list(cdag.graph.edges)
            workloads = {str(t): float(cdag.task_workload(t))
                          for t in nodes if hasattr(cdag, 'task_workload')}
            dag_records.append({
                "dag_id":   dag_id,
                "n_tasks":  len(nodes),
                "nodes":    [str(t) for t in nodes],
                "edges":    [[str(u), str(v)] for u, v in edges],
                "workloads": workloads,
            })
            recorded_dags.add(dag_id)

        # Which task is about to be dispatched?
        ready_tasks = list(env.ready_tasks)
        current_task = ready_tasks[0] if ready_tasks else None

        # Take step
        action = scheduler.schedule(obs, info)
        proc = int(action) if isinstance(action, (int, np.integer)) else \
               int(np.asarray(action).flatten()[0])
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        # Capture timing
        # workload_ms = execution_time on the chosen processor
        cooling_ms = float(next_info.get("cooling_overhead_ms", 0.0))
        compute_ms = float(next_info.get("pure_compute_ms", 0.0))
        # placement timeline: cooling, then compute
        t_dispatch = t_now + cooling_ms
        t_finish = t_dispatch + compute_ms
        schedule.append({
            "step":    step,
            "t_start": round(t_now, 3),
            "t_cool":  round(cooling_ms, 3),
            "t_exec":  round(compute_ms, 3),
            "t_finish": round(t_finish, 3),
            "proc":    proc,
            "task_id": str(current_task) if current_task is not None else None,
            "dag_id":  int(dag_id),
        })

        # Thermal sample at end of step
        T = env.thermal_engine.temperatures.copy()
        peak_T = float(next_info.get("max_temp", float(np.max(T))))
        thermal.append({
            "t":        round(t_finish, 3),
            "peak_T":   round(peak_T, 3),
            "ASIC_T":   round(float(T[0]), 3),
            "OE_max_T": round(float(T[1:].max()), 3),
        })

        t_now = t_finish

        obs, info = next_obs, next_info

        if terminated or truncated:
            break

    return {
        "dags": dag_records,
        "schedule": schedule,
        "thermal":  thermal,
        "n_steps_done": len(schedule),
        "terminated_at_step": step,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=100002,
                   help="env seed for both traces; pick one where HEFT visibly "
                        "exceeds T_pen quickly so the concept figure is striking")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--num_nodes", type=int, default=17)
    p.add_argument("--output", type=str,
                   default="eval_results/concept_figure_trace.json")
    args = p.parse_args()

    common_env_kwargs = dict(
        num_nodes          = args.num_nodes,
        dt                 = 1e-3,
        thermal_target     = 75.0,
        thermal_guardband  = 80.0,
        thermal_critical   = 85.0,
        mask_temp          = 82.0,
        action_mode        = "auto_only",
        initial_temp_range = (60.0, 75.0),
        max_dag_size       = None,
        dags_per_episode   = 20,
        temp_rise_per_ms_asic = 0.08,
        temp_rise_per_ms_oe   = 0.18,
        oe_active_power    = 40.0,
        max_cooling_steps  = 100,
        truncate_mode      = "soft",   # don't truncate on T_crit so we capture
                                       # full thermal trace beyond violation
        reward_config      = RewardConfig(),
        dataset_path       = "./data_pipeline/process/alibaba_dags_v2.json",
    )

    # ----- HEFT -----
    env_h = CPOThermalDAGEnvV2(**common_env_kwargs)
    heft = HEFTScheduler(num_nodes=args.num_nodes, action_mode="auto_only")
    print(f"[trace] HEFT episode seed={args.seed}")
    trace_heft = trace_one_episode(env_h, heft, args.seed, args.max_steps)
    print(f"  HEFT: {trace_heft['n_steps_done']} steps, "
          f"peak T across episode = "
          f"{max(t['peak_T'] for t in trace_heft['thermal']):.2f} °C")

    # ----- Ours-hybrid (hybrid action mode: placement + delay channels) -----
    common_env_kwargs_hybrid = dict(common_env_kwargs)
    common_env_kwargs_hybrid["action_mode"] = "hybrid"
    env_o = CPOThermalDAGEnvV2(**common_env_kwargs_hybrid)
    ours = TrainedPPOScheduler(
        ckpt_path="checkpoints/stage2_hybrid_N17/best.pt",
        action_mode="hybrid",
        deterministic=True, device="cpu",
        scheduler_label="Ours-hybrid",
    )
    print(f"[trace] Ours-hybrid episode seed={args.seed}")
    trace_ours = trace_one_episode(env_o, ours, args.seed, args.max_steps)
    print(f"  Ours: {trace_ours['n_steps_done']} steps, "
          f"peak T across episode = "
          f"{max(t['peak_T'] for t in trace_ours['thermal']):.2f} °C")

    out = {
        "seed":      args.seed,
        "num_nodes": args.num_nodes,
        "HEFT":      trace_heft,
        "Ours-hybrid": trace_ours,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[trace] wrote {args.output}")


if __name__ == "__main__":
    main()
