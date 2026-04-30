# CPO Thermal-Aware DAG Scheduler — v2

A clean, physics-aligned re-implementation of a thermal-aware DAG scheduler
for **Co-Packaged Optics (CPO)** data centres, using a heterogeneous GNN +
factored PPO with curriculum-style two-stage training.

This repository accompanies a paper targeted at **IEEE TPDS / TC**.  The
codebase is organised into 5 numbered "Stages" matching the development
plan:

| Stage | Subsystem                                | Status         |
|-------|------------------------------------------|----------------|
| A     | Infrastructure (env shell, RC, DAG)      | ✅ complete     |
| B     | Reward shaping, curriculum               | ✅ complete     |
| C     | GNN encoder, factored actor, dual critic | ✅ complete     |
| D     | PPO trainer, two-stage curriculum        | ✅ complete     |
| E     | Baselines, evaluation, plots, tables     | ✅ complete     |

---

## Table of Contents

1. [Why this repo exists](#why-this-repo-exists)
2. [Quick start (3 commands)](#quick-start)
3. [Repository layout](#repository-layout)
4. [Physical & ML modelling](#physical--ml-modelling)
5. [Configuration system](#configuration-system)
6. [Two-stage training pipeline](#two-stage-training-pipeline)
7. [Evaluation pipeline (Stage E)](#evaluation-pipeline-stage-e)
8. [SLURM deployment](#slurm-deployment)
9. [TensorBoard metrics legend](#tensorboard-metrics-legend)
10. [Troubleshooting](#troubleshooting)
11. [Reproducibility](#reproducibility)

---

## Why this repo exists

The v1 codebase trained but had structural problems making it unfit for
publication:

* Single-channel reward → broken credit assignment between **placement**
  (where to run a task) and **delay** (when to run it).
* Single-graph GNN → lost the distinction between DAG dependencies (a
  control-flow construct) and RC heat coupling (a physical construct).
* Hard-coded N=17 → no scaling experiment.
* All-in-one rewrite of action space mid-training → couldn't reproduce
  earlier runs.

v2 fixes all of the above by separating concerns:

* **Two reward channels** (placement / delay), routed by physical
  root-cause, consumed by a **factored actor** with **dual advantage**.
* **Heterogeneous GNN** with separate task↔task and proc↔proc edges,
  cross-attention for placement, MLP+max-pool for delay.
* **Parametric in N**: encoder/actor/critic all permutation- and
  size-invariant, evaluated zero-shot at N ∈ {9, 13, 17, 24, 33}.
* **Two-stage curriculum**: Stage 1 trains placement under env auto-cool
  baseline; Stage 2 warm-starts and unlocks the delay head.

---

## Quick start

```bash
# 1. From the project root (containing cpo_thermal_v2/), set Python path:
export PYTHONPATH=".:${PYTHONPATH:-}"

# 2. Smoke-test on Mac/CPU (~2 minutes, validates everything):
python -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage1_auto_only.yaml \
    --override training.device=cpu \
    --override training.total_steps=2000 \
    --override training.num_envs=4 \
    --override training.vector_env_mode=sync \
    --override logging.run_name=local_smoke

# 3. Real training on the GPU server (24h end-to-end):
sbatch cpo_thermal_v2/scripts/train_stage1.sbatch    # ~9h on V100
sbatch cpo_thermal_v2/scripts/train_stage2.sbatch    # ~4h on V100
sbatch cpo_thermal_v2/scripts/eval_scaling.sbatch    # ~12h on V100
```

After all three: `eval_results/scaling_v2/` contains CSVs, PDF figures,
and ready-to-paste LaTeX tables.

---

## Repository layout

```
cpo_thermal_v2/
├── __init__.py                  — top-level re-exports
├── envs/                        — Stage A+B (numpy-only)
│   ├── cpo_thermal_env.py       —   CPOThermalDAGEnvV2 (gymnasium env)
│   ├── rc_dynamics.py           —   RCThermalDynamics (state-space LTI)
│   ├── reward_shaping.py        —   RewardConfig + dual-channel reward
│   └── dag_parser.py            —   MicroserviceDAG (Alibaba traces)
├── models/                      — Stage C (torch + torch_geometric)
│   ├── hetero_encoder.py        —   HeteroEncoder (task graph + proc graph)
│   ├── cross_attention_actor.py —   placement: cross-attention
│   ├── value_critic.py          —   DualCritic (placement + delay heads)
│   └── ppo_actor_critic.py      —   PPOActorCritic (top-level wrapper)
├── data_pipeline/               — offline data preparation
│   ├── compute_dag_features.py  —   enrich Alibaba JSON with ρ, slack, etc
│   └── generate_matrices.py     —   build A/B/D RC matrices for 5 sizes
├── training/                    — Stage D
│   ├── config_loader.py         —   YAML inheritance + CLI overrides
│   ├── curriculum.py            —   cold→warm→hot stage manager
│   ├── gae.py                   —   Generalised Advantage Estimation
│   ├── rollout_buffer.py        —   dual-channel transition store
│   ├── env_factory.py           —   AsyncVectorEnv + smoke test + info parsing
│   ├── ppo_trainer.py           —   factored PPO loss, optimisation step
│   └── train.py                 —   main entry; CLI with --override
├── baselines/                   — Stage E
│   ├── base.py                  —   BaseScheduler abstract interface
│   ├── round_robin.py           —   sanity baseline
│   ├── heft.py                  —   classic Heterogeneous Earliest Finish Time
│   ├── thermal_heft.py          —   PROCHOT-aware HEFT (strongest classical baseline)
│   ├── decima.py                —   Decima-style RL (thermal features masked)
│   └── trained_ppo.py           —   wraps a saved checkpoint
├── evaluation/                  — Stage E
│   ├── metrics.py               —   per-step / per-DAG / per-episode collection
│   ├── runner.py                —   schedulers × N × modes grid driver
│   ├── plots.py                 —   IEEE-style figures (scaling/box/ρ/util)
│   ├── tables.py                —   booktabs+siunitx LaTeX tables
│   └── evaluate.py              —   Stage E main entrypoint
├── configs/
│   ├── default.yaml             —   shared defaults (most fields)
│   ├── stage1_auto_only.yaml    —   inherits default, locks delay head
│   ├── stage2_hybrid.yaml       —   inherits default, warm-start path, hybrid mode
│   └── eval_scaling.yaml        —   inherits default, evaluation grid
└── scripts/
    ├── train_stage1.sbatch      —   ~9h V100, 5M steps auto_only
    ├── train_stage2.sbatch      —   ~4h V100, 2M steps hybrid
    ├── eval_scaling.sbatch      —   ~12h V100, full grid
    └── smoke_server.sbatch      —   server-side validation (5 minutes)
```

---

## Physical & ML modelling

### Hardware model

```
ASIC (proc 0)  +  N OE accelerators (proc 1..N)
                  └─ all coupled via RC thermal network ──┐
                                                          │
T_{t+1} = A · T_t + B · P_t + D · T_amb                   │
                                                          │
Power model (per proc i):                                 │
  P_i(t) = P_active(i) · busy_i(t)                        │
         + P_leak_base · exp(β · (T_i(t) - 25°C))         │
                                                          │
Constraints:                                              │
  T_pen  = 80°C    soft-penalty wall                      │
  T_crit = 85°C    hard truncation                        │
```

The `A` matrix encodes inter-proc heat coupling; non-zero off-diagonal
entries mean putting load on proc *i* will heat proc *j* via shared
substrate.  This information is **explicitly exposed to the GNN** as the
proc-graph edge attributes.

### Workload model

* Alibaba microservice trace (DAGs of tasks)
* ρ-bound: communication-to-computation ratio per DAG, used to compute
  task-level slack.
* Tasks have heterogeneous execution time per proc (ASIC ≈ 5× faster
  than OE but 4× higher power), encoded as task→proc edge attributes.

### Action space

Three modes, switched by config (`env.action_mode`):

| Mode          | Action shape                  | Env behaviour              |
|---------------|-------------------------------|----------------------------|
| `auto_only`   | `int` (proc index)            | env auto-cools when needed |
| `agent_only`  | `[proc_idx, delay_idx]` (1×2) | env does NOT auto-cool     |
| `hybrid`      | `[proc_idx, delay_idx]` (1×2) | env auto-cools as fallback |

`hybrid` is the paper's proposed configuration; `agent_only` is a
zero-shot evaluation that demonstrates the agent has *internalised*
anticipatory cooling.

### Network architecture

```
Input: graph_obs dict
   ├─ task_x:     (V_task, 8)    — workload, slack, depth, in/out-deg, ...
   ├─ proc_x:     (N, 7)         — T, dT/dt, leakage, headroom, busy, ..., is_ASIC
   ├─ edges_t2t:  task→task DAG dependencies
   ├─ edges_p2p:  proc→proc RC coupling (edge_attr = A_ij)
   └─ edges_t2p:  task→proc affinity   (edge_attr = est_time, est_dT)

HeteroEncoder
   ├─ Task GNN:   GATv2 over task graph                 → x_task (V_task, 128)
   ├─ Proc GNN:   GATv2 over proc graph (RC-aware)      → x_proc (N, 128)
   └─ Cross-attn: bidirectional task ↔ proc message     (refines both)

Factored Actor                          Dual Critic
   ├─ Placement (cross-attention):         ├─ V_placement: MLP on pooled state
   │    Query: current_task embedding      └─ V_delay:     MLP on pooled state
   │    Keys/Vals: per-proc embeddings
   │    Output: (N+1) logits
   │
   └─ Delay (MLP + mean+max pool):
        Input: current_task ⊕ pool(x_proc)
        Output: K_delay logits
```

---

## Configuration system

All configs live in `configs/` and use a simple inheritance scheme:

```yaml
# stage1_auto_only.yaml
_inherit: default        # loads default.yaml first; this overrides

env:
  action_mode: auto_only
  num_nodes:   17

training:
  delay_loss_coef: 0.0   # lock the delay head
  total_steps:     5000000
  ...
```

CLI overrides use dot notation:

```bash
python -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage1_auto_only.yaml \
    --override training.total_steps=10000 \
    --override training.device=cuda:0 \
    --override logging.run_name=quick_test
```

You can `--override` any leaf value, including booleans (`true`/`false`),
floats (`3e-4`), nulls (`null`), and lists (`[1,2,3]`).

---

## Two-stage training pipeline

### Why two stages?

`placement` (where) and `delay` (when) have very different learning
dynamics:

| Aspect         | Placement                    | Delay                        |
|----------------|------------------------------|------------------------------|
| Search space   | N+1 procs                    | K_delay = 5 fractions        |
| Reward latency | ~1 step                      | ~3-10 steps                  |
| Signal SNR     | high (T responds in 1 dt)    | low (must predict future)    |
| Influence     | local (one task)             | global (changes all timing)  |

Co-training is unstable: delay head's early random actions inject
noise into the trajectories that V_placement is trying to fit, slowing
both heads.  Stage 1 trains placement against a fixed env-auto-cool
baseline; Stage 2 warm-starts and unlocks delay.

### Stage 1: auto_only (≈ 9h on V100)

* Action space: `Discrete(N+1)` (proc index only)
* Env behaviour: auto-cools when temp approaches T_pen
* `delay_loss_coef = 0` (delay head receives zero gradient)
* Curriculum: cold (≤ 1M steps) → warm (≤ 3M steps) → hot (5M steps)

```bash
sbatch cpo_thermal_v2/scripts/train_stage1.sbatch
```

Output: `checkpoints/stage1_auto_only_N17/{best,final}.pt`

### Stage 2: hybrid warm-start (≈ 4h on V100)

* Action space: `MultiDiscrete([N+1, K_delay])`
* Env behaviour: auto-cool still active as safety net
* `delay_loss_coef = 1.0` (delay head trains)
* Loaded from Stage 1 best.pt (encoder + placement head + V_placement)
* Curriculum: skips cold; warm → hot directly
* Total steps: 2M

```bash
sbatch cpo_thermal_v2/scripts/train_stage2.sbatch
```

Output: `checkpoints/stage2_hybrid_N17/{best,final}.pt`

### What you'll see in TensorBoard

Stage 1 (correct behaviour):
- `policy/entropy_p` decreasing slowly (~2.7 → ~2.5 over 5M steps)
- `policy/entropy_d` flat (~0.46) — delay head not updating
- `loss/value_p` increasing as critic adapts to harder curriculum
- `loss/value_d` ≈ 0 (delay V doesn't update either)
- `episode/return_mean` step-changes at curriculum transitions

Stage 2 first 200-500k steps (calibration period, expected):
- `loss/value_d` jumps from ~0 to a transient peak as V_delay learns
- `advantage/d_mean` rises from -2 toward 0
- `episode/return_mean` may briefly dip then exceed Stage 1 final

---

## Evaluation pipeline (Stage E)

### Grid

```
schedulers (7) × topology sizes (5) × action_modes (3) = 17,500 episodes
```

| Scheduler         | Type             | Notes                                        |
|-------------------|------------------|----------------------------------------------|
| `RoundRobin`      | classical        | sanity baseline                              |
| `HEFT`            | classical        | thermally blind                              |
| `ThermalHEFT`     | classical        | strongest non-learning baseline              |
| `Decima`          | learned (proxy)  | our ckpt + thermal features masked           |
| `Ours-auto_only`  | learned          | env-cool only (Stage 1 reproduction)         |
| `Ours-agent_only` | learned          | **zero-shot**: env-cool disabled, agent-only |
| `Ours-hybrid`     | learned          | the paper's main proposed configuration      |

### Run

```bash
sbatch cpo_thermal_v2/scripts/eval_scaling.sbatch
```

Or interactively:

```bash
PYTHONPATH=. python -m cpo_thermal_v2.evaluation.evaluate \
    --config cpo_thermal_v2/configs/eval_scaling.yaml \
    --override eval.checkpoint_path=checkpoints/stage2_hybrid_N17/best.pt \
    --override eval.num_episodes=100      # quick run
```

To re-render plots/tables from existing CSVs without re-running:

```bash
python -m cpo_thermal_v2.evaluation.evaluate \
    --config cpo_thermal_v2/configs/eval_scaling.yaml \
    --skip-grid
```

### Output

```
eval_results/scaling_v2/
├── episodes.csv               — 17,500 rows, all per-episode metrics
├── dags.csv                   — per-DAG records (~100k rows)
├── plots/
│   ├── scaling_total_makespan_ms.{pdf,png}    — Fig 1
│   ├── scaling_peak_temp_episode.{pdf,png}
│   ├── scaling_violations_total.{pdf,png}
│   ├── box_*.{pdf,png}                         — Fig 2
│   ├── rho_analysis_N17.{pdf,png}              — Fig 3
│   └── proc_util_N17.{pdf,png}                 — Fig 4
└── tables/
    ├── tab1_main.tex                            — main results, N=17
    ├── tab2_scaling_makespan.tex                — vs N
    ├── tab2_scaling_violations.tex
    └── tab3_ablation.tex                        — Ours-{auto/agent/hybrid}
```

The `.tex` files use `booktabs` + `siunitx`; preamble:

```latex
\usepackage{booktabs}
\usepackage{siunitx}
\sisetup{
    round-mode = places, round-precision = 2,
    separate-uncertainty = true,
}
```

---

## SLURM deployment

### Cluster requirements

* SLURM with a GPU partition (any modern GPU; V100 is sufficient)
* Conda environment with: `torch`, `torch-geometric`, `gymnasium ≥ 1.0`,
  `numpy`, `pandas`, `matplotlib`, `pyyaml`, `tensorboard`
* ≥ 64 GB RAM (training uses ~20 GB peak with 16 envs)
* ≥ 100 GB disk for checkpoints + TB logs + eval CSVs

### Per-cluster sbatch customisation

Open each `scripts/*.sbatch` and edit:

```bash
#SBATCH --partition=gpu              # → your GPU partition name
#SBATCH --gres=gpu:1                 # → e.g. gpu:v100:1 to lock V100
#SBATCH --cpus-per-task=16           # match num_envs
#SBATCH --mem=64G

# Uncomment and set your conda env:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_rl
```

### Walltime budgets (V100, measured)

| Job        | Steps          | Wall time | Buffer  |
|------------|----------------|-----------|---------|
| `stage1`   | 5,000,000      | ~9.3h     | 18h     |
| `stage2`   | 2,000,000      | ~3.7h     | 10h     |
| `eval`     | 17,500 eps     | ~12-15h   | 18h     |

### Submission order (sequential)

```bash
JOB1=$(sbatch --parsable cpo_thermal_v2/scripts/train_stage1.sbatch)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
              cpo_thermal_v2/scripts/train_stage2.sbatch)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 \
              cpo_thermal_v2/scripts/eval_scaling.sbatch)
echo "Stage 1: $JOB1, Stage 2: $JOB2, Eval: $JOB3"
```

Total: ~25-28h wall clock from submission to LaTeX-ready figures.

### Live monitoring

```bash
# Job status
squeue -u $USER

# Console heartbeat
tail -f logs/stage1_<JOBID>.out

# TensorBoard (run on local machine, port-forward via SSH)
tensorboard --logdir tb_logs/ --port 6006
```

---

## TensorBoard metrics legend

### `loss/*` — PPO health

| Tag             | Meaning                                            |
|-----------------|----------------------------------------------------|
| `loss/total`    | Total PPO loss; should decrease then plateau       |
| `loss/actor_p`  | Placement actor surrogate loss                     |
| `loss/actor_d`  | Delay actor surrogate loss (≈0 in Stage 1)         |
| `loss/value_p`  | Placement critic MSE                               |
| `loss/value_d`  | Delay critic MSE (≈0 in Stage 1, jumps in Stage 2) |

### `policy/*` — exploration health

| Tag                 | Healthy range          | Warning sign            |
|---------------------|------------------------|-------------------------|
| `policy/entropy_p`  | 1.5 < H < ln(N+1)      | H < 0.5: collapse       |
| `policy/entropy_d`  | flat ~0.46 in Stage 1  | Δ in Stage 2 expected   |
| `policy/approx_kl_p`| < 0.05                 | > 0.1: trust-region break |
| `policy/clip_frac_p`| < 0.3                  | > 0.5: lr too high       |

### `optim/*`

| Tag              | What it tells you                          |
|------------------|--------------------------------------------|
| `optim/lr`       | Linear-decay schedule                      |
| `optim/grad_norm`| ~0.1-0.5 stable; > 5 means instability     |

### `advantage/*`

| Tag              | Note                                        |
|------------------|---------------------------------------------|
| `advantage/p_mean`| ≈0 if normalised; mild drift OK             |
| `advantage/p_std` | should stay ~1 if running normaliser is on  |
| `advantage/d_mean`| **systematically negative in Stage 1**      |
|                  | (V_delay lags R_delay because of zero loss) |
|                  | recovers to ~0 in Stage 2's first 500k steps |

### `episode/*`

| Tag                  | Note                                       |
|----------------------|--------------------------------------------|
| `episode/return_mean`| 100-ep rolling mean; step-changes at curriculum  |

### `env/*` — physics observables

| Tag                          | Description                                  |
|------------------------------|----------------------------------------------|
| `env/peak_temp_during`       | mean of max-T-during-step over phase         |
| `env/peak_temp_max`          | max-of-max in the phase                      |
| `env/idle_temp_step_end`     | mean T at step end (idle baseline)           |
| `env/step_ms_mean`           | mean total step time = exec + cooling + delay |
| `env/cooling_ms_mean`        | env's auto-inserted cooling time             |
| `env/cooling_fraction`       | fraction of steps with cooling > 0           |
| `env/violation_rate`         | fraction of steps with T ≥ T_pen             |
| `env/dag_completion_rate`    | fraction of steps where a DAG completed      |
| `env/curriculum_stage`       | 0=cold, 1=warm, 2=hot, -1=static             |

---

## Troubleshooting

### Smoke test fails on Mac

```bash
cd <project_root>
PYTHONPATH=. python -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage1_auto_only.yaml \
    --override training.device=cpu \
    --override training.total_steps=2000 \
    --override training.num_envs=4 \
    --override training.vector_env_mode=sync \
    --override logging.run_name=local_smoke
```

If this fails: paste the **full traceback**.  Most common issues are
gymnasium 1.x info-format quirks; `env_factory.py:_extract_per_env`
already handles 5 known formats.

### Server first-time setup

Run the 5-minute server smoke first to catch any
environment-specific problems (CUDA, fork-vs-spawn IPC, file paths)
before committing to a 9h training run:

```bash
sbatch cpo_thermal_v2/scripts/smoke_server.sbatch
```

Should complete in ~5-8 minutes with output like:

```
[smoke] OK
=== entering main loop  (target: 10,000 steps) ===
step     1,024 | phase    1 | ep_ret(100)     8.39 | ... | steps/s   139
step    10,240 | phase   10 | ep_ret(100)    11.50 | ... | steps/s   152
=== training complete ===
```

If `steps/s` < 80, async vector env is bottlenecked — try
`--override training.vector_env_mode=sync` or reduce `num_envs`.

### "graph_obs contains non-finite values"

This is a defensive trap inserted in the rollout loop.  It fires if
the env's RC dynamics produced NaN/Inf temperatures, which then
poison the GNN's softmax.  Should never happen due to clamps in
`_proc_features` and `_compute_power`, but if it does the traceback
will tell you exactly which env / step / feature is bad.  Paste it
and check whether reward shaping accidentally produced a positive
feedback loop.

### Stage 1's `episode/return_mean` *decreases* over time

That's **expected**.  The curriculum scheduler raises difficulty at
1M and 3M steps; each transition causes a step-down in raw return
because the policy now faces hotter initial conditions and bigger
DAGs.  What matters is that `loss/total` is decreasing within each
plateau and `policy/entropy_p` isn't collapsing.

### Stage 2 looks worse than Stage 1 for the first 500k steps

Also expected.  Stage 2 starts with a fresh `V_delay` head; for the
first ~500k steps it's miscalibrated and contaminates the policy
gradient.  After ~500k-1M steps `advantage/d_mean` should normalise
toward 0 and `episode/return_mean` should surpass Stage 1's plateau.

If after 1M steps Stage 2 is still worse: training is unstable.  Most
likely fix: lower `training.delay_loss_coef` from 1.0 to 0.5, or
lower `training.lr` from 3e-4 to 1e-4.

---

## Reproducibility

All randomness is controllable:

```yaml
training:
  seed: 42        # default; set to any int
```

* Seeds the Python `random`, `numpy`, and `torch` RNGs.
* Each parallel env gets `seed + env_index` (deterministic shards).
* Eval uses `seed_base + episode_id` (default `seed_base=100000`)
  so eval episodes are independent of training seed.

The `resolved_config.yaml` saved alongside each checkpoint records the
**exact** config used at training time (with all CLI overrides applied)
so a run can be reproduced from just that file.

To reproduce the paper's main results from scratch:

```bash
git clone <repo_url>
cd <repo>
conda env create -f environment.yml
conda activate cpo_rl
JOB1=$(sbatch --parsable cpo_thermal_v2/scripts/train_stage1.sbatch)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
              cpo_thermal_v2/scripts/train_stage2.sbatch)
sbatch --dependency=afterok:$JOB2 cpo_thermal_v2/scripts/eval_scaling.sbatch
# Wait ~28h. Tables in eval_results/scaling_v2/tables/ go directly into
# the LaTeX manuscript; figures into Figures/.
```

---

## Citation

If you use this code, please cite the accompanying paper (placeholder):

```bibtex
@article{cpo_thermal_v2_2026,
  title   = {Thermal-Aware Microservice DAG Scheduling for Co-Packaged
             Optics Data Centers via Proximal Policy Optimization},
  author  = {…},
  journal = {IEEE Transactions on Parallel and Distributed Systems},
  year    = {2026},
}
```

---

## License

(To be specified — likely MIT.)