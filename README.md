# CPO Thermal-Aware DAG Scheduler

Thermal-aware microservice DAG scheduling for Co-Packaged Optics (CPO)
data centres, using a heterogeneous GNN encoder and factored PPO with
a two-stage curriculum.  Accompanies a paper submitted to IEEE TPDS / TC.

For full architectural and deployment details, see
[`cpo_thermal_v2/README.md`](cpo_thermal_v2/README.md).

---

## What the code does

* Simulates an ASIC + N OE accelerators coupled by an RC thermal
  network, scheduling Alibaba microservice DAGs onto them under
  thermal constraints.
* Trains a heterogeneous GNN policy with **factored actions**
  (placement + delay), each with its own reward channel, advantage,
  and value head — so the two decisions can learn independently.
* Trains in **two stages**: Stage 1 learns placement under env auto-cool;
  Stage 2 warm-starts and unlocks the agent-controlled delay head.
* Evaluates against four baselines (Round-Robin, HEFT, Thermal-HEFT,
  Decima-style RL) at five topology sizes (N ∈ {9, 13, 17, 24, 33})
  zero-shot, and emits CSVs, IEEE-style PDF figures, and ready-to-paste
  LaTeX tables.

---

## Installation

```bash
git clone <repo_url>
cd <repo>
conda create -n cpo_rl python=3.10
conda activate cpo_rl
pip install -r requirements.txt
```

If `torch_geometric` install fails, install torch first then follow the
[official PyG install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
to pick the wheel index for your CUDA version.

---

## Configuration

All experiments are driven by YAML configs in `cpo_thermal_v2/configs/`,
which inherit from a shared `default.yaml`:

```
configs/
├── default.yaml             # shared defaults
├── stage1_auto_only.yaml    # Stage 1 training
├── stage2_hybrid.yaml       # Stage 2 (warm-starts from Stage 1)
└── eval_scaling.yaml        # full evaluation grid
```

Override any field from the CLI:

```bash
PYTHONPATH=. python -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage1_auto_only.yaml \
    --override training.total_steps=10000 \
    --override training.device=cuda:0
```

---

## Training

The pipeline is three SLURM jobs, run in sequence:

```bash
# Stage 1: placement-only training (~9h on V100)
sbatch cpo_thermal_v2/scripts/train_stage1.sbatch

# Stage 2: hybrid warm-start, unlocks delay head (~4h on V100)
sbatch cpo_thermal_v2/scripts/train_stage2.sbatch

# Stage E: full evaluation grid + figures + LaTeX tables (~12h on V100)
sbatch cpo_thermal_v2/scripts/eval_scaling.sbatch
```

Or chained with SLURM dependencies:

```bash
JOB1=$(sbatch --parsable cpo_thermal_v2/scripts/train_stage1.sbatch)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
              cpo_thermal_v2/scripts/train_stage2.sbatch)
sbatch --dependency=afterok:$JOB2 \
       cpo_thermal_v2/scripts/eval_scaling.sbatch
```

Total wall time: ~25-28h end-to-end on a single V100.

Output goes to `eval_results/scaling_v2/` — CSVs, PDF figures, and
`*.tex` table files ready to paste into the manuscript.

For a 2-minute local sanity check on Mac/CPU before launching real
training, see
[`cpo_thermal_v2/README.md#quick-start`](cpo_thermal_v2/README.md#quick-start).

---

## Repository layout

```
cpo_thermal_v2/
├── envs/            — gymnasium env, RC dynamics, DAG parser, reward
├── models/          — GNN encoder, factored actor, dual critic
├── data_pipeline/   — Alibaba trace enrichment, RC matrix generation
├── training/        — PPO trainer, GAE, rollout buffer, curriculum
├── baselines/       — Round-Robin, HEFT, Thermal-HEFT, Decima
├── evaluation/      — episode/DAG metrics, plots, LaTeX tables
├── configs/         — YAML configs (with inheritance)
└── scripts/         — SLURM sbatch files
```

---

## Citation

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
