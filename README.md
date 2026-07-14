# Permutation-Equivariant Graph RL for Thermal-Aware Scheduling on CPO

Thermal-aware microservice DAG scheduling for Co-Packaged Optics (CPO) data centres. A permutation-equivariant heterogeneous graph encoder with a cross-attention factored actor and dual-channel critic, trained with PPO under a two-stage curriculum.

This repository accompanies the manuscript *Permutation-Equivariant Graph Reinforcement Learning for Thermal-Aware Microservice Scheduling in Co-Packaged Optics Data Centers*, accepted for publication in *MDPI Symmetry* (2026).

---

## Method at a glance

- **Environment**: An ASIC plus `N-1` optical engines on a shared interposer, coupled by a calibrated RC thermal network. Microservice DAGs from the Alibaba 2021 trace are dispatched onto the processors under a hard thermal cap.
- **Policy**: A heterogeneous-graph encoder (3-layer GATv2 over task / processor / RC-coupling relations) feeds a cross-attention placement head plus a pooled-MLP delay head. A shared-trunk critic emits two value channels aligned with the placement and thermal components of the reward.
- **Training**: PPO with a thermal-aware reward and a two-stage curriculum — Stage 1 trains placement only, Stage 2 warm-starts and enables the anticipatory delay head.
- **Generalisation**: The encoder forward pass is parametric in `N`, so a single trained policy transfers zero-shot across `N ∈ {9, 13, 17, 24, 33}` processors.

---

## Installation

```bash
git clone https://github.com/JackQqiu/cpo-thermal-drl.git
cd cpo-thermal-drl
conda create -n cpo_rl python=3.10
conda activate cpo_rl
pip install -r cpo_thermal_v2/requirements.txt
```

If `torch_geometric` install fails, install PyTorch first, then follow the [PyG install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA version.

---

## Quick start

```bash
# Local smoke test (CPU, ~2 min)
python -m cpo_thermal_v2.scripts.sanity_check_maybe_precool_fix
```

---

## Training and evaluation

Cluster training (one V100 / A100):

```bash
# Stage 1: placement-only (3M steps, ~6h on V100)
sbatch cpo_thermal_v2/scripts/train_stage1.sbatch

# Stage 2: warm-started hybrid (1.5M steps, ~3h)
sbatch cpo_thermal_v2/scripts/train_stage2.sbatch
```

Edit the `--account=YOUR_SLURM_ACCOUNT` line in each `.sbatch` to match your cluster.

Reproducing paper tables and figures (after training):

```bash
# §5.2 main results table (hot N=17, n=500 paired episodes)
python -m cpo_thermal_v2.evaluation.evaluate --config cpo_thermal_v2/configs/eval_grand_matrix.yaml

# §5.4 generalisation envelope (5 sizes × 4 ambient regimes)
sbatch cpo_thermal_v2/scripts/eval_scaling.sbatch

# §5.6 auto-cool ablation
python -m cpo_thermal_v2.scripts.eval_autocool_ablation

# §5.6 RC-coupling edge isolation
python -m cpo_thermal_v2.scripts.eval_ours_no_rc_edge_paired

# §6.2 inference latency
python -m cpo_thermal_v2.scripts.measure_latency \
    --ckpt checkpoints/stage2_hybrid_N17/best.pt --N 17 --mode hybrid
```

Output goes to `eval_results/` as CSVs.

---

## Repository layout

```
cpo_thermal_v2/
├── envs/            gymnasium env, RC dynamics, DAG parser, reward shaping
├── models/          hetero GATv2 encoder, cross-attention actor, dual critic
├── data_pipeline/   Alibaba trace enrichment, RC matrix generation
├── training/        PPO trainer, GAE, rollout buffer, curriculum
├── baselines/       HEFT, Thermal-HEFT, Round-Robin, Decima (vanilla + thermal),
│                    HGATE-PPO, D2 (homog trunk + our actor), Throttled-HEFT
├── evaluation/      paired-episode evaluator, McNemar / Wilcoxon statistics
├── configs/         YAML configs (training, ablations, evaluation matrix)
└── scripts/         cluster launchers and evaluation scripts
```

---

## Citation

```bibtex
@article{qiu2026cpo,
  title   = {Permutation-Equivariant Graph Reinforcement Learning for
             Thermal-Aware Microservice Scheduling in Co-Packaged
             Optics Data Centers},
  author  = {Qiu, Zhaoqi and Peng, Linya and Fan, Fuming and Zuo, Haoran
             and Qiu, Wenjie and Xu, Bo and Deng, Tianping},
  journal = {Symmetry},
  year    = {2026},
  note    = {In press.}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
