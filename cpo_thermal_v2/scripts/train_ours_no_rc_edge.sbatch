#!/bin/bash
#SBATCH --job-name=cpo_v2_ours_no_rc_edge
#SBATCH --account=yanjia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ours_no_rc_edge_%j.out
#SBATCH --error=logs/ours_no_rc_edge_%j.err
# -------------------------------------------------------------------
# CPO v2 — Ours-NoRCEdge (Stage-1-style from scratch, RC edge ablated)
#
# Paper §5.3 / §6.2 isolation experiment for the RC-coupling edge
# attribute. Identical architecture, training schedule, and reward
# shaping to Ours-auto_only; the env emits empty edges_p2p so the
# encoder's p2p layers contribute no signal during the entire run.
#
# 3M steps; cold/warm/hot curriculum at 0 / 400k / 1.5M.
# Expected wall time on 1×V100 + 16 envs: ~6h.
#
# Outputs:
#   checkpoints/ours_no_rc_edge_N17/
#     ├── resolved_config.yaml
#     ├── ckpt_step_*.pt
#     ├── best.pt
#     └── final.pt
#   tb_logs/ours_no_rc_edge_N17/
# -------------------------------------------------------------------

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Ours-NoRCEdge starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# Activate conda env
source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

echo "[$(date)] Launching Ours-NoRCEdge PPO trainer..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/ablation_ours_no_rc_edge.yaml \
    "$@"

echo "[$(date)] Ours-NoRCEdge done"
