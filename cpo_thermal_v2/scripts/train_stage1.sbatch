#!/bin/bash
#SBATCH --job-name=cpo_v2_stage1_auto_only
#SBATCH --account=yanjia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage 1 (auto_only warm-up training, N=17)
#
# Trained on env post-Stage 0 fix + HK-1 + HK-2.1 + HK-2.1.5 + HK-1.5.5.
# 3M steps (was 5M paper); curriculum cold/warm/hot at 0/400k/1.5M.
# Expected wall time on 1×V100: ~6h.
#
# Outputs:
#   checkpoints/stage1_auto_only_N17/
#     ├── resolved_config.yaml
#     ├── ckpt_step_*.pt
#     ├── best.pt
#     └── final.pt
#   tb_logs/stage1_auto_only_N17/
# -------------------------------------------------------------------

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Stage 1 starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# Activate conda env
source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

# Train
echo "[$(date)] Launching Stage 1 PPO trainer..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage1_auto_only.yaml \
    "$@"

echo "[$(date)] Stage 1 done"
