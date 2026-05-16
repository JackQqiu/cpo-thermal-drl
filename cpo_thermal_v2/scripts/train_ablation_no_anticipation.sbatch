#!/bin/bash
#SBATCH --job-name=cpo_v2_ablation_no_anticipation
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # 24h budget (Stage 2 ablation ~3h)
#SBATCH --output=logs/ablation_no_anticipation_%j.out
#SBATCH --error=logs/ablation_no_anticipation_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage 2 ablation: zero anticipation_bonus
#
# Tests the contribution of reward.anticipation_bonus (=0 in this run,
# 1.5 in main Stage 2).  All other knobs unchanged.  Warm-starts from
# Stage 1 best.pt (same as stage2_hybrid.yaml).
#
# Output: checkpoints/stage2_ablation_no_anticipation_N17/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Stage 2 ablation (no_anticipation) starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# Activate conda env
source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

# -------------------------------------------------------------------
# Sanity: Stage 1 checkpoint must exist (warm-start source)
# -------------------------------------------------------------------
WARM_CKPT="checkpoints/stage1_auto_only_N17/best.pt"
if [ ! -f "$WARM_CKPT" ]; then
    echo "❌ Stage 1 checkpoint not found at: $WARM_CKPT"
    echo "   Run scripts/train_stage1.sbatch first (or override warm_start_path)."
    exit 1
fi
echo "✅ found warm-start checkpoint: $WARM_CKPT"

# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------
echo "[$(date)] Launching ablation (no_anticipation) PPO trainer ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/ablation_no_anticipation.yaml \
    "$@"

echo "[$(date)] Stage 2 ablation (no_anticipation) done"
