#!/bin/bash
#SBATCH --job-name=cpo_v2_stage2_hybrid_v2
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # 24h budget (Stage 2 v2 ~3-6h)
#SBATCH --output=logs/stage2_v2_%j.out
#SBATCH --error=logs/stage2_v2_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage 2 v2 (HK-1.5.8 fixes: gated anticipation_bonus + delay-loss warmup)
#
# Loads checkpoints/stage1_auto_only_N17/best.pt as warm start.
# Trains the hybrid agent under the v2 reward shaping (anticipation_bonus
# requires would_violate_without_delay) and v2 trainer (delay_loss_coef
# linear-warmup over first 100k env-steps).
#
# Output: checkpoints/stage2_hybrid_v2_N17/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Stage 2 v2 starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# Activate conda env (uncomment and adjust for your cluster)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_drl

mkdir -p logs

# -------------------------------------------------------------------
# Sanity: Stage 1 checkpoint must exist
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
echo "[$(date)] Launching Stage 2 v2 PPO trainer (hybrid, HK-1.5.8) ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage2_hybrid_v2.yaml \
    "$@"

echo "[$(date)] Stage 2 v2 done"
