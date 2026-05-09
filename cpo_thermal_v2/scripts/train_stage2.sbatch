#!/bin/bash
#SBATCH --job-name=cpo_v2_stage2_hybrid
#SBATCH --account=yanjia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # 24h budget (Stage 2 ~14-18h)
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage 2 (hybrid fine-tune from Stage 1 warm-start)
#
# Loads checkpoints/stage1_auto_only_N17/best.pt:
#   - encoder weights  → loaded
#   - placement head   → loaded
#   - V_placement      → loaded
#   - delay head       → fresh init (was disabled gradient in Stage 1)
#   - V_delay          → fresh init
#
# Then trains the full factored agent with dual-advantage credit
# assignment.  Output: checkpoints/stage2_hybrid_N17/.
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Stage 2 starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# Activate conda env
source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

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
echo "[$(date)] Launching Stage 2 PPO trainer (hybrid) ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage2_hybrid.yaml \
    "$@"

echo "[$(date)] Stage 2 done"
