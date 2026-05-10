#!/bin/bash
#SBATCH --job-name=stage2_v3_stress
#SBATCH --account=yanjia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00                  # 8h budget (Stage 2 v3 stress ~6-8h on N=33)
#SBATCH --output=logs/stage2_v3_%j.out
#SBATCH --error=logs/stage2_v3_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage 2 v3 (HK-1.5.9 stress curriculum, N=33)
#
# Loads checkpoints/stage1_auto_only_N17/best.pt as warm start
# (warm_start_strict=false; encoder + GNN transfer, actor heads reinit
# because N differs).  Trains the hybrid agent under HK-1.5.8 reward +
# trainer (gated anticipation_bonus, delay_loss_coef linear-warmup) and
# a stress curriculum that pushes training peak_T into the T_pen
# trigger zone so the delay channel receives dense signal.
#
# Output: checkpoints/stage2_hybrid_v3_stress_N33/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Stage 2 v3 stress starting on node $(hostname)"
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
echo "[$(date)] Launching Stage 2 v3 stress PPO trainer (hybrid, HK-1.5.9, N=33) ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage2_hybrid_v3_stress.yaml \
    "$@"

echo "[$(date)] Stage 2 v3 stress done"
