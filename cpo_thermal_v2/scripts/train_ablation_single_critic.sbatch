#!/bin/bash
#SBATCH --job-name=cpo_v2_ablation_single_critic
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation_single_critic_%j.out
#SBATCH --error=logs/ablation_single_critic_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage 2 ablation: single critic head
#
# !!!! BLOCKED — DO NOT SUBMIT YET !!!!
# This sbatch references configs/ablation_single_critic.yaml, which
# is currently a STUB.  See yaml header for the required code
# changes.  The runtime guard below aborts the job before any
# training starts if the BLOCKED marker is present in the yaml.
#
# Once code support is implemented:
#   1. Verify SingleCritic + dual_critic flag plumbing per yaml header
#   2. Remove the `_BLOCKED_DO_NOT_RUN:` line from the yaml
#   3. Resubmit this sbatch
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Stage 2 ablation (single_critic) starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# -------------------------------------------------------------------
# BLOCKED-marker guard: refuse to run if yaml is still a stub
# -------------------------------------------------------------------
ABLATION_YAML="cpo_thermal_v2/configs/ablation_single_critic.yaml"
if grep -q "^_BLOCKED_DO_NOT_RUN:" "$ABLATION_YAML"; then
    echo "❌ Ablation yaml is marked BLOCKED — code support not yet implemented."
    echo "   See yaml header for required code changes."
    echo "   File: $ABLATION_YAML"
    exit 1
fi

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
echo "[$(date)] Launching ablation (single_critic) PPO trainer ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/ablation_single_critic.yaml \
    "$@"

echo "[$(date)] Stage 2 ablation (single_critic) done"
