#!/bin/bash
#SBATCH --job-name=decima_thermal
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00                  # 18h budget — REINFORCE is slow
#SBATCH --output=logs/decima_thermal_%j.out
#SBATCH --error=logs/decima_thermal_%j.err
# -------------------------------------------------------------------
# CPO v2 — Decima (Mao 2019 SIGCOMM) — thermal variant
#
# Variant 2 of 2: thermal-aware reward (SAME as Ours). Pair with
# train_decima_true_vanilla.sbatch for the full ablation cube.
#
# Output: checkpoints/decima_true_thermal_N17/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Decima true (thermal) starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# -------------------------------------------------------------------
# BLOCKED-marker guard: refuse to run while yaml is still a stub
# -------------------------------------------------------------------
DECIMA_YAML="cpo_thermal_v2/configs/train_decima_true_thermal.yaml"
if grep -q "^_BLOCKED_DO_NOT_RUN:" "$DECIMA_YAML"; then
    echo "❌ Decima true thermal yaml is marked BLOCKED — code support not yet implemented."
    echo "   See the yaml header."
    echo "   File: $DECIMA_YAML"
    exit 1
fi

# Activate conda env (uncomment and adjust for your cluster)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_drl

mkdir -p logs

# -------------------------------------------------------------------
# Train (REINFORCE — no warm-start; trains from scratch)
# -------------------------------------------------------------------
echo "[$(date)] Launching Decima true (thermal) REINFORCE trainer ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/train_decima_true_thermal.yaml \
    "$@"

echo "[$(date)] Decima true (thermal) done"
