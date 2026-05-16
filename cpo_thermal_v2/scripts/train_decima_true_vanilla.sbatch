#!/bin/bash
#SBATCH --job-name=decima_vanilla
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00                  # 18h budget — REINFORCE is slow
#SBATCH --output=logs/decima_vanilla_%j.out
#SBATCH --error=logs/decima_vanilla_%j.err
# -------------------------------------------------------------------
# CPO v2 — Decima (Mao 2019 SIGCOMM) — vanilla variant
#
# !!!! BLOCKED — DO NOT SUBMIT YET !!!!
# This sbatch references configs/train_decima_true_vanilla.yaml +
# uses baselines/decima_true.py, both of which are currently STUBS.
# See paper_drafts/decima_true_implementation_checklist.md for the
# 10-step implementation plan (~5-7 days work).
#
# Variant 1 of 2: makespan-only reward (Mao 2019 faithful). Pair with
# train_decima_true_thermal.sbatch for the full ablation cube.
#
# Train order: vanilla FIRST (faster signal — if vanilla doesn't
# converge at all, thermal won't either, save 18h).
#
# Once code support is implemented:
#   1. Implement decima_true.py per checklist
#   2. Wire algorithm dispatch + architecture registry per yaml header
#   3. Remove the `_BLOCKED_DO_NOT_RUN:` line from the yaml
#   4. Resubmit this sbatch
#
# Output: checkpoints/decima_true_vanilla_N17/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Decima true (vanilla) starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# -------------------------------------------------------------------
# BLOCKED-marker guard: refuse to run while yaml is still a stub
# -------------------------------------------------------------------
DECIMA_YAML="cpo_thermal_v2/configs/train_decima_true_vanilla.yaml"
if grep -q "^_BLOCKED_DO_NOT_RUN:" "$DECIMA_YAML"; then
    echo "❌ Decima true vanilla yaml is marked BLOCKED — code support not yet implemented."
    echo "   See yaml header + paper_drafts/decima_true_implementation_checklist.md"
    echo "   File: $DECIMA_YAML"
    exit 1
fi

# Activate conda env
source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

# -------------------------------------------------------------------
# Train (REINFORCE — no warm-start; trains from scratch)
# -------------------------------------------------------------------
echo "[$(date)] Launching Decima true (vanilla) REINFORCE trainer ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/train_decima_true_vanilla.yaml \
    "$@"

echo "[$(date)] Decima true (vanilla) done"
