#!/bin/bash
#SBATCH --job-name=cpo_decima_fair
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00          # ~4h actual + buffer
#SBATCH --output=logs/decima_fair_%j.out
#SBATCH --error=logs/decima_fair_%j.err
# -------------------------------------------------------------------
# Fair Decima training — the from-scratch baseline that NEVER sees
# thermal features (input or edge attr).
#
# Output: checkpoints/decima_fair_N17/best.pt
# After this finishes, re-run Stage E with Decima column updated to
# use the new checkpoint.
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Fair Decima training starting on $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Activate conda env (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_rl

mkdir -p logs checkpoints/decima_fair_N17 tb_logs/decima_fair_N17

# Verify RC matrices exist
if [ ! -f "data/thermal_matrics/N17/matrix_A.npy" ]; then
    echo "[$(date)] Generating RC matrices ..."
    python -u -m cpo_thermal_v2.data_pipeline.generate_matrices \
        --output-root data/thermal_matrics/
fi

# Run training
python -u -m cpo_thermal_v2.scripts.train_decima_fair \
    --config cpo_thermal_v2/configs/train_decima_fair.yaml

echo
echo "[$(date)] Fair Decima training complete"
ls -lh checkpoints/decima_fair_N17/
