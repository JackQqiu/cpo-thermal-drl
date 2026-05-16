#!/bin/bash
#SBATCH --job-name=cpo_v2_eval_scaling
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                     # any GPU; CPU is fine for most baselines
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # 24h budget; estimated ~12-18h actual for 4 settings
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage E: Scaling-curve evaluation (4 operating settings)
#
# Drives the full evaluation grid for each of 4 thermal settings:
#     7 schedulers × 5 topology sizes × 500 episodes × 4 settings
# = 70,000 episodes total.
#
# Schedulers:
#   - RoundRobin       (sanity baseline)
#   - HEFT             (classic, thermally blind)
#   - ThermalHEFT      (PROCHOT-aware, strongest non-learning baseline)
#   - Decima           (Stage 1 ckpt with thermal features masked)
#   - Ours-auto_only   (our model in env-cool-only mode)
#   - Ours-agent_only  (our model controls delay; env-cool disabled)
#   - Ours-hybrid      (our model + env-cool both active)
#
# Settings (initial_temp_range):
#   - easy     [30, 45]
#   - warm     [40, 55]
#   - hot      [50, 65]   (default in eval_scaling.yaml)
#   - extreme  [60, 75]
#
# Each setting produces its own output subdirectory.
# Total output: eval_results/scaling_v2/{easy,warm,hot,extreme}/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Eval scaling starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  PYTHONPATH   = $PYTHONPATH"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# Activate conda env (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_rl

mkdir -p logs

# -------------------------------------------------------------------
# Pre-flight: Stage 2 checkpoint MUST exist
# -------------------------------------------------------------------
EVAL_CKPT="checkpoints/stage2_hybrid_N17/best.pt"
if [ ! -f "$EVAL_CKPT" ]; then
    echo "❌ Stage 2 checkpoint not found at: $EVAL_CKPT"
    echo "   Run scripts/train_stage2.sbatch first."
    exit 1
fi
echo "  checkpoint   = $EVAL_CKPT"

# RC matrices for all 5 sizes — need them for cross-size evaluation.
for N in 9 13 17 24 33; do
    if [ ! -f "data/thermal_matrics/N${N}/matrix_A.npy" ]; then
        echo "[$(date)] Generating RC matrices (missing N=${N}) ..."
        python -u -m cpo_thermal_v2.data_pipeline.generate_matrices \
            --output-root data/thermal_matrics/
        break
    fi
done

# -------------------------------------------------------------------
# Loop over 4 operating settings
# -------------------------------------------------------------------
declare -A SETTINGS=(
    [easy]="30.0,45.0"
    [warm]="40.0,55.0"
    [hot]="50.0,65.0"
    [extreme]="60.0,75.0"
)

for SETTING in easy warm hot extreme; do
    RANGE_CSV="${SETTINGS[$SETTING]}"
    LOW="${RANGE_CSV%,*}"
    HIGH="${RANGE_CSV#*,}"
    OUT_DIR="eval_results/scaling_v2/${SETTING}"

    echo
    echo "======================================================================"
    echo "[$(date)] Setting: ${SETTING}  T0 ∈ [${LOW}, ${HIGH}]"
    echo "  output_dir = ${OUT_DIR}"
    echo "======================================================================"

    python -u -m cpo_thermal_v2.evaluation.evaluate \
        --config cpo_thermal_v2/configs/eval_scaling.yaml \
        --override "eval.checkpoint_path=${EVAL_CKPT}" \
        --override "eval.initial_temp_range=[${LOW},${HIGH}]" \
        --override "eval.output_dir=${OUT_DIR}" \
        "$@"

    echo "[$(date)] Setting ${SETTING} done -> ${OUT_DIR}"
done

echo
echo "[$(date)] All 4 settings done"
echo
echo "Artifacts:"
echo "  CSVs:   eval_results/scaling_v2/{easy,warm,hot,extreme}/episodes.csv"
echo "  Plots:  eval_results/scaling_v2/{easy,warm,hot,extreme}/plots/"
echo "  Tables: eval_results/scaling_v2/{easy,warm,hot,extreme}/tables/"
