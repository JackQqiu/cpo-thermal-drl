#!/bin/bash
#SBATCH --job-name=cpo_v2_horizon_scan
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00                 # 12h budget; estimated ~9-10h actual
#SBATCH --output=logs/horizon_%j.out
#SBATCH --error=logs/horizon_%j.err
# -------------------------------------------------------------------
# CPO v2 — Stage E supplementary: Horizon generalization scan
#
# Goal: demonstrate that Stage 2 (hybrid) maintains horizon-invariant
# safety while Stage 1 (auto_only) violations grow linearly with
# episode length.
#
# Grid:
#   schedulers (7) × N (1) × episodes (200) × horizons (4) × settings (4)
# = 22,400 episodes total
#
# Horizons:
#   20  DAGs   = training distribution (in-distribution sanity check)
#   50  DAGs   = mild OOD — Stage 1 starts to leak violations
#   100 DAGs   = moderate OOD — gap should be ~0.5 violations
#   200 DAGs   = strong OOD — extreme generalization test
#
# Settings (initial_temp_range):
#   easy    [30, 45]
#   warm    [40, 55]
#   hot     [50, 65]
#   extreme [60, 75]
#
# Output: eval_results/horizon_scan_v2/<setting>/dags<H>/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Horizon scan starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"

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
    exit 1
fi
echo "  checkpoint   = $EVAL_CKPT"

# -------------------------------------------------------------------
# 4 settings × 4 horizons = 16 cells
# -------------------------------------------------------------------
declare -A SETTINGS=(
    [easy]="30.0,45.0"
    [warm]="40.0,55.0"
    [hot]="50.0,65.0"
    [extreme]="60.0,75.0"
)
HORIZONS=(20 50 100 200)

for SETTING in easy warm hot extreme; do
    RANGE_CSV="${SETTINGS[$SETTING]}"
    LOW="${RANGE_CSV%,*}"
    HIGH="${RANGE_CSV#*,}"

    for H in "${HORIZONS[@]}"; do
        OUT_DIR="eval_results/horizon_scan_v2/${SETTING}/dags${H}"

        echo
        echo "======================================================================"
        echo "[$(date)] Cell: setting=${SETTING} horizon=${H}"
        echo "  T0 ∈ [${LOW}, ${HIGH}]"
        echo "  output_dir = ${OUT_DIR}"
        echo "======================================================================"

        # Skip if already done (resumable)
        if [ -f "${OUT_DIR}/episodes.csv" ]; then
            echo "  ✓ existing episodes.csv found — skipping (delete to rerun)"
            continue
        fi

        python -u -m cpo_thermal_v2.evaluation.evaluate \
            --config cpo_thermal_v2/configs/eval_horizon_scan.yaml \
            --override "eval.checkpoint_path=${EVAL_CKPT}" \
            --override "eval.initial_temp_range=[${LOW},${HIGH}]" \
            --override "eval.dags_per_episode=${H}" \
            --override "eval.output_dir=${OUT_DIR}" \
            "$@"

        echo "[$(date)] Cell ${SETTING}/dags${H} done"
    done
done

echo
echo "[$(date)] Horizon scan complete"
echo
echo "Artifacts (16 cells):"
echo "  eval_results/horizon_scan_v2/<setting>/dags<H>/episodes.csv"
echo
echo "To assemble Section 5.3 paper figure, see"
echo "  cpo_thermal_v2/scripts/aggregate_horizon_scan.py  (TODO)"