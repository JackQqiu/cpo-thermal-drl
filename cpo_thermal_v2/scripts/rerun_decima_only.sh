#!/usr/bin/env bash
# =====================================================================
# rerun_decima_only.sh — Replace Decima rows in existing eval CSVs
# =====================================================================
# After fair Decima training completes, run this to:
#   1. Re-evaluate Decima cells using the new fair Decima ckpt
#   2. Merge new Decima rows into existing episodes.csv files
#      (replacing the old broken Decima rows; other 6 schedulers stay)
#
# Prerequisites:
#   - checkpoints/decima_fair_N17/best.pt exists
#   - eval_results/scaling_v2/{easy,warm,hot,extreme}/episodes.csv exist
#   - eval_results/horizon_scan_v2/<setting>/dags<H>/episodes.csv exist
#
# Usage:
#   bash cpo_thermal_v2/scripts/rerun_decima_only.sh
#
# Estimated runtime on M3 Pro (CPU): ~2-3h for main + horizon
#                  on V100 (cuda):   ~1-1.5h
# =====================================================================

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cpu}"
DECIMA_FAIR_CKPT="checkpoints/decima_fair_N17/best.pt"

if [ ! -f "$DECIMA_FAIR_CKPT" ]; then
    echo "❌ Fair Decima ckpt not found at: $DECIMA_FAIR_CKPT"
    echo "   Train it first with scripts/train_decima_fair.sbatch"
    exit 1
fi

echo "[$(date)] Rerun-Decima-only starting"
echo "  cwd               = $(pwd)"
echo "  device            = $DEVICE  (override with DEVICE=cuda:0)"
echo "  decima_fair_ckpt  = $DECIMA_FAIR_CKPT"
echo

mkdir -p logs eval_results/_decima_rerun

LOG_FILE="logs/rerun_decima_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------
# Phase 1: Main table (4 settings, only Decima column)
# ---------------------------------------------------------------------
echo
echo "######################################################################"
echo "# Phase 1: Main table — re-running Decima only @ 4 settings           "
echo "######################################################################"

for SETTING in easy warm hot extreme; do
    case "$SETTING" in
        easy)    LOW=30.0; HIGH=45.0 ;;
        warm)    LOW=40.0; HIGH=55.0 ;;
        hot)     LOW=50.0; HIGH=65.0 ;;
        extreme) LOW=60.0; HIGH=75.0 ;;
    esac

    OUT_DIR="eval_results/_decima_rerun/scaling_v2/${SETTING}"
    EXISTING_CSV="eval_results/scaling_v2/${SETTING}/episodes.csv"

    if [ ! -f "$EXISTING_CSV" ]; then
        echo "  ⚠ ${EXISTING_CSV} missing — skipping ${SETTING}"
        continue
    fi

    echo
    echo "[$(date)] Setting: ${SETTING}  T0 in [${LOW}, ${HIGH}]"

    python -u -m cpo_thermal_v2.evaluation.evaluate \
        --config cpo_thermal_v2/configs/eval_scaling.yaml \
        --override "eval.checkpoint_path=checkpoints/stage2_hybrid_N17/best.pt" \
        --override "eval.decima_fair_ckpt=${DECIMA_FAIR_CKPT}" \
        --override "eval.scheduler_filter=[Decima]" \
        --override "eval.initial_temp_range=[${LOW},${HIGH}]" \
        --override "eval.output_dir=${OUT_DIR}" \
        --override "eval.device=${DEVICE}" \
        2>&1 | tee -a "$LOG_FILE"
done

# ---------------------------------------------------------------------
# Phase 2: Horizon scan (4 settings × 4 horizons, only Decima)
# ---------------------------------------------------------------------
echo
echo "######################################################################"
echo "# Phase 2: Horizon scan — re-running Decima only @ 16 cells           "
echo "######################################################################"

for SETTING in easy warm hot extreme; do
    case "$SETTING" in
        easy)    LOW=30.0; HIGH=45.0 ;;
        warm)    LOW=40.0; HIGH=55.0 ;;
        hot)     LOW=50.0; HIGH=65.0 ;;
        extreme) LOW=60.0; HIGH=75.0 ;;
    esac

    for H in 20 50 100 200; do
        OUT_DIR="eval_results/_decima_rerun/horizon_scan_v2/${SETTING}/dags${H}"
        EXISTING_CSV="eval_results/horizon_scan_v2/${SETTING}/dags${H}/episodes.csv"

        if [ ! -f "$EXISTING_CSV" ]; then
            echo "  ⚠ ${EXISTING_CSV} missing — skipping ${SETTING}/dags${H}"
            continue
        fi

        echo
        echo "[$(date)] horizon: setting=${SETTING} dags=${H}"

        python -u -m cpo_thermal_v2.evaluation.evaluate \
            --config cpo_thermal_v2/configs/eval_horizon_scan.yaml \
            --override "eval.checkpoint_path=checkpoints/stage2_hybrid_N17/best.pt" \
            --override "eval.decima_fair_ckpt=${DECIMA_FAIR_CKPT}" \
            --override "eval.scheduler_filter=[Decima]" \
            --override "eval.initial_temp_range=[${LOW},${HIGH}]" \
            --override "eval.dags_per_episode=${H}" \
            --override "eval.output_dir=${OUT_DIR}" \
            --override "eval.device=${DEVICE}" \
            2>&1 | tee -a "$LOG_FILE"
    done
done

# ---------------------------------------------------------------------
# Phase 3: Merge — replace old Decima rows in existing csv with new ones
# ---------------------------------------------------------------------
echo
echo "######################################################################"
echo "# Phase 3: Merging fair Decima data into existing episodes.csv        "
echo "######################################################################"

python -u cpo_thermal_v2/scripts/merge_decima_results.py \
    --rerun-root eval_results/_decima_rerun \
    --target-roots eval_results/scaling_v2 eval_results/horizon_scan_v2 \
    2>&1 | tee -a "$LOG_FILE"

echo
echo "[$(date)] Decima rerun + merge complete"
echo
echo "Backup of original Decima rows is in:"
echo "  eval_results/<dir>/episodes.csv.before_decima_fair"
echo
echo "Next: rerun verify_eval_complete.sh to check the new Decima numbers."
