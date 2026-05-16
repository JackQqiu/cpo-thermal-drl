#!/usr/bin/env bash
# =====================================================================
# eval_horizon_scan_v3_local_parallel.sh — LOCAL MAC, N=1000
# =====================================================================
# This version avoids the [lo,hi] CLI-quoting bug by using one yaml
# per ambient. The shell only overrides scalar values (output_dir,
# dags_per_episode), never arrays.
#
# Usage:
#   cd <REPO_ROOT>
#   caffeinate -is bash cpo_thermal_v2/scripts/eval_horizon_scan_v3_local_parallel.sh
# =====================================================================

set -e
set -u
set -o pipefail

HALF_CORES="${HALF_CORES:-4}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"

export OMP_NUM_THREADS="$HALF_CORES"
export MKL_NUM_THREADS="$HALF_CORES"
export OPENBLAS_NUM_THREADS="$HALF_CORES"
export VECLIB_MAXIMUM_THREADS="$HALF_CORES"
export NUMEXPR_NUM_THREADS="$HALF_CORES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"

mkdir -p logs eval_results/horizon_scan_v3

LOG_FILE="logs/horizon_v3_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Local horizon-scan v3 (N=1000) starting"
echo "  cwd            = $(pwd)"
echo "  python         = $(which python)"
echo "  HALF_CORES     = $HALF_CORES"
echo "  MAX_CONCURRENT = $MAX_CONCURRENT"
echo "  log            = $LOG_FILE"

# 4 ambients × 4 horizons = 16 cells.  Format: "<setting> <horizon>"
WORK_LIST=(
    "easy 20"    "easy 50"    "easy 100"    "easy 200"
    "warm 20"    "warm 50"    "warm 100"    "warm 200"
    "hot 20"     "hot 50"     "hot 100"     "hot 200"
    "extreme 20" "extreme 50" "extreme 100" "extreme 200"
)

run_one_cell() {
    local SETTING="$1"
    local H="$2"

    local YAML="cpo_thermal_v2/configs/eval_horizon_scan_v3_${SETTING}.yaml"
    local OUT_DIR="eval_results/horizon_scan_v3/${SETTING}/dags${H}"
    local CELL_LOG="logs/horizon_v3_${SETTING}_dags${H}_$(date +%Y%m%d).log"

    if [ ! -f "$YAML" ]; then
        echo "[$(date)] [FAIL] missing config: $YAML"
        return 1
    fi

    # Resume detection: 4 schedulers × 1000 ep + 1 header = 4001 rows
    if [ -f "${OUT_DIR}/episodes.csv" ]; then
        local n_rows
        n_rows=$(wc -l < "${OUT_DIR}/episodes.csv")
        if [ "$n_rows" -ge 4000 ]; then
            echo "[$(date)] [SKIP] ${SETTING}/dags${H}: $n_rows rows already present"
            return 0
        fi
    fi

    echo "[$(date)] [START] ${SETTING}/dags${H}  (yaml=$YAML)"

    # Only scalar overrides — no arrays!  initial_temp_range comes
    # from the per-ambient yaml directly.
    python -u -m cpo_thermal_v2.evaluation.evaluate \
        --config "$YAML" \
        --override "eval.dags_per_episode=${H}" \
        --override "eval.output_dir=${OUT_DIR}" \
        > "${CELL_LOG}" 2>&1

    local RC=$?
    if [ "$RC" -ne 0 ]; then
        echo "[$(date)] [FAIL]  ${SETTING}/dags${H} (rc=$RC) — see ${CELL_LOG}"
        return $RC
    fi
    echo "[$(date)] [DONE]  ${SETTING}/dags${H}"
}

export -f run_one_cell

START_T=$(date +%s)

printf '%s\n' "${WORK_LIST[@]}" | xargs -n 1 -P "$MAX_CONCURRENT" -I{} bash -c '
    line="{}"
    set -- $line
    run_one_cell "$1" "$2"
' 2>&1 | tee "$LOG_FILE"

END_T=$(date +%s)
ELAPSED=$((END_T - START_T))

echo
echo "[$(date)] Horizon-scan v3 complete"
echo "  total wall-clock: $((ELAPSED / 3600))h $(( (ELAPSED % 3600) / 60 ))m"
echo "Artifacts: eval_results/horizon_scan_v3/<setting>/dags<H>/"
