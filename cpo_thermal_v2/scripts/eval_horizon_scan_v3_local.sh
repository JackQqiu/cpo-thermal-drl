#!/usr/bin/env bash
# =====================================================================
# eval_horizon_scan_v3_local.sh — N=1000 horizon scan
# =====================================================================
# IDENTICAL to eval_horizon_scan_local_parallel.sh except:
#   1. config:     eval_horizon_scan.yaml -> eval_horizon_scan_v3.yaml
#   2. output_dir: horizon_scan_v2 -> horizon_scan_v3
#   3. resume threshold: 1400 -> 4000  (since N=1000 not 200)
#   4. log file prefix: horizon_local -> horizon_v3
#
# Runs serially (one cell at a time) just like v2. Wall-clock estimate
# on Mac (~1.5 ep/s with 6 BLAS threads):
#   16 cells × 4 schedulers × 1000 ep ≈ 64,000 ep
#   ÷ ~1.5 ep/s = ~12 hours overnight
#
# Usage:
#   cd <REPO_ROOT>
#   bash cpo_thermal_v2/scripts/eval_horizon_scan_v3_local.sh
# =====================================================================

set -e
set -u
set -o pipefail

HALF_CORES="${HALF_CORES:-6}"

export OMP_NUM_THREADS="$HALF_CORES"
export MKL_NUM_THREADS="$HALF_CORES"
export OPENBLAS_NUM_THREADS="$HALF_CORES"
export VECLIB_MAXIMUM_THREADS="$HALF_CORES"
export NUMEXPR_NUM_THREADS="$HALF_CORES"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Local horizon-scan v3 starting (N=1000)"
echo "  cwd          = $(pwd)"
echo "  python       = $(which python)"
echo "  HALF_CORES   = $HALF_CORES"
echo "  PID          = $$"

EVAL_CKPT="checkpoints/stage2_hybrid_N17/best.pt"
if [ ! -f "$EVAL_CKPT" ]; then
    echo "❌ Stage 2 checkpoint not found at: $EVAL_CKPT"
    exit 1
fi
echo "  checkpoint   = $EVAL_CKPT"

mkdir -p logs

WORK_LIST=$(cat <<'EOF'
easy 30.0 45.0 20
easy 30.0 45.0 50
easy 30.0 45.0 100
easy 30.0 45.0 200
warm 40.0 55.0 20
warm 40.0 55.0 50
warm 40.0 55.0 100
warm 40.0 55.0 200
hot 50.0 65.0 20
hot 50.0 65.0 50
hot 50.0 65.0 100
hot 50.0 65.0 200
extreme 60.0 75.0 20
extreme 60.0 75.0 50
extreme 60.0 75.0 100
extreme 60.0 75.0 200
EOF
)

LOG_FILE="logs/horizon_v3_$(date +%Y%m%d_%H%M%S).log"
echo "  log          = $LOG_FILE"
echo

caffeinate -is bash <<BASH_EOF 2>&1 | tee "$LOG_FILE"
set -e
set -u
set -o pipefail
cd "$PROJ_ROOT"
export PYTHONPATH=".:\${PYTHONPATH:-}"
export OMP_NUM_THREADS="$HALF_CORES"
export MKL_NUM_THREADS="$HALF_CORES"
export OPENBLAS_NUM_THREADS="$HALF_CORES"
export VECLIB_MAXIMUM_THREADS="$HALF_CORES"
export NUMEXPR_NUM_THREADS="$HALF_CORES"

EVAL_CKPT="$EVAL_CKPT"

while read -r SETTING LOW HIGH H; do
    [ -z "\$SETTING" ] && continue

    OUT_DIR="eval_results/horizon_scan_v3/\${SETTING}/dags\${H}"

    echo
    echo "----------------------------------------------------------------------"
    echo "[\$(date)] [horizon-v3] Cell: setting=\${SETTING} horizon=\${H}"
    echo "  T0 in [\${LOW}, \${HIGH}]"
    echo "  output_dir = \${OUT_DIR}"
    echo "----------------------------------------------------------------------"

    # Resume detection: 4 schedulers × 1000 ep + 1 header ≈ 4000 rows
    if [ -f "\${OUT_DIR}/episodes.csv" ]; then
        n_rows=\$(wc -l < "\${OUT_DIR}/episodes.csv")
        if [ "\$n_rows" -ge 4000 ]; then
            echo "  OK existing episodes.csv has \$n_rows rows -- skipping"
            continue
        else
            echo "  WARN episodes.csv has only \$n_rows rows -- rerunning"
        fi
    fi

    python -u -m cpo_thermal_v2.evaluation.evaluate \\
        --config cpo_thermal_v2/configs/eval_horizon_scan_v3.yaml \\
        --override "eval.checkpoint_path=\${EVAL_CKPT}" \\
        --override "eval.initial_temp_range=[\${LOW},\${HIGH}]" \\
        --override "eval.dags_per_episode=\${H}" \\
        --override "eval.output_dir=\${OUT_DIR}" \\
        --override "eval.device=cpu"

    echo "[\$(date)] [horizon-v3] Cell \${SETTING}/dags\${H} done"
done <<'WORK_EOF'
$WORK_LIST
WORK_EOF
BASH_EOF

echo
echo "[$(date)] Horizon-scan v3 complete"
echo "Artifacts: eval_results/horizon_scan_v3/<setting>/dags<H>/"
