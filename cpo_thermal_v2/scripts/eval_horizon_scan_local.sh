#!/usr/bin/env bash
# =====================================================================
# eval_horizon_scan_local.sh — Run horizon scan locally on Mac
# =====================================================================
# Mirrors scripts/eval_horizon_scan.sbatch as a plain shell loop.
#
# Grid:  4 settings × 4 horizons × 7 schedulers × 1 N (17) × 200 episodes
#      = 22,400 episodes total
#
# Estimated runtime on M3 Pro (CPU): ~9-11h
# Estimated runtime on M3 Max:       ~7-9h
#
# Usage:
#   cd <REPO_ROOT>
#   bash cpo_thermal_v2/scripts/eval_horizon_scan_local.sh
#
# Resumable: cells with an existing episodes.csv are skipped.
# =====================================================================

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Local horizon scan starting"
echo "  cwd        = $(pwd)"
echo "  python     = $(which python)"

EVAL_CKPT="checkpoints/stage2_hybrid_N17/best.pt"
if [ ! -f "$EVAL_CKPT" ]; then
    echo "❌ Stage 2 checkpoint not found at: $EVAL_CKPT"
    exit 1
fi
echo "  checkpoint = $EVAL_CKPT"

mkdir -p logs

# ---------------------------------------------------------------------
# Build the work list as one cell per line: "setting low high horizon"
# This is portable: no associative arrays, no declare -p tricks.
# ---------------------------------------------------------------------
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

LOG_FILE="logs/horizon_local_$(date +%Y%m%d_%H%M%S).log"
echo "  log        = $LOG_FILE"
echo

# ---------------------------------------------------------------------
# Run the loop INSIDE caffeinate so the Mac stays awake.
# caffeinate -dis  =  prevent display sleep + idle sleep + system sleep
# We run a sub-bash and pipe both stdout+stderr through tee.
# ---------------------------------------------------------------------
caffeinate -dis bash <<BASH_EOF 2>&1 | tee "$LOG_FILE"
set -e
set -u
set -o pipefail
cd "$PROJ_ROOT"
export PYTHONPATH=".:\${PYTHONPATH:-}"

EVAL_CKPT="$EVAL_CKPT"

while read -r SETTING LOW HIGH H; do
    # skip blank lines just in case
    [ -z "\$SETTING" ] && continue

    OUT_DIR="eval_results/horizon_scan_v2/\${SETTING}/dags\${H}"

    echo
    echo "----------------------------------------------------------------------"
    echo "[\$(date)] Cell: setting=\${SETTING} horizon=\${H}"
    echo "  T0 in [\${LOW}, \${HIGH}]"
    echo "  output_dir = \${OUT_DIR}"
    echo "----------------------------------------------------------------------"

    if [ -f "\${OUT_DIR}/episodes.csv" ]; then
        n_rows=\$(wc -l < "\${OUT_DIR}/episodes.csv")
        # Each cell expects 7 schedulers × 200 eps = 1400 + 1 header
        if [ "\$n_rows" -ge 1400 ]; then
            echo "  OK existing episodes.csv has \$n_rows rows -- skipping"
            continue
        else
            echo "  WARN episodes.csv has only \$n_rows rows -- rerunning"
        fi
    fi

    python -u -m cpo_thermal_v2.evaluation.evaluate \\
        --config cpo_thermal_v2/configs/eval_horizon_scan.yaml \\
        --override "eval.checkpoint_path=\${EVAL_CKPT}" \\
        --override "eval.initial_temp_range=[\${LOW},\${HIGH}]" \\
        --override "eval.dags_per_episode=\${H}" \\
        --override "eval.output_dir=\${OUT_DIR}" \\
        --override "eval.device=cpu"

    echo "[\$(date)] Cell \${SETTING}/dags\${H} done"
done <<'WORK_EOF'
$WORK_LIST
WORK_EOF
BASH_EOF

echo
echo "[$(date)] All 16 horizon cells complete"
echo "Artifacts: eval_results/horizon_scan_v2/<setting>/dags<H>/"