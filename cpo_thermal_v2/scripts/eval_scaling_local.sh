#!/usr/bin/env bash
# =====================================================================
# eval_scaling_local.sh — Run Stage E main table locally on Mac (M-series)
# =====================================================================
# Mirrors scripts/eval_scaling.sbatch but as a plain shell loop so it
# runs on a developer machine (no SLURM).  Uses `caffeinate` to keep
# the Mac awake during the long run.
#
# Estimated runtime on M3 Pro (CPU): ~12-18h
# Estimated runtime on M3 Max:       ~10-14h
#
# Usage:
#   cd <REPO_ROOT>
#   bash cpo_thermal_v2/scripts/eval_scaling_local.sh
#
# To resume after interruption: re-run the same command.  Already-
# completed (setting, ...) cells are detected by an existing
# eval_results/scaling_v2/<setting>/episodes.csv and skipped.
# =====================================================================

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Local eval starting"
echo "  cwd        = $(pwd)"
echo "  PYTHONPATH = $PYTHONPATH"
echo "  python     = $(which python)"

EVAL_CKPT="checkpoints/stage2_hybrid_N17/best.pt"
if [ ! -f "$EVAL_CKPT" ]; then
    echo "❌ Stage 2 checkpoint not found at: $EVAL_CKPT"
    exit 1
fi
echo "  checkpoint = $EVAL_CKPT"

mkdir -p logs

# ---------------------------------------------------------------------
# Work list: one setting per line: "setting low high"
# Plain text — survives caffeinate subshell trivially.
# ---------------------------------------------------------------------
WORK_LIST=$(cat <<'EOF'
easy 30.0 45.0
warm 40.0 55.0
hot 50.0 65.0
extreme 60.0 75.0
EOF
)

LOG_FILE="logs/eval_local_$(date +%Y%m%d_%H%M%S).log"
echo "  log        = $LOG_FILE"
echo

# ---------------------------------------------------------------------
# caffeinate -dis  =  prevent display sleep + idle sleep + system sleep
# Run sub-bash; tee both stdout+stderr to log.
# ---------------------------------------------------------------------
caffeinate -dis bash <<BASH_EOF 2>&1 | tee "$LOG_FILE"
set -e
set -u
set -o pipefail
cd "$PROJ_ROOT"
export PYTHONPATH=".:\${PYTHONPATH:-}"

EVAL_CKPT="$EVAL_CKPT"

while read -r SETTING LOW HIGH; do
    [ -z "\$SETTING" ] && continue

    OUT_DIR="eval_results/scaling_v2/\${SETTING}"

    echo
    echo "======================================================================"
    echo "[\$(date)] Setting: \${SETTING}  T0 in [\${LOW}, \${HIGH}]"
    echo "  output_dir = \${OUT_DIR}"
    echo "======================================================================"

    if [ -f "\${OUT_DIR}/episodes.csv" ]; then
        n_rows=\$(wc -l < "\${OUT_DIR}/episodes.csv")
        # Main table cell expects 7 schedulers × 5 N × 500 eps ≈ 17,500 rows
        if [ "\$n_rows" -ge 17000 ]; then
            echo "  OK existing episodes.csv has \$n_rows rows -- skipping"
            continue
        else
            echo "  WARN episodes.csv has only \$n_rows rows -- rerunning"
        fi
    fi

    python -u -m cpo_thermal_v2.evaluation.evaluate \\
        --config cpo_thermal_v2/configs/eval_scaling.yaml \\
        --override "eval.checkpoint_path=\${EVAL_CKPT}" \\
        --override "eval.initial_temp_range=[\${LOW},\${HIGH}]" \\
        --override "eval.output_dir=\${OUT_DIR}" \\
        --override "eval.device=cpu"

    echo "[\$(date)] Setting \${SETTING} done"
done <<'WORK_EOF'
$WORK_LIST
WORK_EOF
BASH_EOF

echo
echo "[$(date)] All 4 settings complete"
echo "Artifacts: eval_results/scaling_v2/{easy,warm,hot,extreme}/"
