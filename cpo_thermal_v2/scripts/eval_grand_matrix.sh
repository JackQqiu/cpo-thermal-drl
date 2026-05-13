#!/usr/bin/env bash
# =====================================================================
# eval_grand_matrix.sh — Phase X-B full ambient sweep driver
# =====================================================================
# Runs the 4-ambient grand matrix sequentially:
#   - hot:     ambient [60, 75] °C  (paper main-claim regime)
#   - cold:    ambient [25, 40] °C  (cold start)
#   - warm:    ambient [40, 65] °C  (middle regime)
#   - extreme: ambient [70, 80] °C  (bounded-claim panel; T_0 above T_target=75)
#
# Each ambient is 10 schedulers × 5 N × 500 ep = 25,000 ep.
# Total 100,000 ep across all 4 ambients.
#
# Mac CPU wallclock estimate: ~20h (will be calibrated by Phase X-A first).
# If too slow, sbatch to V100/A100 for ~7-10× speedup.
#
# Output structure:
#   eval_results/grand_matrix_hot/episodes.csv      (paper main)
#   eval_results/grand_matrix_cold/episodes.csv     (scaling fig)
#   eval_results/grand_matrix_warm/episodes.csv     (scaling fig)
#   eval_results/grand_matrix_extreme/episodes.csv  (bounded-claim panel)
#
# Phase G aggregator joins all 4 CSVs (paired by seed_base=100000) for
# Wilcoxon + Holm-Bonferroni stats and paper §5 figure data.
# =====================================================================

set -e
set -u
set -o pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=".:${PYTHONPATH:-}"

START_TS=$(date +%s)
echo "[$(date)] Phase X-B grand matrix driver starting"

# Order: HOT first (most paper-critical), then cold/warm/extreme.
# If you interrupt midway, the CSVs from completed ambients are
# preserved — just re-run with the ambients you haven't done yet.
for amb in hot cold warm extreme; do
    yaml="cpo_thermal_v2/configs/eval_grand_matrix_${amb}.yaml"
    out="eval_results/grand_matrix_${amb}"
    echo ""
    echo "============================================================"
    echo "[$(date)] X-B ambient = $amb  yaml=$yaml  -> $out"
    echo "============================================================"
    if [ -f "$out/episodes.csv" ]; then
        echo "[skip] $out/episodes.csv already exists; remove to re-run"
        continue
    fi
    python -u -m cpo_thermal_v2.evaluation.evaluate --config "$yaml"
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "[$(date)] Phase X-B done in $((ELAPSED / 60)) min ($((ELAPSED / 3600)) h)"
