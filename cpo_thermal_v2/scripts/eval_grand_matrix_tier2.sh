#!/usr/bin/env bash
# =====================================================================
# eval_grand_matrix_tier2.sh — Phase X-B Tier 2 driver (cold + warm + extreme)
# =====================================================================
# Phase X-B Tier 2 full ambient sweep: 3 ambient × 2 yamls (main + extras)
# × 5 N × 500 ep ≈ 90,000 ep total, est ~15h Mac CPU.
#
# HOT slice already done (eval_results/grand_matrix_hot/ +
# eval_results/grand_matrix_hot_extras/).
#
# Each yaml is roughly:
# - main (8 sched × 5 N × 500 ep = 20k ep)   ~3h
# - extras (4 sched × 5 N × 500 ep × varies) ~2h
# Per ambient: ~5h.  3 ambients × 5h = ~15h.
#
# Ambient order: cold → warm → extreme (paper-relevance ascending).
# Skips cells whose output already exists.
# =====================================================================

set -e
set -u
set -o pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=".:${PYTHONPATH:-}"

START_TS=$(date +%s)
echo "[$(date)] Phase X-B Tier 2 starting"

for amb in cold warm extreme; do
    for variant in "" "_extras"; do
        yaml="cpo_thermal_v2/configs/eval_grand_matrix_${amb}${variant}.yaml"
        out="eval_results/grand_matrix_${amb}${variant}"
        echo ""
        echo "============================================================"
        echo "[$(date)] ambient=${amb} variant=${variant:-main} yaml=$yaml -> $out"
        echo "============================================================"
        if [ -f "$out/episodes.csv" ]; then
            echo "[skip] $out/episodes.csv already exists; remove to re-run"
            continue
        fi
        python -u -m cpo_thermal_v2.evaluation.evaluate --config "$yaml"
    done
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "[$(date)] Phase X-B Tier 2 done in $((ELAPSED / 60)) min ($(awk -v s=$ELAPSED 'BEGIN{printf "%.1fh", s/3600}'))"
