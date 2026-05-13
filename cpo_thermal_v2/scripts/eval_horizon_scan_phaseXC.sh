#!/usr/bin/env bash
# Phase X-C horizon scan driver.  Calls eval_horizon_scan_phaseXC.yaml
# 4 times with --override eval.dags_per_episode=H for H in {20,50,100,200}.
set -e
set -u
set -o pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=".:${PYTHONPATH:-}"

START_TS=$(date +%s)
for H in 20 50 100 200; do
    out="eval_results/horizon_scan_phaseXC/h${H}"
    echo ""
    echo "============================================================"
    echo "[$(date)] X-C horizon = $H DAGs/ep  -> $out"
    echo "============================================================"
    if [ -f "$out/episodes.csv" ]; then
        echo "[skip] $out/episodes.csv already exists"
        continue
    fi
    python -u -m cpo_thermal_v2.evaluation.evaluate \
        --config cpo_thermal_v2/configs/eval_horizon_scan_phaseXC.yaml \
        --override "eval.dags_per_episode=${H}" \
        --override "eval.output_dir=${out}"
done
echo "[$(date)] X-C done in $(( ($(date +%s) - START_TS) / 60 )) min"
