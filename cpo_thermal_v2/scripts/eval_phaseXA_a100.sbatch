#!/bin/bash
#SBATCH --job-name=eval_phaseXA
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=a100x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_phaseXA_%j.out
#SBATCH --error=logs/eval_phaseXA_%j.err
# -------------------------------------------------------------------
# CPO v2 — Phase X-A: D2 sanity 6-way eval (A100 timing baseline)
#
# Drives the decision between Mac CPU (estimated ~20h for X-B+X-C)
# vs A100 server (estimated ~2-3h elapsed).  This sbatch runs the
# 6-way × N=17 × HOT × 50 ep sanity (300 ep total) and reports
# elapsed time.
#
# Mac CPU reference (2026-05-13, Phase X-A run):
#   170 seconds (300 ep) → 0.57 sec/(sched, ep) avg
#
# A100 expected: 5-15× faster for neural scheduler forwards (Decima-v/t,
# HGATE-PPO, D2, Ours-auto_only), but classical schedulers are CPU-bound
# and won't speed up much.  Realistic gain: ~3-5× overall.
#
# If A100 result is good, swap eval_grand_matrix.yaml + sibling configs
# to device=cuda (or write _cuda.yaml variants) and sbatch X-B/X-C.
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

# NFS pre-warm
cat data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || \
    cat cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || true

echo "[$(date)] Phase X-A starting on $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  GPU = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs = $SLURM_CPUS_PER_TASK"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_drl

mkdir -p logs

START_TS=$(date +%s)

python -u -m cpo_thermal_v2.evaluation.evaluate \
    --config cpo_thermal_v2/configs/eval_main_6way_hot_n17_cuda.yaml

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "============================================================"
echo "[$(date)] Phase X-A A100 elapsed: ${ELAPSED}s ($(( ELAPSED / 60 )) min)"
echo "  Mac CPU reference (300 ep): 170s"
echo "  Speedup vs Mac: $(awk -v a=170 -v b=${ELAPSED} 'BEGIN{printf "%.1fx", a/b}')"
echo "  Per-(sched, ep): $(awk -v t=${ELAPSED} 'BEGIN{printf "%.3f sec", t/300}')"
echo "  Extrapolated X-B (100k ep): $(awk -v t=${ELAPSED} 'BEGIN{h=t/300*100000/3600; printf "%.1f h", h}')"
echo "  Extrapolated X-C (18k ep):  $(awk -v t=${ELAPSED} 'BEGIN{h=t/300*18000/3600; printf "%.1f h", h}')"
echo "============================================================"
