#!/bin/bash
#SBATCH --job-name=hgate_ppo
#SBATCH --account=yanjia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # 24h — single-env PPO ~50-80 step/s
                                           # 5M steps ~= 17-20h wallclock
#SBATCH --output=logs/hgate_ppo_%j.out
#SBATCH --error=logs/hgate_ppo_%j.err
# -------------------------------------------------------------------
# CPO v2 — HGATE-PPO (Wu 2025 IEEE IoT Journal) baseline
#
# Status (HK-4.5): UNBLOCKED — code shipped HK-4.1..HK-4.4 + HK-3.1.4.
#
# Wallclock estimate: 24h --time chosen because train_hgate_ppo.py
# is currently single-env (one SyncVectorEnv worker).  Mac CPU smoke
# clocked ~75 step/s; V100 CPU expected to be similar (env stepping
# is Python-bound, not GPU-bound, see paper_drafts/throughput_debug.md).
#   1 M steps  ~ 3.7h
#   5 M steps  ~ 18h  <-- the configured total_steps
# A future multi-env refactor would shrink this to ~3h via 8-16-way
# async batching, but that's not on the HK-4.x critical path.
#
# §5 ablation chain isolation point:
#   Ours-NoThermal  --(strip RC edge + cross-attention)-->  HGATE-PPO
#
# Output: checkpoints/hgate_ppo_N17/
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

# NFS pre-warm (cheap if hot, ~30 s if cold; see
# paper_drafts/throughput_debug.md §4)
cat data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || \
    cat cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || \
    true

echo "[$(date)] HGATE-PPO starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# -------------------------------------------------------------------
# BLOCKED-marker guard: refuse to run while yaml is still a stub
# -------------------------------------------------------------------
HGATE_YAML="cpo_thermal_v2/configs/train_hgate_ppo.yaml"
if grep -q "^_BLOCKED_DO_NOT_RUN:" "$HGATE_YAML"; then
    echo "❌ HGATE-PPO yaml is marked BLOCKED — code support not yet implemented."
    echo "   See yaml header + paper_drafts/hgate_ppo_checklist.md"
    echo "   File: $HGATE_YAML"
    exit 1
fi

# Activate conda env
source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------
echo "[$(date)] Launching HGATE-PPO trainer ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/train_hgate_ppo.yaml \
    "$@"

echo "[$(date)] HGATE-PPO done"
