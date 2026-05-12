#!/bin/bash
#SBATCH --job-name=hgate_prof
#SBATCH --account=yanjia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/hgate_prof_%j.out
#SBATCH --error=logs/hgate_prof_%j.err
# -------------------------------------------------------------------
# CPO v2 — HGATE-PPO per-phase profile run (HK-4.5.3+ diagnostic)
#
# Purpose: pin down WHERE the V100 wallclock actually goes.  Sets
# HGATE_PROFILE=1 so each rollout-update heartbeat is followed by:
#   [profile] rollout=Xms (model_act / env_step / info_extract / other)
#             update=Yms
# 3-5 such heartbeats are enough to identify the real bottleneck.
#
# Disposable run:
#   - total_steps=50000 (you SHOULD scancel after 3-5 heartbeats; this
#     ceiling exists so the job can't run away)
#   - rollout_length=64 (small => heartbeats fire ~every 25s @ 40 step/s
#     instead of every 100s with the production rollout_length=256).
#     This does NOT bias the profile RATIOS; only the wall-clock per
#     heartbeat.
#   - separate run_name + ckpt_dir + tb_dir so the data doesn't pollute
#     the real pilot's directories.  Delete after analysis.
#
# Usage:
#   sbatch cpo_thermal_v2/scripts/profile_hgate_ppo.sbatch
#   tail -f logs/hgate_prof_<JOBID>.out      # wait for 3-5 [profile] rows
#   scancel <JOBID>                          # stop after enough data
#   rm -rf checkpoints/hgate_ppo_profile tb_logs/hgate_ppo_profile
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

# NFS pre-warm (cheap if hot, ~30 s if cold)
cat data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || \
    cat cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || \
    true

echo "[$(date)] HGATE-PPO profile run starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# Activate conda env (same as the real training sbatch)
source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

# -------------------------------------------------------------------
# THE KEY KNOB: enable per-phase timing in train_hgate_ppo.py
# -------------------------------------------------------------------
export HGATE_PROFILE=1

# -------------------------------------------------------------------
# Run (short rollout, small total_steps — meant to be scancelled)
# -------------------------------------------------------------------
echo "[$(date)] Launching HGATE-PPO profile run ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/train_hgate_ppo.yaml \
    --override training.total_steps=50000 \
    --override training.rollout_length=64 \
    --override logging.run_name=hgate_ppo_profile \
    --override logging.checkpoint_dir=checkpoints/hgate_ppo_profile \
    --override logging.tb_log_dir=tb_logs/hgate_ppo_profile

echo "[$(date)] HGATE-PPO profile run done (or scancelled)"
