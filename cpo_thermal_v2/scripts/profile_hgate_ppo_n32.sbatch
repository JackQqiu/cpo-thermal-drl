#!/bin/bash
#SBATCH --job-name=hgate_prof32
#SBATCH --account=yanjia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=logs/hgate_prof32_%j.out
#SBATCH --error=logs/hgate_prof32_%j.err
# -------------------------------------------------------------------
# CPO v2 — HGATE-PPO profile, num_envs=32 / cpus=32 variant
#
# Hypothesis under test: V100 wallclock at num_envs=16 is dominated by
# vec_env.step (~85% per the n16 profile).  Since each vec_env.step
# call's wallclock is bounded by the slowest of N workers (~300ms with
# N=16 currently), doubling N to 32 with matching --cpus-per-task should
# keep per-call wallclock roughly the same but DOUBLE the env-steps
# returned per call → ~2× total throughput.
#
# Predicted: HK-4.5.4 (n16) ~43 step/s  →  this (n32) ~75-90 step/s
#
# Risks / verify:
#   - SLURM partition GPU node must have ≥32 cores AVAILABLE.  If the
#     job pends > 30s on resource allocation, drop to cpus=24 + n=24.
#   - --mem bumped 64G → 96G; each env worker has its own copy of the
#     simulator state.  Bump higher if OOM.
#   - Per-rollout buffer doubles (T*N = 64*32 = 2048 transitions vs 1024
#     before).  Minibatch sweeps grow proportionally; update% may rise
#     slightly.  Monitor in the [profile] row.
#
# Usage (after `git pull` on the server):
#   sbatch cpo_thermal_v2/scripts/profile_hgate_ppo_n32.sbatch
#   tail -f logs/hgate_prof32_<JOBID>.out      # wait for 3-5 [profile] rows
#   scancel <JOBID>
#   rm -rf checkpoints/hgate_ppo_profile_n32 tb_logs/hgate_ppo_profile_n32
#
# If this profile shows step/s ≈ 80+ AND env_step% still ~85%, then
# the production train_hgate_ppo.sbatch should be amended to
#   --cpus-per-task=32  /  --mem=96G  +  training.num_envs=32
# (the yaml's num_envs:16 needs --override or yaml edit).
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

# NFS pre-warm
cat data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || \
    cat cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json > /dev/null 2>&1 || \
    true

echo "[$(date)] HGATE-PPO n32 profile run starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"
echo "  expecting    : --cpus-per-task=32 (n32 variant)"

source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

# KEY: per-phase profile + override num_envs to 32
export HGATE_PROFILE=1

echo "[$(date)] Launching HGATE-PPO n32 profile run ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/train_hgate_ppo.yaml \
    --override training.total_steps=100000 \
    --override training.num_envs=32 \
    --override training.rollout_length=64 \
    --override logging.run_name=hgate_ppo_profile_n32 \
    --override logging.checkpoint_dir=checkpoints/hgate_ppo_profile_n32 \
    --override logging.tb_log_dir=tb_logs/hgate_ppo_profile_n32

echo "[$(date)] HGATE-PPO n32 profile run done (or scancelled)"
