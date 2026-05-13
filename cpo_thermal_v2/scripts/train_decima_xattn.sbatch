#!/bin/bash
#SBATCH --job-name=decima_xattn
#SBATCH --account=yanjia
#SBATCH --partition=a100x4
#SBATCH --gres=gpu:1                       # cluster CPU-per-GPU policy:
                                           # `gpu` partition: 5 cpus per gpu
                                           # `a100x4`       : 16 cpus per gpu
                                           # one A100 gives us 16 real cores,
                                           # no need to over-claim GPUs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                 # 16 worker processes for num_envs=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00                    # 24h cap (HGATE precedent: 5M steps
                                           # at ~58 step/s ≈ 24h with 16 real cores)
#SBATCH --output=logs/decima_xattn_%j.out
#SBATCH --error=logs/decima_xattn_%j.err
# -------------------------------------------------------------------
# CPO v2 — D2 baseline (Decima encoder + cross-attention actor)
#
# Status (HK-5.0): NEW.  D2 = composition baseline that isolates the
# cross-attention actor's contribution over a homogeneous GCN trunk.
#
# §5 ablation chain isolation point:
#   Decima --(swap MLP scorer for cross-attn)--> D2
#   D2     --(swap homog GCN for hetero GAT)---> HGATE-PPO
#
# Wallclock estimate (matches HGATE-PPO precedent):
#   ~58 step/s on A100 single-card, 16 real CPUs
#   5 M steps  ~ 24h
#
# Output: checkpoints/decima_xattn_N17/
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

echo "[$(date)] D2 (decima_xattn) starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

source /project/qiuzhaoqi/miniconda3/etc/profile.d/conda.sh
conda activate cpo_drl

mkdir -p logs

echo "[$(date)] Launching D2 trainer ..."
python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/train_decima_xattn.yaml \
    "$@"

echo "[$(date)] D2 done"
