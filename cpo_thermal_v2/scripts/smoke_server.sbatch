#!/bin/bash
#SBATCH --job-name=cpo_v2_smoke_server
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=gpu                  # adjust to your cluster's partition
#SBATCH --gres=gpu:1                # 1× A100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00                  # 20-min budget; smoke takes ~5-8 min
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
# -------------------------------------------------------------------
# CPO v2 — Server-side smoke validation
#
# Validates the full Stage A+B+C+D stack on the actual training
# hardware (A100 + Linux + AsyncVectorEnv) BEFORE committing to the
# 28-32h Stage 1 sbatch.  Catches any environment-specific issues
# (CUDA, PyG kernels, fork-vs-spawn, file paths) in 5-8 minutes
# instead of hours.
#
# Compared to the Mac smoke we just ran:
#   - device: GPU (cuda:0) instead of CPU
#   - num_envs: 8 instead of 4 (real parallelism)
#   - vector_env_mode: async (default) instead of sync
#   - total_steps: 10,000 (~5 min on A100) instead of 2,000
#
# What "success" looks like
# -------------------------
# Output should show:
#   - "device = cuda:0" + GPU name from nvidia-smi
#   - "[smoke] OK" after the startup smoke
#   - 4-5 phase heartbeat lines with finite loss/KL/H values
#   - "training complete" at step ~10,000
#
# If anything explodes, the traceback comes back instantly — copy it
# back and we debug, exactly like the Mac smoke loop.
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

# -------------------------------------------------------------------
# Working directory + Python path
# -------------------------------------------------------------------
# Submit this sbatch from the project root (the directory that
# CONTAINS the cpo_thermal_v2/ folder).
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"

echo "[$(date)] Server smoke starting on node $(hostname)"
echo "  SLURM_JOB_ID = $SLURM_JOB_ID"
echo "  cwd          = $(pwd)"
echo "  PYTHONPATH   = $PYTHONPATH"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  CPUs         = $SLURM_CPUS_PER_TASK"

# Activate conda env (uncomment if using conda)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_drl

mkdir -p logs

# -------------------------------------------------------------------
# Pre-flight: data preparation (idempotent — skips if files exist)
# -------------------------------------------------------------------
DATASET_V2="data_pipeline/process/alibaba_dags_v2.json"
if [ ! -f "$DATASET_V2" ]; then
    DATASET_V1="data_pipeline/process/alibaba_dags.json"
    if [ ! -f "$DATASET_V1" ]; then
        # Maybe the user staged the v2 JSON inside the package
        if [ -f "cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json" ]; then
            echo "✅ found v2 JSON inside package; env autodiscover will resolve."
        else
            echo "❌ Neither $DATASET_V2 nor $DATASET_V1 found"
            echo "   The env's autodiscover will look in:"
            echo "     - \$CPO_DAGS_V2 environment variable"
            echo "     - ./data_pipeline/process/alibaba_dags_v2.json"
            echo "     - cpo_thermal_v2/data_pipeline/process/alibaba_dags_v2.json"
            echo "   Make sure the v2 JSON exists in one of those locations,"
            echo "   or run compute_dag_features first."
            exit 1
        fi
    else
        echo "[$(date)] Computing v2 DAG features (one-time) ..."
        python -u -m cpo_thermal_v2.data_pipeline.compute_dag_features \
            --input  "$DATASET_V1" \
            --output "$DATASET_V2"
    fi
fi

if [ ! -f "data/thermal_matrics/N17/matrix_A.npy" ]; then
    echo "[$(date)] Generating RC matrices (5 sizes, ~10s) ..."
    python -u -m cpo_thermal_v2.data_pipeline.generate_matrices \
        --output-root data/thermal_matrics/
fi

# -------------------------------------------------------------------
# Smoke run
# -------------------------------------------------------------------
echo "[$(date)] Launching server smoke (10,000 steps, GPU + AsyncVectorEnv) ..."
echo

python -u -m cpo_thermal_v2.training.train \
    --config cpo_thermal_v2/configs/stage1_auto_only.yaml \
    --override training.total_steps=20000 \
    --override training.rollout_length=128 \
    --override training.num_envs=16 \
    --override logging.checkpoint_dir=/tmp/server_smoke_16 \
    --override logging.tb_log_dir=/tmp/server_smoke_16_tb \
    --override logging.run_name=smoke_16envs

echo
echo "[$(date)] Smoke completed.  Inspect /tmp/server_smoke_16/ for"
echo "  - resolved_config.yaml  (config used)"
echo "  - final.pt              (final checkpoint)"
echo "  - any ckpt_step_*.pt    (periodic checkpoints)"
echo "If everything looks healthy, submit Stage 1:"
echo "    sbatch cpo_thermal_v2/scripts/train_stage1.sbatch"
