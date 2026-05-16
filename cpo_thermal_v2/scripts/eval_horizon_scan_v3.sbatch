#!/bin/bash
#SBATCH --job-name=cpo_v3_horizon
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=hpib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/horizon_v3_%j/_main.out
#SBATCH --error=logs/horizon_v3_%j/_main.err
# -------------------------------------------------------------------
# CPO v2 — Horizon scan v3 (N=1000) with diagnostic+fix preamble.
#
# This job does THREE things in order:
#
#   PHASE 1 (DIAGNOSE): inspect torch_geometric installation, dump
#     paths and cache locations.
#
#   PHASE 2 (FIX): exhaustively remove stale JIT cache files from
#     EVERY known location (~/.cache, /tmp, /dev/shm, site-packages,
#     conda env). Run a single-process warmup to verify GATv2Conv
#     imports cleanly. If warmup fails, the job aborts BEFORE running
#     16 cells (so we don't waste hours).
#
#   PHASE 3 (RUN): run the 16-cell horizon scan, 4 cells concurrent,
#     each with torch.set_num_threads(1) to avoid BLAS contention.
#
# Total wall-clock estimate: 30-90 min if torch_geometric installation
# is recoverable. If it's totally broken, the job aborts at PHASE 2.
# -------------------------------------------------------------------

set -e
set -u
set -o pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=".:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

JOB_LOG_DIR="logs/horizon_v3_${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"
mkdir -p eval_results/horizon_scan_v3

# =========================================================
# Activate conda env (uncomment + edit if needed)
# =========================================================
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cpo_drl

echo "================================================================"
echo "[$(date)] Horizon scan v3 starting"
echo "  hostname        = $(hostname)"
echo "  SLURM_JOB_ID    = $SLURM_JOB_ID"
echo "  SLURM_PARTITION = $SLURM_JOB_PARTITION"
echo "  SLURM_CPUS      = $SLURM_CPUS_PER_TASK"
echo "  cwd             = $(pwd)"
echo "  log_dir         = $JOB_LOG_DIR"
echo "  python          = $(which python)"
echo "================================================================"

# ===================================================================
# PHASE 1 — DIAGNOSE
# ===================================================================
echo
echo "========== PHASE 1: DIAGNOSE =========="

python -u -c "
import sys, os
print('python version :', sys.version.split()[0])
print('python path    :', sys.executable)

import torch
print('torch version  :', torch.__version__)

import torch_geometric
print('torch_geometric:', torch_geometric.__version__)
TG_DIR = os.path.dirname(torch_geometric.__file__)
print('torch_geometric path:', TG_DIR)

# Where torch_geometric writes its JIT-generated modules
import torch_geometric.utils._jit  # may or may not exist
" 2>&1 | tee "${JOB_LOG_DIR}/_phase1_diag.log" || true

echo
echo "----- listing potential cache directories -----"
for D in \
    "$HOME/.cache/torch_geometric" \
    "$HOME/.cache/pyg" \
    "$HOME/.cache/torch_extensions" \
    "/tmp/torch_geometric" \
    "/tmp/pyg" \
    "/dev/shm/torch_geometric" \
    "${TMPDIR:-/tmp}"
do
    if [ -d "$D" ]; then
        echo "  $D — exists, contents:"
        ls -la "$D" 2>&1 | head -20 | sed 's/^/    /'
    else
        echo "  $D — does not exist"
    fi
done | tee -a "${JOB_LOG_DIR}/_phase1_diag.log"

echo
echo "----- looking for stale JIT files in torch_geometric package dir -----"
TG_DIR=$(python -c "import torch_geometric, os; print(os.path.dirname(torch_geometric.__file__))")
echo "  torch_geometric package dir: $TG_DIR"
find "$TG_DIR/nn/conv/" -maxdepth 2 -name "*GATv2Conv*" -o -name "gatv2_conv_*" 2>/dev/null | tee -a "${JOB_LOG_DIR}/_phase1_diag.log" || echo "  (none found in package dir)"
echo

# ===================================================================
# PHASE 2 — FIX (clean caches, then warmup)
# ===================================================================
echo
echo "========== PHASE 2: FIX =========="

echo "[$(date)] Removing stale JIT cache files everywhere ..."

# Standard cache locations
rm -rf "$HOME/.cache/torch_geometric"      2>/dev/null || true
rm -rf "$HOME/.cache/pyg"                  2>/dev/null || true
rm -rf "/tmp/torch_geometric"              2>/dev/null || true
rm -rf "/tmp/pyg"                          2>/dev/null || true
rm -rf "/dev/shm/torch_geometric"          2>/dev/null || true

# torch_geometric package dir (some versions write generated code here)
find "$TG_DIR/nn/conv/" -maxdepth 2 -name "gatv2_conv_*edge_updater*" -delete 2>/dev/null || true
find "$TG_DIR/nn/conv/" -maxdepth 2 -name "GATv2Conv_*" -delete 2>/dev/null || true

# torch_geometric also stores compiled modules in __pycache__
find "$TG_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# torch jit extensions (used by some torch_geometric versions)
rm -rf "$HOME/.cache/torch_extensions"     2>/dev/null || true

echo "[$(date)] Cache cleanup done."

echo
echo "[$(date)] Running single-process warmup to rebuild JIT cache ..."

python -u -c "
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print('[warmup] threads:', torch.get_num_threads(), '/', torch.get_num_interop_threads(), flush=True)

# Test 1: single GATv2Conv instance (simple)
from torch_geometric.nn import GATv2Conv
print('[warmup] test 1: GATv2Conv(8, 8, edge_dim=2) ...', flush=True)
layer = GATv2Conv(8, 8, edge_dim=2)
print('[warmup] test 1: OK', flush=True)

# Test 2: multiple instances with same kwargs (the actual failure mode)
print('[warmup] test 2: 3 instances same kwargs ...', flush=True)
l1 = GATv2Conv(64, 64, edge_dim=4, heads=4, add_self_loops=False)
l2 = GATv2Conv(64, 64, edge_dim=4, heads=4, add_self_loops=False)
l3 = GATv2Conv(64, 64, edge_dim=4, heads=4, add_self_loops=False)
print('[warmup] test 2: OK', flush=True)

# Test 3: instantiate the actual model architecture used in eval
print('[warmup] test 3: PPOActorCritic full model ...', flush=True)
import sys
sys.path.insert(0, '.')
from cpo_thermal_v2.models import PPOActorCritic
m = PPOActorCritic(
    action_mode='auto_only',
    K_delay=1,
    proc_in_dim=3,
    edge_dim_t2p=1,
)
print('[warmup] test 3: OK — model has', sum(p.numel() for p in m.parameters()), 'parameters', flush=True)
" 2>&1 | tee "${JOB_LOG_DIR}/_phase2_warmup.log"

WARMUP_RC=${PIPESTATUS[0]}
if [ "$WARMUP_RC" -ne 0 ]; then
    echo
    echo "================================================================"
    echo "[$(date)] WARMUP FAILED — torch_geometric installation is broken"
    echo "  See: ${JOB_LOG_DIR}/_phase2_warmup.log"
    echo
    echo "Likely fix: reinstall torch_geometric to a known-good version."
    echo "Suggested commands (run in a separate sbatch with internet access):"
    echo "  pip install torch-geometric==2.5.3 --force-reinstall --no-deps"
    echo "Or check what version your local Mac has:"
    echo "  python -c 'import torch_geometric; print(torch_geometric.__version__)'"
    echo "================================================================"
    exit 1
fi

echo "[$(date)] Warmup OK — torch_geometric is healthy"

# ===================================================================
# PHASE 3 — RUN (the actual horizon scan)
# ===================================================================
echo
echo "========== PHASE 3: RUN =========="

EVAL_CKPT="checkpoints/stage2_hybrid_N17/best.pt"
if [ ! -f "$EVAL_CKPT" ]; then
    echo "ERROR: Stage 2 checkpoint not found at: $EVAL_CKPT"
    exit 1
fi

# Decima fair checkpoint — must be set so eval uses retrained baseline
# (otherwise evaluate.py falls back to old DecimaScheduler with thermal-
#  masked Stage-2 weights, which is NOT what we want)
DECIMA_FAIR_CKPT="checkpoints/decima_fair_N17/best.pt"
if [ ! -f "$DECIMA_FAIR_CKPT" ]; then
    echo "WARNING: decima_fair checkpoint not found at: $DECIMA_FAIR_CKPT"
    echo "  evaluate.py will fall back to old DecimaScheduler — NOT the"
    echo "  retrained Ours-NoThermal baseline we need for v3 data."
    echo "  Continue anyway? Aborting to be safe."
    exit 1
fi
echo "  ours-hybrid ckpt   = $EVAL_CKPT"
echo "  decima-fair ckpt   = $DECIMA_FAIR_CKPT"

MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
echo "  MAX_CONCURRENT     = $MAX_CONCURRENT"
echo

run_one_cell() {
    local SETTING="$1"
    local LOW="$2"
    local HIGH="$3"
    local H="$4"

    local OUT_DIR="eval_results/horizon_scan_v3/${SETTING}/dags${H}"
    local CELL_LOG="${JOB_LOG_DIR}/${SETTING}_dags${H}.log"

    if [ -f "${OUT_DIR}/episodes.csv" ]; then
        local n_rows
        n_rows=$(wc -l < "${OUT_DIR}/episodes.csv")
        if [ "$n_rows" -ge 4000 ]; then
            echo "[$(date)] [SKIP] ${SETTING}/dags${H}: $n_rows rows"
            return 0
        fi
    fi

    echo "[$(date)] [START] ${SETTING}/dags${H}  (log=${CELL_LOG})"

    python -u -c "
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import sys
sys.argv = [
    'evaluate',
    '--config', 'cpo_thermal_v2/configs/eval_horizon_scan_v3.yaml',
    '--override', 'eval.checkpoint_path=${EVAL_CKPT}',
    '--override', 'eval.decima_fair_ckpt=${DECIMA_FAIR_CKPT}',
    '--override', 'eval.initial_temp_range=[${LOW},${HIGH}]',
    '--override', 'eval.dags_per_episode=${H}',
    '--override', 'eval.output_dir=${OUT_DIR}',
    '--override', 'eval.device=cpu',
]
from cpo_thermal_v2.evaluation.evaluate import main
main()
" > "${CELL_LOG}" 2>&1

    local RC=$?
    if [ "$RC" -ne 0 ]; then
        echo "[$(date)] [FAIL]  ${SETTING}/dags${H} (rc=$RC) — see ${CELL_LOG}"
        return $RC
    fi
    echo "[$(date)] [DONE]  ${SETTING}/dags${H}"
}

export -f run_one_cell
export EVAL_CKPT DECIMA_FAIR_CKPT JOB_LOG_DIR

WORK_LIST=(
    "easy 30.0 45.0 20"
    "easy 30.0 45.0 50"
    "easy 30.0 45.0 100"
    "easy 30.0 45.0 200"
    "warm 40.0 55.0 20"
    "warm 40.0 55.0 50"
    "warm 40.0 55.0 100"
    "warm 40.0 55.0 200"
    "hot 50.0 65.0 20"
    "hot 50.0 65.0 50"
    "hot 50.0 65.0 100"
    "hot 50.0 65.0 200"
    "extreme 60.0 75.0 20"
    "extreme 60.0 75.0 50"
    "extreme 60.0 75.0 100"
    "extreme 60.0 75.0 200"
)

START_T=$(date +%s)

printf '%s\n' "${WORK_LIST[@]}" | xargs -n 1 -P "$MAX_CONCURRENT" -I{} bash -c '
    line="{}"
    set -- $line
    run_one_cell "$1" "$2" "$3" "$4"
'

END_T=$(date +%s)
ELAPSED=$((END_T - START_T))
H_PART=$((ELAPSED / 3600))
M_PART=$(( (ELAPSED % 3600) / 60 ))

echo
echo "================================================================"
echo "[$(date)] Horizon scan v3 complete"
echo "  PHASE 3 wall-clock: ${H_PART}h ${M_PART}m"
echo
echo "Artifacts (16 cells expected):"
echo "  eval_results/horizon_scan_v3/<setting>/dags<H>/episodes.csv"
echo "Per-cell logs:"
echo "  ${JOB_LOG_DIR}/"
echo "================================================================"
