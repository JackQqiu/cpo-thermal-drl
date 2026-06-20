#!/usr/bin/env bash
# =====================================================================
# run_eval_lagrangian_parallel.sh — parallel launcher for R2.2 eval
# =====================================================================
# Runs ONE process per topology size N (so the 5 default sizes run in
# parallel), each writing repro_outputs/eval_cells/N<N>.csv, then merges
# all per-N CSVs into repro_outputs/eval_lagrangian_matrix.csv and prints
# the combined table.
#
# CRITICAL: each torch process otherwise spawns ~128 BLAS threads; 5 of
# them oversubscribe the box and thrash.  We pin every BLAS backend to a
# single thread below so 5 single-threaded processes coexist cleanly.
#
# Usage
# -----
#   # server (default interpreter path)
#   bash cpo_thermal_v2/scripts/run_eval_lagrangian_parallel.sh
#
#   # local (use the conda env python on PATH, or any interpreter)
#   PYTHON=python   bash cpo_thermal_v2/scripts/run_eval_lagrangian_parallel.sh
#   PYTHON="conda run -n cpo_rl python" bash cpo_thermal_v2/scripts/run_eval_lagrangian_parallel.sh
#
# Env knobs (all optional):
#   PYTHON    interpreter (default: server cpo_rl python; falls back to
#             `python` if that path is absent)
#   NODES     space-separated topology sizes (default "9 13 17 24 33")
#   EPISODES  episodes per (scheduler, N) cell    (default 500)
#   DEVICE    torch device                        (default cpu)
#   OUTDIR    base output dir                     (default repro_outputs)
# =====================================================================
set -euo pipefail

# --- thread pinning: MUST be set before any torch/numpy process starts ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- interpreter resolution (server default, but works locally) ---
SERVER_PY="/home/mfy/anaconda3/envs/cpo_rl/bin/python"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${SERVER_PY}" ]]; then
    PYTHON="${SERVER_PY}"
  else
    PYTHON="python"
  fi
fi

NODES="${NODES:-9 13 17 24 33}"
EPISODES="${EPISODES:-500}"
DEVICE="${DEVICE:-cpu}"
OUTDIR="${OUTDIR:-repro_outputs}"

# Resolve repo root = two levels up from this script (cpo_thermal_v2/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CELL_DIR="${OUTDIR}/eval_cells"
MERGED="${OUTDIR}/eval_lagrangian_matrix.csv"
mkdir -p "${CELL_DIR}"

echo "=================================================================="
echo "[launcher] PYTHON   = ${PYTHON}"
echo "[launcher] NODES    = ${NODES}"
echo "[launcher] EPISODES = ${EPISODES}"
echo "[launcher] DEVICE   = ${DEVICE}"
echo "[launcher] OUTDIR   = ${OUTDIR}"
echo "[launcher] threads pinned: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS}"
echo "=================================================================="

# --- launch one process per N (parallel) ---
pids=()
for N in ${NODES}; do
  out="${CELL_DIR}/N${N}.csv"
  log="${CELL_DIR}/N${N}.log"
  echo "[launcher] starting N=${N} -> ${out} (log: ${log})"
  ${PYTHON} -m cpo_thermal_v2.scripts.eval_lagrangian_matrix \
      --nodes "${N}" \
      --episodes "${EPISODES}" \
      --device "${DEVICE}" \
      --out "${out}" \
      > "${log}" 2>&1 &
  pids+=("$!")
done

# --- wait for all, capture failures ---
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[launcher] ERROR: a worker (pid ${pids[$i]}) failed; see logs in ${CELL_DIR}/"
    fail=1
  fi
done

# --- merge per-N CSVs (header from the first file, data from all) ---
echo "[launcher] merging per-N CSVs into ${MERGED}"
header_written=0
: > "${MERGED}.tmp"
for N in ${NODES}; do
  f="${CELL_DIR}/N${N}.csv"
  if [[ ! -s "${f}" ]]; then
    echo "[launcher] WARNING: missing/empty ${f}; skipping in merge"
    continue
  fi
  if [[ "${header_written}" -eq 0 ]]; then
    cat "${f}" >> "${MERGED}.tmp"
    header_written=1
  else
    tail -n +2 "${f}" >> "${MERGED}.tmp"
  fi
done
mv "${MERGED}.tmp" "${MERGED}"

echo "=================================================================="
echo "[launcher] merged table -> ${MERGED}"
echo "=================================================================="
# Pretty-print via pandas so columns align (falls back to cat on failure).
${PYTHON} - "${MERGED}" <<'PYEOF' || cat "${MERGED}"
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
with pd.option_context("display.width", 160, "display.max_columns", None):
    print(df.to_string(index=False))
PYEOF

if [[ "${fail}" -ne 0 ]]; then
  echo "[launcher] completed WITH errors (see per-N logs)."
  exit 1
fi
echo "[launcher] done."
