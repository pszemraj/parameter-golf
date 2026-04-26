#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

export RUN_VERSION="${RUN_VERSION:-geom1}"
export SEEDS="${SEEDS:-1337}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"

RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_GEOMETRY_SCREEN="${RUN_GEOMETRY_SCREEN:-1}"
FRONTIER_BATCH_ID="${FRONTIER_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/5090_final3day/${FRONTIER_BATCH_ID}}"
BENCHMARK_JSON="${BENCHMARK_JSON:-${LOG_DIR}/geometry_frontier_benchmark.json}"

benchmark_cmd=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/benchmark_core_amp_perf.py"
  --mode full
  --shape current_d48_l12_i480:48:12:10
  --shape d64_l10_i512:64:10:8
  --shape d96_l6_i512:96:6:5.333333333333333
  --shape d96_l8_i512:96:8:5.333333333333333
  --shape d128_l4_i512:128:4:4
  --shape d128_l5_i512:128:5:4
  --shape d128_l6_i384:128:6:3
  --shape d160_l4_i512:160:4:3.2
  --batch-size "${BENCHMARK_BATCH_SIZE:-256}"
  --seq-len "${BENCHMARK_SEQ_LEN:-512}"
  --warmup "${BENCHMARK_WARMUP:-4}"
  --steps "${BENCHMARK_STEPS:-10}"
  --trigram-top-k "${TRIGRAM_TOP_K}"
  --num-blocks "${BENCHMARK_NUM_BLOCKS:-0}"
  --output "${BENCHMARK_JSON}"
)

analysis_cmd=(
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/analyze_5090_geometry_frontier.py"
  --run-version "${RUN_VERSION}"
  --benchmark "${BENCHMARK_JSON}"
)

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

cat <<EOF
5090 final-three-day frontier batch
repo_root=${REPO_ROOT}
python=${PYTHON_BIN}
run_version=${RUN_VERSION}
seeds=${SEEDS}
trigram_top_k=${TRIGRAM_TOP_K}
run_benchmark=${RUN_BENCHMARK}
run_geometry_screen=${RUN_GEOMETRY_SCREEN}
log_dir=${LOG_DIR}
benchmark_json=${BENCHMARK_JSON}
stop_after=stage1_geometry_screen
EOF
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
fi

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  echo
  echo "=== Stage 0: synthetic frontier benchmark ==="
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    print_cmd "${benchmark_cmd[@]}"
  else
    mkdir -p "${LOG_DIR}"
    "${benchmark_cmd[@]}" | tee "${LOG_DIR}/geometry_frontier_benchmark.log"
  fi
fi

if [[ "${RUN_GEOMETRY_SCREEN}" == "1" ]]; then
  echo
  echo "=== Stage 1: fixed-token K2 blocks0 geometry screen ==="
  bash "${SCRIPT_DIR}/run_5090_trigram_geometry_matrix.sh"
fi

echo
echo "Stage batch complete. It intentionally stops here before confirmations."
echo "Next analysis command:"
print_cmd "${analysis_cmd[@]}"
