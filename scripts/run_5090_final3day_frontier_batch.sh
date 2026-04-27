#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

RUN_VERSION="${RUN_VERSION:-geom1}"
SEEDS="${SEEDS:-1337}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"
TRIGRAM_COUNT_WORKERS="${TRIGRAM_COUNT_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
GEOMETRY_SPECS="${GEOMETRY_SPECS:-}"
GEOMETRY_NUM_STEPS="${GEOMETRY_NUM_STEPS:-}"
GEOMETRY_LR_HOLD_STEPS="${GEOMETRY_LR_HOLD_STEPS:-}"
GEOMETRY_BATCH_SIZE="${GEOMETRY_BATCH_SIZE:-}"
GEOMETRY_SEQ_LEN="${GEOMETRY_SEQ_LEN:-}"
TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-}"
VAL_EVERY="${VAL_EVERY:-}"
VAL_STEPS="${VAL_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"
LOG_STATE_EVERY="${LOG_STATE_EVERY:-}"
SAVE_EVERY="${SAVE_EVERY:-}"
TRIGRAM_MAX_TOKENS="${TRIGRAM_MAX_TOKENS:-}"
SPEC_MAX_TOKENS="${SPEC_MAX_TOKENS:-}"
DATA_MAX_TOKENS="${DATA_MAX_TOKENS:-}"
GEOMETRY_TRAIN_LABEL="${GEOMETRY_TRAIN_LABEL:-}"
WANDB="${WANDB:-}"

RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_GEOMETRY_SCREEN="${RUN_GEOMETRY_SCREEN:-1}"
FRONTIER_BATCH_ID="${FRONTIER_BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/5090_final3day/${FRONTIER_BATCH_ID}}"
BENCHMARK_JSON="${BENCHMARK_JSON:-${LOG_DIR}/geometry_frontier_benchmark.json}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --run-version VALUE
  --seeds VALUE
  --trigram-top-k VALUE
  --count-workers VALUE
  --run-benchmark | --no-run-benchmark
  --run-geometry-screen | --no-run-geometry-screen
  --frontier-batch-id VALUE
  --log-dir VALUE
  --benchmark-json VALUE
  --benchmark-steps VALUE
  --benchmark-warmup VALUE
  --geometry-specs VALUE
  --num-steps VALUE
  --lr-hold-steps VALUE
  --batch-size VALUE
  --seq-len VALUE
  --target-effective-step-tokens VALUE
  --val-every VALUE
  --val-steps VALUE
  --log-every VALUE
  --log-state-every VALUE
  --save-every VALUE
  --trigram-max-tokens VALUE
  --spec-max-tokens VALUE
  --data-max-tokens VALUE
  --geometry-train-label VALUE
  --no-wandb
  --smoke-test
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-version) RUN_VERSION="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --trigram-top-k) TRIGRAM_TOP_K="$2"; shift 2 ;;
    --trigram-count-workers|--count-workers) TRIGRAM_COUNT_WORKERS="$2"; shift 2 ;;
    --run-benchmark) RUN_BENCHMARK=1; shift ;;
    --no-run-benchmark) RUN_BENCHMARK=0; shift ;;
    --run-geometry-screen) RUN_GEOMETRY_SCREEN=1; shift ;;
    --no-run-geometry-screen) RUN_GEOMETRY_SCREEN=0; shift ;;
    --frontier-batch-id)
      FRONTIER_BATCH_ID="$2"
      LOG_DIR="${REPO_ROOT}/logs/5090_final3day/${FRONTIER_BATCH_ID}"
      BENCHMARK_JSON="${LOG_DIR}/geometry_frontier_benchmark.json"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      BENCHMARK_JSON="${LOG_DIR}/geometry_frontier_benchmark.json"
      shift 2
      ;;
    --benchmark-json) BENCHMARK_JSON="$2"; shift 2 ;;
    --benchmark-steps) BENCHMARK_STEPS="$2"; shift 2 ;;
    --benchmark-warmup) BENCHMARK_WARMUP="$2"; shift 2 ;;
    --geometry-specs) GEOMETRY_SPECS="$2"; shift 2 ;;
    --num-steps|--geometry-num-steps) GEOMETRY_NUM_STEPS="$2"; shift 2 ;;
    --lr-hold-steps|--geometry-lr-hold-steps) GEOMETRY_LR_HOLD_STEPS="$2"; shift 2 ;;
    --batch-size|--geometry-batch-size) GEOMETRY_BATCH_SIZE="$2"; shift 2 ;;
    --seq-len|--geometry-seq-len) GEOMETRY_SEQ_LEN="$2"; shift 2 ;;
    --target-effective-step-tokens) TARGET_EFFECTIVE_STEP_TOKENS="$2"; shift 2 ;;
    --val-every) VAL_EVERY="$2"; shift 2 ;;
    --val-steps) VAL_STEPS="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --log-state-every) LOG_STATE_EVERY="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --trigram-max-tokens) TRIGRAM_MAX_TOKENS="$2"; shift 2 ;;
    --spec-max-tokens) SPEC_MAX_TOKENS="$2"; shift 2 ;;
    --data-max-tokens) DATA_MAX_TOKENS="$2"; shift 2 ;;
    --geometry-train-label) GEOMETRY_TRAIN_LABEL="$2"; shift 2 ;;
    --no-wandb) WANDB=0; shift ;;
    --smoke-test) SMOKE_TEST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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

geometry_matrix_cmd=(
  bash "${SCRIPT_DIR}/run_5090_trigram_geometry_matrix.sh"
  --run-version "${RUN_VERSION}"
  --seeds "${SEEDS}"
  --trigram-top-k "${TRIGRAM_TOP_K}"
  --count-workers "${TRIGRAM_COUNT_WORKERS}"
)
append_if_set() {
  local flag="$1"
  local value="$2"
  if [[ -n "${value:-}" ]]; then
    geometry_matrix_cmd+=("${flag}" "${value}")
  fi
}
append_if_set "--geometry-specs" "${GEOMETRY_SPECS}"
append_if_set "--num-steps" "${GEOMETRY_NUM_STEPS}"
append_if_set "--lr-hold-steps" "${GEOMETRY_LR_HOLD_STEPS}"
append_if_set "--batch-size" "${GEOMETRY_BATCH_SIZE}"
append_if_set "--seq-len" "${GEOMETRY_SEQ_LEN}"
append_if_set "--target-effective-step-tokens" "${TARGET_EFFECTIVE_STEP_TOKENS}"
append_if_set "--val-every" "${VAL_EVERY}"
append_if_set "--val-steps" "${VAL_STEPS}"
append_if_set "--log-every" "${LOG_EVERY}"
append_if_set "--log-state-every" "${LOG_STATE_EVERY}"
append_if_set "--save-every" "${SAVE_EVERY}"
append_if_set "--trigram-max-tokens" "${TRIGRAM_MAX_TOKENS}"
append_if_set "--spec-max-tokens" "${SPEC_MAX_TOKENS}"
append_if_set "--data-max-tokens" "${DATA_MAX_TOKENS}"
append_if_set "--geometry-train-label" "${GEOMETRY_TRAIN_LABEL}"
if [[ "${WANDB:-}" == "0" ]]; then
  geometry_matrix_cmd+=(--no-wandb)
fi
if [[ "${SMOKE_TEST}" == "1" ]]; then
  geometry_matrix_cmd+=(--smoke-test)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  geometry_matrix_cmd+=(--dry-run)
fi

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
smoke_test=${SMOKE_TEST}
log_dir=${LOG_DIR}
benchmark_json=${BENCHMARK_JSON}
stop_after=stage1_geometry_screen
EOF
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
fi

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  echo
  echo "=== Stage 0: synthetic frontier benchmark ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${benchmark_cmd[@]}"
  else
    mkdir -p "${LOG_DIR}"
    "${benchmark_cmd[@]}" | tee "${LOG_DIR}/geometry_frontier_benchmark.log"
  fi
fi

if [[ "${RUN_GEOMETRY_SCREEN}" == "1" ]]; then
  echo
  echo "=== Stage 1: fixed-token K2 blocks0 geometry screen ==="
  "${geometry_matrix_cmd[@]}"
fi

echo
echo "Stage batch complete. It intentionally stops here before confirmations."
echo "Next analysis command:"
print_cmd "${analysis_cmd[@]}"
