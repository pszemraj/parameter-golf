#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

RUN_VERSION="geom1"
SEED="1337"
FRONTIER_BATCH_ID="geom1"
TRIGRAM_TOP_K="2"
TRIGRAM_COUNT_WORKERS="1"
MAX_CONFIRMATIONS="2"
STOP_AFTER="k4"
RUN_BENCHMARK="auto"
DRY_RUN="0"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs/5090_adaptive_closeout/${RUN_ID}"
BENCHMARK_JSON=""
SMOKE_TEST="0"
RUN_VERSION_SET="0"
FRONTIER_BATCH_ID_SET="0"
LOG_DIR_SET="0"
RUN_BENCHMARK_SET="0"
BASELINE_BPB=""
SMOKE_GEOMETRY_SPECS="blocks0_d96_l6_i512 96 6 512"
SMOKE_SCREEN_STEPS="2"
SMOKE_CONFIRM_STEPS="3"
SMOKE_VARIANT_STEPS="2"
SMOKE_HOLD_STEPS="1"
SMOKE_BATCH_SIZE="2"
SMOKE_BPTT_BATCH_SIZE="1"
SMOKE_BPTT_CHUNKS="2"
SMOKE_SEQ_LEN="64"
SMOKE_EFFECTIVE_STEP_TOKENS="128"
SMOKE_VAL_EVERY="1"
SMOKE_VAL_STEPS="1"
SMOKE_LOG_EVERY="1"
SMOKE_LOG_STATE_EVERY="0"
SMOKE_SAVE_EVERY="0"
SMOKE_TRIGRAM_MAX_TOKENS="200000"
SMOKE_DATA_MAX_TOKENS="131072"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Adaptive 5090 closeout:
  1. finish the fixed-token K2 blocks0 geometry frontier
  2. confirm at most N promoted geometry rows
  3. run BPTT2 only on the best completed confirmation
  4. run one K4 screen on the selected best geometry/BPTT setting

Options:
  --run-version VALUE          default: geom1
  --seed VALUE                 default: 1337
  --frontier-batch-id VALUE    default: geom1
  --benchmark-json PATH
  --count-workers VALUE        trigram count workers for any cache miss
  --max-confirmations VALUE    default: 2
  --stop-after VALUE           frontier|confirm|bptt|k4, default: k4
  --baseline-bpb VALUE
  --run-benchmark | --no-run-benchmark
  --run-id VALUE
  --log-dir PATH
  --smoke-test                 tiny local execution, W&B disabled
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-version) RUN_VERSION="$2"; RUN_VERSION_SET="1"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --frontier-batch-id) FRONTIER_BATCH_ID="$2"; FRONTIER_BATCH_ID_SET="1"; shift 2 ;;
    --benchmark-json) BENCHMARK_JSON="$2"; shift 2 ;;
    --trigram-top-k) TRIGRAM_TOP_K="$2"; shift 2 ;;
    --trigram-count-workers|--count-workers) TRIGRAM_COUNT_WORKERS="$2"; shift 2 ;;
    --max-confirmations) MAX_CONFIRMATIONS="$2"; shift 2 ;;
    --stop-after) STOP_AFTER="$2"; shift 2 ;;
    --baseline-bpb) BASELINE_BPB="$2"; shift 2 ;;
    --run-benchmark) RUN_BENCHMARK="1"; RUN_BENCHMARK_SET="1"; shift ;;
    --no-run-benchmark) RUN_BENCHMARK="0"; RUN_BENCHMARK_SET="1"; shift ;;
    --run-id)
      RUN_ID="$2"
      LOG_DIR="${REPO_ROOT}/logs/5090_adaptive_closeout/${RUN_ID}"
      shift 2
      ;;
    --log-dir) LOG_DIR="$2"; LOG_DIR_SET="1"; shift 2 ;;
    --smoke-test) SMOKE_TEST="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${STOP_AFTER}" in
  frontier|confirm|bptt|k4) ;;
  *) echo "Invalid --stop-after: ${STOP_AFTER}" >&2; usage >&2; exit 2 ;;
esac

if [[ "${TRIGRAM_TOP_K}" != "2" ]]; then
  echo "Adaptive closeout frontier must start from trigram top-K=2, got ${TRIGRAM_TOP_K}" >&2
  exit 2
fi

if [[ "${SMOKE_TEST}" == "1" ]]; then
  export ALLOW_DEGRADED_5090_SERIOUS=1
  export WANDB=0
  if [[ "${RUN_VERSION_SET}" == "0" ]]; then
    RUN_VERSION="smoke_adaptive_${RUN_ID}"
  fi
  if [[ "${FRONTIER_BATCH_ID_SET}" == "0" ]]; then
    FRONTIER_BATCH_ID="${RUN_VERSION}"
  fi
  if [[ "${LOG_DIR_SET}" == "0" ]]; then
    LOG_DIR="${REPO_ROOT}/logs/5090_adaptive_closeout/${RUN_ID}"
  fi
  if [[ -z "${BASELINE_BPB}" ]]; then
    BASELINE_BPB="99.0"
  fi
  if [[ "${RUN_BENCHMARK_SET}" == "0" ]]; then
    RUN_BENCHMARK="0"
  fi
fi

if [[ -z "${BENCHMARK_JSON}" ]]; then
  BENCHMARK_JSON="${REPO_ROOT}/logs/5090_final3day/${FRONTIER_BATCH_ID}/geometry_frontier_benchmark.json"
fi

if [[ "${RUN_BENCHMARK}" == "auto" ]]; then
  if [[ -f "${BENCHMARK_JSON}" ]]; then
    RUN_BENCHMARK="0"
  else
    RUN_BENCHMARK="1"
  fi
fi

export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p "${LOG_DIR}"

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

stage_rank() {
  case "$1" in
    frontier) echo 1 ;;
    confirm) echo 2 ;;
    bptt) echo 3 ;;
    k4) echo 4 ;;
    *) echo 999 ;;
  esac
}

should_run_stage() {
  local stage="$1"
  [[ "$(stage_rank "${stage}")" -le "$(stage_rank "${STOP_AFTER}")" ]]
}

run_logged() {
  local name="$1"
  shift
  echo
  echo "=== ${name} ==="
  print_cmd "$@"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

PLAN_SCRIPT=""

write_plan() {
  local stage="$1"
  local out_script="${LOG_DIR}/${stage}.sh"
  local out_md="${LOG_DIR}/${stage}.md"
  local plan_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/plan_5090_adaptive_closeout.py" \
      --stage "${stage}" \
      --run-version "${RUN_VERSION}" \
      --seed "${SEED}" \
      --benchmark "${BENCHMARK_JSON}" \
      --max-confirmations "${MAX_CONFIRMATIONS}" \
      --count-workers "${TRIGRAM_COUNT_WORKERS}" \
      --write-script "${out_script}" \
      --emit markdown
  )
  if [[ -n "${BASELINE_BPB}" ]]; then
    plan_cmd+=(--baseline-bpb "${BASELINE_BPB}")
  fi
  if [[ "${SMOKE_TEST}" == "1" ]]; then
    plan_cmd+=(
      --smoke-test
      --no-wandb
      --screen-steps "${SMOKE_SCREEN_STEPS}"
      --effective-step-tokens "${SMOKE_EFFECTIVE_STEP_TOKENS}"
      --confirm-steps "${SMOKE_CONFIRM_STEPS}"
      --confirm-hold-steps "${SMOKE_HOLD_STEPS}"
      --no-confirm-full-val-final
      --variant-steps "${SMOKE_VARIANT_STEPS}"
      --variant-hold-steps "${SMOKE_HOLD_STEPS}"
      --no-variant-full-val-final
      --screen-batch-size "${SMOKE_BATCH_SIZE}"
      --bptt-batch-size "${SMOKE_BPTT_BATCH_SIZE}"
      --bptt-chunks "${SMOKE_BPTT_CHUNKS}"
      --seq-len "${SMOKE_SEQ_LEN}"
      --val-steps "${SMOKE_VAL_STEPS}"
      --trigram-max-tokens "${SMOKE_TRIGRAM_MAX_TOKENS}"
      --data-max-tokens "${SMOKE_DATA_MAX_TOKENS}"
    )
  fi
  run_logged "plan_${stage}" "${plan_cmd[@]}"
  cp "${LOG_DIR}/plan_${stage}.log" "${out_md}"
  PLAN_SCRIPT="${out_script}"
}

run_plan_script() {
  local stage="$1"
  local script_path="$2"
  echo
  echo "=== execute_${stage} ==="
  if [[ -f "${script_path}" ]]; then
    sed -n '1,220p' "${script_path}"
  else
    echo "Missing generated plan script: ${script_path}" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  bash "${script_path}" 2>&1 | tee "${LOG_DIR}/execute_${stage}.log"
}

cat <<EOF | tee "${LOG_DIR}/header.txt"
5090 adaptive closeout runner
repo_root=${REPO_ROOT}
python=${PYTHON_BIN}
run_version=${RUN_VERSION}
seed=${SEED}
frontier_batch_id=${FRONTIER_BATCH_ID}
benchmark_json=${BENCHMARK_JSON}
run_benchmark=${RUN_BENCHMARK}
max_confirmations=${MAX_CONFIRMATIONS}
stop_after=${STOP_AFTER}
count_workers=${TRIGRAM_COUNT_WORKERS}
log_dir=${LOG_DIR}
cublaslt=${TORCH_BLAS_PREFER_CUBLASLT}
dry_run=${DRY_RUN}
smoke_test=${SMOKE_TEST}
EOF

if should_run_stage frontier; then
  frontier_cmd=(
    bash "${SCRIPT_DIR}/run_5090_final3day_frontier_batch.sh"
    --frontier-batch-id "${FRONTIER_BATCH_ID}"
    --run-version "${RUN_VERSION}"
    --seeds "${SEED}"
    --trigram-top-k "2"
    --count-workers "${TRIGRAM_COUNT_WORKERS}"
    --benchmark-json "${BENCHMARK_JSON}"
  )
  if [[ "${RUN_BENCHMARK}" == "1" ]]; then
    frontier_cmd+=(--run-benchmark)
  else
    frontier_cmd+=(--no-run-benchmark)
  fi
  if [[ "${SMOKE_TEST}" == "1" ]]; then
    frontier_cmd+=(
      --smoke-test
      --geometry-specs "${SMOKE_GEOMETRY_SPECS}"
      --num-steps "${SMOKE_SCREEN_STEPS}"
      --lr-hold-steps "${SMOKE_HOLD_STEPS}"
      --batch-size "${SMOKE_BATCH_SIZE}"
      --seq-len "${SMOKE_SEQ_LEN}"
      --target-effective-step-tokens "${SMOKE_EFFECTIVE_STEP_TOKENS}"
      --val-every "${SMOKE_VAL_EVERY}"
      --val-steps "${SMOKE_VAL_STEPS}"
      --log-every "${SMOKE_LOG_EVERY}"
      --log-state-every "${SMOKE_LOG_STATE_EVERY}"
      --save-every "${SMOKE_SAVE_EVERY}"
      --trigram-max-tokens "${SMOKE_TRIGRAM_MAX_TOKENS}"
      --data-max-tokens "${SMOKE_DATA_MAX_TOKENS}"
      --geometry-train-label smoke_screen
      --no-wandb
    )
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    frontier_cmd+=(--dry-run)
  fi
  run_logged frontier "${frontier_cmd[@]}"
fi

if should_run_stage confirm; then
  write_plan confirmations
  run_plan_script confirmations "${PLAN_SCRIPT}"
fi

if should_run_stage bptt; then
  write_plan bptt
  run_plan_script bptt "${PLAN_SCRIPT}"
fi

if should_run_stage k4; then
  write_plan k4
  run_plan_script k4 "${PLAN_SCRIPT}"
fi

echo
echo "Adaptive closeout runner complete. Logs: ${LOG_DIR}"
