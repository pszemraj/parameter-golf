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
START_AT="frontier"
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
FINALIST_LABEL="blocks0_d128_l5_i512"
K6_RUN_VERSION="geom1_seq2048_bptt2_k6"
K7_PREFLIGHT_RUN_VERSION="geom1_seq2048_bptt2_k7_preflight"
K7_RUN_VERSION="geom1_seq2048_bptt2_k7"
SEQ4096_RUN_VERSION="geom1_seq4096_k6_probe"
K7_PREFLIGHT_MAX_BYTES="15500000"
K7_PROMOTION_BPB="0.004"
PREFLIGHT_TRAINABLE_PAYLOAD_BYTES="1267367"
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
  5. optionally run K7 finalist preflight, gated K7 training, stability, and seq4096 probe

Options:
  --run-version VALUE          default: geom1
  --seed VALUE                 default: 1337
  --frontier-batch-id VALUE    default: geom1
  --benchmark-json PATH
  --count-workers VALUE        trigram count workers for any cache miss
  --max-confirmations VALUE    default: 2
  --start-at VALUE             frontier|confirm|bptt|k4|k7-preflight|k7|k7-stability|seq4096
  --stop-after VALUE           frontier|confirm|bptt|k4|k7-preflight|k7|k7-stability|seq4096
  --baseline-bpb VALUE
  --finalist-label VALUE       default: blocks0_d128_l5_i512
  --k6-run-version VALUE       default: geom1_seq2048_bptt2_k6
  --k7-preflight-run-version VALUE
  --k7-run-version VALUE
  --seq4096-run-version VALUE
  --k7-preflight-max-bytes VALUE
  --k7-promotion-bpb VALUE
  --preflight-trainable-payload-bytes VALUE
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
    --start-at) START_AT="$2"; shift 2 ;;
    --stop-after) STOP_AFTER="$2"; shift 2 ;;
    --baseline-bpb) BASELINE_BPB="$2"; shift 2 ;;
    --finalist-label) FINALIST_LABEL="$2"; shift 2 ;;
    --k6-run-version) K6_RUN_VERSION="$2"; shift 2 ;;
    --k7-preflight-run-version) K7_PREFLIGHT_RUN_VERSION="$2"; shift 2 ;;
    --k7-run-version) K7_RUN_VERSION="$2"; shift 2 ;;
    --seq4096-run-version) SEQ4096_RUN_VERSION="$2"; shift 2 ;;
    --k7-preflight-max-bytes) K7_PREFLIGHT_MAX_BYTES="$2"; shift 2 ;;
    --k7-promotion-bpb) K7_PROMOTION_BPB="$2"; shift 2 ;;
    --preflight-trainable-payload-bytes) PREFLIGHT_TRAINABLE_PAYLOAD_BYTES="$2"; shift 2 ;;
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

for boundary in "${START_AT}" "${STOP_AFTER}"; do
  case "${boundary}" in
    frontier|confirm|bptt|k4|k7-preflight|k7|k7-stability|seq4096) ;;
    *)
      echo "Invalid adaptive boundary: ${boundary}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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
    k7-preflight) echo 5 ;;
    k7) echo 6 ;;
    k7-stability) echo 7 ;;
    seq4096) echo 8 ;;
    *) echo 999 ;;
  esac
}

if [[ "$(stage_rank "${START_AT}")" -gt "$(stage_rank "${STOP_AFTER}")" ]]; then
  echo "--start-at must not be after --stop-after: ${START_AT} > ${STOP_AFTER}" >&2
  exit 2
fi

should_run_stage() {
  local stage="$1"
  [[ "$(stage_rank "${stage}")" -ge "$(stage_rank "${START_AT}")" && "$(stage_rank "${stage}")" -le "$(stage_rank "${STOP_AFTER}")" ]]
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

write_finalist_plan() {
  local logical_stage="$1"
  shift
  local out_script="${LOG_DIR}/${logical_stage}.sh"
  local out_md="${LOG_DIR}/${logical_stage}.md"
  local plan_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/plan_5090_adaptive_closeout.py" \
      --stage finalist \
      --seed "${SEED}" \
      --count-workers "${TRIGRAM_COUNT_WORKERS}" \
      --write-script "${out_script}" \
      --emit markdown \
      "$@"
  )
  if [[ "${SMOKE_TEST}" == "1" ]]; then
    plan_cmd+=(--smoke-test --no-wandb)
  fi
  run_logged "plan_${logical_stage}" "${plan_cmd[@]}"
  cp "${LOG_DIR}/plan_${logical_stage}.log" "${out_md}"
  PLAN_SCRIPT="${out_script}"
}

run_plan_script() {
  local stage="$1"
  local script_path="$2"
  local status
  echo
  status="$(sed -n 's/^# adaptive_status=//p' "${script_path}" 2>/dev/null | head -n 1)"
  echo "=== planned_${stage} status=${status:-unknown} ==="
  if [[ -f "${script_path}" ]]; then
    sed -n '1,220p' "${script_path}"
  else
    echo "Missing generated plan script: ${script_path}" >&2
    exit 1
  fi
  case "${status}" in
    commands)
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN=1: not executing ${stage}."
        return 0
      fi
      echo
      echo "=== execute_${stage} ==="
      bash "${script_path}" 2>&1 | tee "${LOG_DIR}/execute_${stage}.log"
      ;;
    already_complete|not_selected)
      echo "No execution needed for ${stage}: ${status}"
      ;;
    blocked)
      echo "Planner blocked ${stage}; refusing to continue." >&2
      return 2
      ;;
    *)
      echo "Unknown adaptive_status for ${stage}: ${status:-empty}" >&2
      return 2
      ;;
  esac
}

run_results_path() {
  local seed="$1"
  local run_version="$2"
  FINALIST_LABEL="${FINALIST_LABEL}" SEED_VALUE="${seed}" RUN_VERSION_VALUE="${run_version}" REPO_ROOT="${REPO_ROOT}" \
    "${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

root = (
    Path(os.environ["REPO_ROOT"])
    / "experiments"
    / "5090_architecture"
    / f"{os.environ['FINALIST_LABEL']}_trigram_seed{os.environ['SEED_VALUE']}_{os.environ['RUN_VERSION_VALUE']}"
)
paths = list(root.rglob("run_results.json"))
if not paths:
    sys.exit(f"run_results.json not found under {root}")
print(max(paths, key=lambda p: p.stat().st_mtime))
PY
}

json_field() {
  local path="$1"
  local field="$2"
  FIELD="${field}" "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(os.environ["FIELD"])
if value is None:
    sys.exit(f"missing field {os.environ['FIELD']} in {sys.argv[1]}")
print(value)
PY
}

float_compare() {
  local lhs="$1"
  local op="$2"
  local rhs="$3"
  "${PYTHON_BIN}" - "${lhs}" "${op}" "${rhs}" <<'PY'
import operator
import sys

lhs = float(sys.argv[1])
op = sys.argv[2]
rhs = float(sys.argv[3])
ops = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
raise SystemExit(0 if ops[op](lhs, rhs) else 1)
PY
}

bpb_improvement() {
  local baseline="$1"
  local candidate="$2"
  "${PYTHON_BIN}" - "${baseline}" "${candidate}" <<'PY'
import sys

print(float(sys.argv[1]) - float(sys.argv[2]))
PY
}

k7_preflight_json() {
  COREAMP_SPEC_CACHE_ROOT="${COREAMP_SPEC_CACHE_ROOT:-${HOME}/.cache/experiments/param-golf-coreamp}" \
  K7_PREFLIGHT_RUN_VERSION="${K7_PREFLIGHT_RUN_VERSION}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["COREAMP_SPEC_CACHE_ROOT"]).expanduser() / "shared_specs"
run_version = os.environ["K7_PREFLIGHT_RUN_VERSION"]
matches = []
for path in root.glob("*/artifact_preflight.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if payload.get("run_version") == run_version:
        matches.append(path)
if not matches:
    sys.exit(f"artifact_preflight.json not found for {run_version}")
print(max(matches, key=lambda p: p.stat().st_mtime))
PY
}

K7_ALLOWED="1"

check_k7_preflight_gate() {
  local preflight_rc="${1:-0}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    K7_ALLOWED="1"
    return 0
  fi

  local preflight_path artifact_status artifact_bytes
  if ! preflight_path="$(k7_preflight_json)"; then
    if [[ "${preflight_rc}" != "0" ]]; then
      return "${preflight_rc}"
    fi
    return 2
  fi
  artifact_status="$(json_field "${preflight_path}" artifact_status)"
  artifact_bytes="$(json_field "${preflight_path}" artifact_estimate_bytes)"

  echo
  echo "=== k7_preflight_gate ==="
  echo "artifact_preflight=${preflight_path}"
  echo "artifact_status=${artifact_status}"
  echo "artifact_estimate_bytes=${artifact_bytes}"
  echo "k7_preflight_max_bytes=${K7_PREFLIGHT_MAX_BYTES}"

  K7_ALLOWED="1"
  if [[ "${artifact_status}" == "OVER_LIMIT" ]]; then
    K7_ALLOWED="0"
  elif [[ "${artifact_status}" != "LEFT_ON_TABLE" && "${artifact_status}" != "UNDER_LIMIT" && "${artifact_status}" != "EXACT_LIMIT" ]]; then
    echo "K7 preflight artifact status is not evidence-grade: ${artifact_status}" >&2
    return 2
  elif ! float_compare "${artifact_bytes}" "<=" "${K7_PREFLIGHT_MAX_BYTES}"; then
    K7_ALLOWED="0"
  fi

  if [[ "${K7_ALLOWED}" == "0" ]]; then
    echo "K7 training skipped by artifact gate."
    return 0
  fi

  if [[ "${preflight_rc}" != "0" ]]; then
    echo "K7 preflight command failed despite an allowable artifact estimate." >&2
    return "${preflight_rc}"
  fi
  echo "K7 artifact gate passed."
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
start_at=${START_AT}
stop_after=${STOP_AFTER}
count_workers=${TRIGRAM_COUNT_WORKERS}
finalist_label=${FINALIST_LABEL}
k6_run_version=${K6_RUN_VERSION}
k7_preflight_run_version=${K7_PREFLIGHT_RUN_VERSION}
k7_run_version=${K7_RUN_VERSION}
seq4096_run_version=${SEQ4096_RUN_VERSION}
k7_preflight_max_bytes=${K7_PREFLIGHT_MAX_BYTES}
k7_promotion_bpb=${K7_PROMOTION_BPB}
preflight_trainable_payload_bytes=${PREFLIGHT_TRAINABLE_PAYLOAD_BYTES}
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

if should_run_stage k7-preflight; then
  write_finalist_plan k7-preflight \
    --run-version "${K7_PREFLIGHT_RUN_VERSION}" \
    --label "${FINALIST_LABEL}" \
    --finalist-run-version "${K7_PREFLIGHT_RUN_VERSION}" \
    --finalist-seeds "${SEED}" \
    --finalist-trigram-top-k 7 \
    --finalist-seq-len 2048 \
    --finalist-batch-size 32 \
    --finalist-bptt-chunks 2 \
    --finalist-steps 8192 \
    --finalist-hold-steps 7000 \
    --finalist-train-label preflight_seq2048_bptt2_k7 \
    --finalist-preflight-only \
    --preflight-trainable-payload-bytes "${PREFLIGHT_TRAINABLE_PAYLOAD_BYTES}"
  set +e
  run_plan_script k7-preflight "${PLAN_SCRIPT}"
  preflight_rc=$?
  set -e
  check_k7_preflight_gate "${preflight_rc}"
elif should_run_stage k7 || should_run_stage k7-stability; then
  check_k7_preflight_gate 0
fi

if should_run_stage k7; then
  if [[ "${K7_ALLOWED}" == "1" ]]; then
    write_finalist_plan k7 \
      --run-version "${K7_RUN_VERSION}" \
      --label "${FINALIST_LABEL}" \
      --finalist-run-version "${K7_RUN_VERSION}" \
      --finalist-seeds "${SEED}" \
      --finalist-trigram-top-k 7 \
      --finalist-seq-len 2048 \
      --finalist-batch-size 32 \
      --finalist-bptt-chunks 2 \
      --finalist-steps 8192 \
      --finalist-hold-steps 7000 \
      --finalist-train-label 1b_seq2048_bptt2_k7
    run_plan_script k7 "${PLAN_SCRIPT}"
  else
    echo
    echo "=== k7 skipped ==="
    echo "K7 artifact gate did not pass."
  fi
fi

if should_run_stage k7-stability; then
  if [[ "${K7_ALLOWED}" != "1" ]]; then
    echo
    echo "=== k7-stability skipped ==="
    echo "K7 artifact gate did not pass."
  elif [[ "${DRY_RUN}" == "1" ]]; then
    write_finalist_plan k7-stability \
      --run-version "${K7_RUN_VERSION}" \
      --label "${FINALIST_LABEL}" \
      --finalist-run-version "${K7_RUN_VERSION}" \
      --finalist-seeds "1337 2027 3141" \
      --finalist-stability-check \
      --finalist-trigram-top-k 7 \
      --finalist-seq-len 2048 \
      --finalist-batch-size 32 \
      --finalist-bptt-chunks 2 \
      --finalist-steps 8192 \
      --finalist-hold-steps 7000 \
      --finalist-train-label 1b_seq2048_bptt2_k7
    run_plan_script k7-stability "${PLAN_SCRIPT}"
  else
    k6_results="$(run_results_path "${SEED}" "${K6_RUN_VERSION}")"
    k7_results="$(run_results_path "${SEED}" "${K7_RUN_VERSION}")"
    k6_bpb="$(json_field "${k6_results}" last_val_bpb)"
    k7_bpb="$(json_field "${k7_results}" last_val_bpb)"
    improvement="$(bpb_improvement "${k6_bpb}" "${k7_bpb}")"
    echo
    echo "=== k7_promotion_gate ==="
    echo "k6_results=${k6_results}"
    echo "k7_results=${k7_results}"
    echo "k6_last_val_bpb=${k6_bpb}"
    echo "k7_last_val_bpb=${k7_bpb}"
    echo "improvement_bpb=${improvement}"
    echo "required_improvement_bpb=${K7_PROMOTION_BPB}"
    if float_compare "${improvement}" ">=" "${K7_PROMOTION_BPB}"; then
      write_finalist_plan k7-stability \
        --run-version "${K7_RUN_VERSION}" \
        --label "${FINALIST_LABEL}" \
        --finalist-run-version "${K7_RUN_VERSION}" \
        --finalist-seeds "1337 2027 3141" \
        --finalist-stability-check \
        --finalist-trigram-top-k 7 \
        --finalist-seq-len 2048 \
        --finalist-batch-size 32 \
        --finalist-bptt-chunks 2 \
        --finalist-steps 8192 \
        --finalist-hold-steps 7000 \
        --finalist-train-label 1b_seq2048_bptt2_k7
      run_plan_script k7-stability "${PLAN_SCRIPT}"
    else
      echo "K7 stability skipped by BPB promotion gate."
    fi
  fi
fi

if should_run_stage seq4096; then
  write_finalist_plan seq4096 \
    --run-version "${SEQ4096_RUN_VERSION}" \
    --label "${FINALIST_LABEL}" \
    --finalist-run-version "${SEQ4096_RUN_VERSION}" \
    --finalist-seeds "${SEED}" \
    --finalist-trigram-top-k 6 \
    --finalist-seq-len 4096 \
    --finalist-batch-size 32 \
    --finalist-bptt-chunks 1 \
    --finalist-steps 4096 \
    --finalist-hold-steps 3500 \
    --finalist-train-label 512m_seq4096_k6_probe
  run_plan_script seq4096 "${PLAN_SCRIPT}"
fi

echo
echo "Adaptive closeout runner complete. Logs: ${LOG_DIR}"
