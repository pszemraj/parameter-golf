#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=""
DRY_RUN="0"
PLAN_ARGS=()
PLAN_SCRIPT=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [runner options] -- [planner options]

Run the generic adaptive gated-followup stage. The planner options are passed to
tools/plan_5090_adaptive_closeout.py with --stage gated-followup.

Runner options:
  --run-id VALUE
  --log-dir PATH
  --dry-run

Required planner options for gated follow-up:
  --run-version VALUE
  --gate-evidence-run-version VALUE
  --gate-followup-run-version VALUE

Useful planner contract options:
  --label VALUE
  --seed VALUE
  --benchmark PATH
  --count-workers VALUE
  --gate-max-worse-bpb VALUE
  --gate-evidence-trigram-top-k VALUE
  --gate-evidence-seq-len VALUE
  --gate-evidence-batch-size VALUE
  --gate-evidence-bptt-chunks VALUE
  --gate-followup-trigram-top-k VALUE
  --gate-followup-seq-len VALUE
  --gate-followup-batch-size VALUE
  --gate-followup-bptt-chunks VALUE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PLAN_ARGS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PLAN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${REPO_ROOT}/logs/5090_gated_followup/${RUN_ID}"
fi

export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p "${LOG_DIR}"

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

plan_once() {
  local pass="$1"
  local out_script="${LOG_DIR}/pass${pass}.sh"
  local log_path="${LOG_DIR}/plan_pass${pass}.log"
  local plan_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/plan_5090_adaptive_closeout.py"
    --stage gated-followup
    "${PLAN_ARGS[@]}"
    --write-script "${out_script}"
    --emit markdown
  )
  echo
  echo "=== plan_pass${pass} ==="
  print_cmd "${plan_cmd[@]}"
  "${plan_cmd[@]}" 2>&1 | tee "${log_path}"
  PLAN_SCRIPT="${out_script}"
}

script_status() {
  local script_path="$1"
  sed -n 's/^# adaptive_status=//p' "${script_path}" | head -n 1
}

execute_script_if_needed() {
  local pass="$1"
  local script_path="$2"
  local status
  status="$(script_status "${script_path}")"
  echo
  echo "=== planned_pass${pass} status=${status:-unknown} ==="
  sed -n '1,220p' "${script_path}"
  if [[ "${status}" != "commands" ]]; then
    return 1
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 1
  fi
  echo
  echo "=== execute_pass${pass} ==="
  bash "${script_path}" 2>&1 | tee "${LOG_DIR}/execute_pass${pass}.log"
  return 0
}

cat <<EOF | tee "${LOG_DIR}/header.txt"
5090 gated follow-up runner
repo_root=${REPO_ROOT}
python=${PYTHON_BIN}
run_id=${RUN_ID}
log_dir=${LOG_DIR}
cublaslt=${TORCH_BLAS_PREFER_CUBLASLT}
dry_run=${DRY_RUN}
planner_args=${PLAN_ARGS[*]}
EOF

plan_once 1
if execute_script_if_needed 1 "${PLAN_SCRIPT}"; then
  plan_once 2
  execute_script_if_needed 2 "${PLAN_SCRIPT}" || true
fi

echo
echo "Gated follow-up runner complete. Logs: ${LOG_DIR}"
