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

Run the explicit finalist adaptive stage. Planner options are passed to
tools/plan_5090_adaptive_closeout.py with --stage finalist.

Runner options:
  --run-id VALUE
  --log-dir PATH
  --dry-run

Required planner options:
  --label VALUE
  --finalist-run-version VALUE
  --finalist-seeds VALUE
  --finalist-trigram-top-k VALUE
  --finalist-seq-len VALUE
  --finalist-batch-size VALUE
  --finalist-bptt-chunks VALUE

Useful planner options:
  --finalist-steps VALUE
  --finalist-hold-steps VALUE
  --finalist-train-label VALUE
  --finalist-preflight-only
  --count-workers VALUE
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
  LOG_DIR="${REPO_ROOT}/logs/5090_finalist_closeout/${RUN_ID}"
fi

export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
mkdir -p "${LOG_DIR}"

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

script_status() {
  local script_path="$1"
  sed -n 's/^# adaptive_status=//p' "${script_path}" | head -n 1
}

plan_once() {
  local out_script="${LOG_DIR}/finalist.sh"
  local log_path="${LOG_DIR}/plan_finalist.log"
  local plan_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/plan_5090_adaptive_closeout.py"
    --stage finalist
    "${PLAN_ARGS[@]}"
    --write-script "${out_script}"
    --emit markdown
  )
  echo
  echo "=== plan_finalist ==="
  print_cmd "${plan_cmd[@]}"
  "${plan_cmd[@]}" 2>&1 | tee "${log_path}"
  PLAN_SCRIPT="${out_script}"
}

run_planned_script() {
  local script_path="$1"
  local status
  status="$(script_status "${script_path}")"
  echo
  echo "=== planned_finalist status=${status:-unknown} ==="
  sed -n '1,220p' "${script_path}"
  case "${status}" in
    commands)
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN=1: not executing finalist stage."
        return 0
      fi
      echo
      echo "=== execute_finalist ==="
      bash "${script_path}" 2>&1 | tee "${LOG_DIR}/execute_finalist.log"
      ;;
    already_complete|not_selected)
      echo "No execution needed for finalist stage: ${status}"
      ;;
    blocked)
      echo "Planner blocked finalist stage; refusing to continue." >&2
      return 2
      ;;
    *)
      echo "Unknown adaptive_status for finalist stage: ${status:-empty}" >&2
      return 2
      ;;
  esac
}

cat <<EOF | tee "${LOG_DIR}/header.txt"
5090 finalist closeout runner
repo_root=${REPO_ROOT}
python=${PYTHON_BIN}
run_id=${RUN_ID}
log_dir=${LOG_DIR}
cublaslt=${TORCH_BLAS_PREFER_CUBLASLT}
dry_run=${DRY_RUN}
planner_args=${PLAN_ARGS[*]}
EOF

plan_once
run_planned_script "${PLAN_SCRIPT}"

echo
echo "Finalist closeout runner complete. Logs: ${LOG_DIR}"
