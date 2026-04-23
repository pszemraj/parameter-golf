#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"
source "${SCRIPT_DIR}/5090_common.sh"

SEEDS="${SEEDS:-1337 2027 3141}"
RUN_VERSION="${RUN_VERSION:-v2}"
DEFAULT_LEARNING_RATE="${DEFAULT_LEARNING_RATE:-0.0035}"
DEFAULT_WARMUP_STEPS="${DEFAULT_WARMUP_STEPS:-100}"
DEFAULT_LR_HOLD_STEPS="${DEFAULT_LR_HOLD_STEPS:-7000}"
DEFAULT_MIN_LR="${DEFAULT_MIN_LR:-0.0003}"
DEFAULT_NUM_STEPS="${DEFAULT_NUM_STEPS:-8192}"
FINALIST_SPECS="${FINALIST_SPECS:-$'blocks1_resid10_e12_lr0035_final '"${REPO_ROOT}"$'/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b 10 12.0 none current none 0.0035\nblocks0_resid12_e10_lr0035_final '"${REPO_ROOT}"$'/experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1/blocks0_resid12_e10_h7000_1b 12 10.0 none current none 0.0035'}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PRESET="${PRESET:-controller_default}"
export COMPILE="${COMPILE:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export REBUILD_SHARED="${REBUILD_SHARED:-0}"
export CORE_AMP_PHASE="${CORE_AMP_PHASE:-5090_finalist_confirm1b}"
export TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-131072}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
export VAL_EVERY="${VAL_EVERY:-512}"
export VAL_STEPS="${VAL_STEPS:-8}"
export LOG_EVERY="${LOG_EVERY:-128}"
export LOG_STATE_EVERY="${LOG_STATE_EVERY:-512}"
export SAVE_EVERY="${SAVE_EVERY:-4096}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.98}"
export SCAN_BACKEND="${SCAN_BACKEND:-auto}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pg-hconv-ablations}"

pg_5090_require_serious_launcher_defaults "$(basename "$0")"

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
  if [[ ! -f "$path/spec.pt" ]]; then
    echo "Missing required shared spec: $path/spec.pt" >&2
    exit 1
  fi
  if [[ ! -f "$path/config.json" ]]; then
    echo "Missing required shared config: $path/config.json" >&2
    exit 1
  fi
}

print_header() {
  echo "5090 finalist 1B confirmation batch"
  echo "repo_root=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "seeds=${SEEDS}"
  echo "run_version=${RUN_VERSION}"
  echo "finalist_specs:"
  printf '%s\n' "${FINALIST_SPECS}"
  echo "default_learning_rate=${DEFAULT_LEARNING_RATE} default_warmup_steps=${DEFAULT_WARMUP_STEPS}"
  echo "default_lr_hold_steps=${DEFAULT_LR_HOLD_STEPS} default_min_lr=${DEFAULT_MIN_LR} default_num_steps=${DEFAULT_NUM_STEPS}"
  echo "compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}"
  echo "target_effective_step_tokens=${TARGET_EFFECTIVE_STEP_TOKENS}"
  echo "scan_backend=${SCAN_BACKEND} wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT} py_unbuffered=${PYTHONUNBUFFERED}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

run_finalist_seed() {
  local seed="$1"
  local name="$2"
  local shared_spec_dir="$3"
  local core_layers="$4"
  local core_expansion="$5"
  local gate_mode="$6"
  local temporal_mode="$7"
  local router_mode="$8"
  local learning_rate="$9"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/finalist_confirm1b_seed${seed}_${RUN_VERSION}"
  local run_name="${name}_h7000_1b_s${seed}"
  local run_specs

  require_dir "${shared_spec_dir}"
  read -r -d '' run_specs <<EOF || true
${run_name} ${core_layers} ${core_expansion} 8 1 1 -3.0 ${learning_rate} ${DEFAULT_WARMUP_STEPS} ${DEFAULT_LR_HOLD_STEPS} ${DEFAULT_MIN_LR} ${DEFAULT_NUM_STEPS} 256 512
EOF

  echo
  echo "[finalist] seed=${seed} run=${run_name} gate=${gate_mode} temporal=${temporal_mode} router=${router_mode} lr=${learning_rate}"
  env \
    SEED="${seed}" \
    RESIDUAL_TOKEN_GATE_MODE="${gate_mode}" \
    BRANCH_TEMPORAL_MODE="${temporal_mode}" \
    BRANCH_ROUTER_MODE="${router_mode}" \
    SHARED_SPEC_DIR="${shared_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="finalist_confirm1b_${RUN_VERSION}" \
    WANDB_TAGS="core_amp,5090,final_week,confirmation,1b,gate_${gate_mode},temporal_${temporal_mode},router_${router_mode}" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

main() {
  local seed line
  local name shared_spec_dir core_layers core_expansion gate_mode temporal_mode router_mode learning_rate
  print_header

  for seed in ${SEEDS}; do
    while IFS= read -r line; do
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      if [[ -z "${line}" || "${line}" == \#* ]]; then
        continue
      fi
      read -r name shared_spec_dir core_layers core_expansion gate_mode temporal_mode router_mode learning_rate <<<"${line}"
      learning_rate="${learning_rate:-${DEFAULT_LEARNING_RATE}}"
      if [[ -z "${router_mode:-}" ]]; then
        echo "Invalid FINALIST_SPECS line: ${line}" >&2
        exit 1
      fi
      run_finalist_seed \
        "${seed}" \
        "${name}" \
        "${shared_spec_dir}" \
        "${core_layers}" \
        "${core_expansion}" \
        "${gate_mode}" \
        "${temporal_mode}" \
        "${router_mode}" \
        "${learning_rate}"
    done <<<"${FINALIST_SPECS}"
  done
}

main "$@"
