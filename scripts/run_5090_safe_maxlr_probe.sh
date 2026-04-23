#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

SEEDS="${SEEDS:-1337}"
LRS="${LRS:-0.0025 0.0030 0.0035}"
RUN_BLOCKS1="${RUN_BLOCKS1:-1}"
RUN_BLOCKS0="${RUN_BLOCKS0:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PRESET="${PRESET:-controller_default}"
export COMPILE="${COMPILE:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export REBUILD_SHARED="${REBUILD_SHARED:-0}"
export TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-131072}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
export VAL_EVERY="${VAL_EVERY:-256}"
export VAL_STEPS="${VAL_STEPS:-8}"
export LOG_EVERY="${LOG_EVERY:-64}"
export LOG_STATE_EVERY="${LOG_STATE_EVERY:-256}"
export SAVE_EVERY="${SAVE_EVERY:-2048}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.98}"
export BRANCH_TEMPORAL_MODE="${BRANCH_TEMPORAL_MODE:-current}"
export RESIDUAL_TOKEN_GATE_MODE="${RESIDUAL_TOKEN_GATE_MODE:-none}"
export BRANCH_ROUTER_MODE="${BRANCH_ROUTER_MODE:-none}"
export SCAN_BACKEND="${SCAN_BACKEND:-auto}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pg-core-amp}"

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

lr_tag() {
  local lr="$1"
  local tag="${lr//./p}"
  tag="${tag#0p}"
  printf 'lr%s' "${tag}"
}

print_header() {
  echo "5090 safe max_lr probe"
  echo "repo_root=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "seeds=${SEEDS}"
  echo "lrs=${LRS}"
  echo "run_blocks1=${RUN_BLOCKS1} run_blocks0=${RUN_BLOCKS0}"
  echo "compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}"
  echo "target_effective_step_tokens=${TARGET_EFFECTIVE_STEP_TOKENS}"
  echo "branch_temporal_mode=${BRANCH_TEMPORAL_MODE} residual_token_gate_mode=${RESIDUAL_TOKEN_GATE_MODE}"
  echo "branch_router_mode=${BRANCH_ROUTER_MODE} scan_backend=${SCAN_BACKEND}"
  echo "wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT} py_unbuffered=${PYTHONUNBUFFERED}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

build_run_specs() {
  local prefix="$1"
  local core_layers="$2"
  local core_expansion="$3"
  local seed="$4"
  local lr
  local run_specs=""
  local tag=""

  for lr in ${LRS}; do
    tag="$(lr_tag "${lr}")"
    run_specs+="${prefix}_${tag}_h3500_512m_s${seed} ${core_layers} ${core_expansion} 8 1 1 -3.0 ${lr} 100 3500 0.0003 4096 256 512"$'\n'
  done

  printf '%s' "${run_specs%$'\n'}"
}

run_family_seed() {
  local family="$1"
  local seed="$2"
  local shared_spec_dir="$3"
  local model_root="$4"
  local wandb_group="$5"
  local wandb_tags="$6"
  local prefix="$7"
  local core_layers="$8"
  local core_expansion="$9"
  local run_specs

  require_dir "${shared_spec_dir}"
  run_specs="$(build_run_specs "${prefix}" "${core_layers}" "${core_expansion}" "${seed}")"

  echo
  echo "[${family}] seed=${seed} model_root=${model_root}"
  env \
    SEED="${seed}" \
    SHARED_SPEC_DIR="${shared_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="${wandb_group}" \
    WANDB_TAGS="${wandb_tags}" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

run_blocks1_seed() {
  local seed="$1"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks1_maxlr_probe_seed${seed}_v1"
  run_family_seed \
    "blocks1" \
    "${seed}" \
    "${shared_spec_dir}" \
    "${model_root}" \
    "blocks1_maxlr_probe512m_v1" \
    "core_amp,5090,final_week,safe_lane,maxlr,screening,blocks1" \
    "blocks1_resid10_e12" \
    "10" \
    "12.0"
}

run_blocks0_seed() {
  local seed="$1"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1/blocks0_resid12_e10_h7000_1b"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks0_maxlr_probe_seed${seed}_v1"
  run_family_seed \
    "blocks0" \
    "${seed}" \
    "${shared_spec_dir}" \
    "${model_root}" \
    "blocks0_maxlr_probe512m_v1" \
    "core_amp,5090,final_week,safe_lane,maxlr,screening,blocks0" \
    "blocks0_resid12_e10" \
    "12" \
    "10.0"
}

main() {
  local seed
  print_header

  for seed in ${SEEDS}; do
    if [[ "${RUN_BLOCKS1}" == "1" ]]; then
      run_blocks1_seed "${seed}"
    fi
    if [[ "${RUN_BLOCKS0}" == "1" ]]; then
      run_blocks0_seed "${seed}"
    fi
  done
}

main "$@"
