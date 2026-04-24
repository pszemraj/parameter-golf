#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"
source "${SCRIPT_DIR}/5090_common.sh"

SEEDS="${SEEDS:-1337}"
RUN_VERSION="${RUN_VERSION:-v1}"
RANKS="${RANKS:-128 256}"
RUN_BLOCKS1="${RUN_BLOCKS1:-1}"
RUN_BLOCKS0="${RUN_BLOCKS0:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0035}"
LEARNING_RATE_TAG="$(pg_5090_lr_slug "${LEARNING_RATE}")"
RESIDUAL_READOUT_DELTA_INIT_STD="${RESIDUAL_READOUT_DELTA_INIT_STD:-0.02}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PRESET="${PRESET:-controller_default}"
export COMPILE="${COMPILE:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export REBUILD_SHARED="${REBUILD_SHARED:-0}"
export CORE_AMP_PHASE="${CORE_AMP_PHASE:-5090_readout_delta_screen}"
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
  echo "5090 readout delta screen"
  echo "repo_root=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "seeds=${SEEDS}"
  echo "run_version=${RUN_VERSION}"
  echo "ranks=${RANKS}"
  echo "delta_init_std=${RESIDUAL_READOUT_DELTA_INIT_STD}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "run_blocks1=${RUN_BLOCKS1} run_blocks0=${RUN_BLOCKS0}"
  echo "compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}"
  echo "target_effective_step_tokens=${TARGET_EFFECTIVE_STEP_TOKENS}"
  echo "branch_temporal_mode=${BRANCH_TEMPORAL_MODE} residual_token_gate_mode=${RESIDUAL_TOKEN_GATE_MODE} branch_router_mode=${BRANCH_ROUTER_MODE}"
  echo "scan_backend=${SCAN_BACKEND} wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT} py_unbuffered=${PYTHONUNBUFFERED}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

run_family_seed_rank() {
  local family="$1"
  local seed="$2"
  local rank="$3"
  local shared_spec_dir="$4"
  local model_root="$5"
  local wandb_group="$6"
  local wandb_tags="$7"
  local run_name="$8"
  local core_layers="$9"
  local core_expansion="${10}"

  require_dir "${shared_spec_dir}"

  local run_specs
  read -r -d '' run_specs <<EOF || true
${run_name} ${core_layers} ${core_expansion} 8 1 1 -3.0 ${LEARNING_RATE} 100 3500 0.0003 4096 256 512
EOF

  echo
  echo "[${family}] seed=${seed} readout_delta_rank=${rank} lr=${LEARNING_RATE} model_root=${model_root}"
  env \
    SEED="${seed}" \
    RESIDUAL_READOUT_DELTA_RANK="${rank}" \
    RESIDUAL_READOUT_DELTA_INIT_STD="${RESIDUAL_READOUT_DELTA_INIT_STD}" \
    SHARED_SPEC_DIR="${shared_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="${wandb_group}" \
    WANDB_TAGS="${wandb_tags}" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

run_blocks1_seed_rank() {
  local seed="$1"
  local rank="$2"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks1_readout_delta_seed${seed}_${RUN_VERSION}"
  run_family_seed_rank \
    "blocks1" \
    "${seed}" \
    "${rank}" \
    "${shared_spec_dir}" \
    "${model_root}" \
    "blocks1_readout_delta512m_${RUN_VERSION}" \
    "core_amp,5090,architecture,readout_delta,screening,blocks1,rank_${rank},${LEARNING_RATE_TAG}" \
    "blocks1_resid10_e12_delta_r${rank}_${LEARNING_RATE_TAG}_h3500_512m_s${seed}" \
    "10" \
    "12.0"
}

run_blocks0_seed_rank() {
  local seed="$1"
  local rank="$2"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1/blocks0_resid12_e10_h7000_1b"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks0_readout_delta_seed${seed}_${RUN_VERSION}"
  run_family_seed_rank \
    "blocks0" \
    "${seed}" \
    "${rank}" \
    "${shared_spec_dir}" \
    "${model_root}" \
    "blocks0_readout_delta512m_${RUN_VERSION}" \
    "core_amp,5090,architecture,readout_delta,screening,blocks0,rank_${rank},${LEARNING_RATE_TAG}" \
    "blocks0_resid12_e10_delta_r${rank}_${LEARNING_RATE_TAG}_h3500_512m_s${seed}" \
    "12" \
    "10.0"
}

main() {
  local seed rank
  print_header

  for seed in ${SEEDS}; do
    for rank in ${RANKS}; do
      if [[ "${RUN_BLOCKS1}" == "1" ]]; then
        run_blocks1_seed_rank "${seed}" "${rank}"
      fi
      if [[ "${RUN_BLOCKS0}" == "1" ]]; then
        run_blocks0_seed_rank "${seed}" "${rank}"
      fi
    done
  done
}

main "$@"
