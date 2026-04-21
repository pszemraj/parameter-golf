#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

SEEDS="${SEEDS:-1337 2027 3141}"
RUN_BLOCKS1="${RUN_BLOCKS1:-1}"
RUN_BLOCKS0="${RUN_BLOCKS0:-1}"
RUN_BLOCKS2="${RUN_BLOCKS2:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PRESET="${PRESET:-controller_default}"
export COMPILE="${COMPILE:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export REBUILD_SHARED="${REBUILD_SHARED:-0}"
export VAL_EVERY="${VAL_EVERY:-512}"
export VAL_STEPS="${VAL_STEPS:-8}"
export LOG_EVERY="${LOG_EVERY:-128}"
export LOG_STATE_EVERY="${LOG_STATE_EVERY:-512}"
export SAVE_EVERY="${SAVE_EVERY:-4096}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.98}"
export BRANCH_TEMPORAL_MODE="${BRANCH_TEMPORAL_MODE:-current}"
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

print_header() {
  echo "5090 wide confirmation batch"
  echo "repo_root=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "seeds=${SEEDS}"
  echo "run_blocks1=${RUN_BLOCKS1} run_blocks0=${RUN_BLOCKS0} run_blocks2=${RUN_BLOCKS2}"
  echo "compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}"
  echo "wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

run_blocks1_seed() {
  local seed="$1"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid12_e10_h7000_1b"
  local model_root="${REPO_ROOT}/experiments/5090_schedule/blocks1_wide_confirm1b_seed${seed}_v1"
  local run_specs

  require_dir "$shared_spec_dir"

  read -r -d '' run_specs <<EOF || true
blocks1_resid12_e10_h7000_1b_s${seed} 12 10.0 8 1 1 -3.0 0.003 100 7000 0.0003 8192 256 512
blocks1_resid10_e12_h7000_1b_s${seed} 10 12.0 8 1 1 -3.0 0.003 100 7000 0.0003 8192 256 512
EOF

  echo
  echo "[blocks1] seed=${seed} model_root=${model_root}"
  env \
    SEED="${seed}" \
    SHARED_SPEC_DIR="${shared_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="blocks1_wide_confirm1b_v1" \
    WANDB_TAGS="core_amp,5090,confirmation,multiseed,blocks1" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

run_blocks0_seed() {
  local seed="$1"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_structure/fullspec_blocks0_radical_v1/blocks0_resid12_e6_c8t1_r3_current_512m"
  local model_root="${REPO_ROOT}/experiments/5090_schedule/blocks0_wide_confirm1b_seed${seed}_v1"
  local run_specs

  require_dir "$shared_spec_dir"

  read -r -d '' run_specs <<EOF || true
blocks0_resid10_e12_h7000_1b_s${seed} 10 12.0 8 1 1 -3.0 0.003 100 7000 0.0003 8192 256 512
blocks0_resid12_e10_h7000_1b_s${seed} 12 10.0 8 1 1 -3.0 0.003 100 7000 0.0003 8192 256 512
EOF

  echo
  echo "[blocks0] seed=${seed} model_root=${model_root}"
  env \
    SEED="${seed}" \
    SHARED_SPEC_DIR="${shared_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="blocks0_wide_confirm1b_v1" \
    WANDB_TAGS="core_amp,5090,confirmation,multiseed,blocks0" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

run_blocks2_seed() {
  local seed="$1"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_controller/fullspec_blocks2_radical_v1/blocks2_resid12_e8_c8t1_r3_current_512m"
  local model_root="${REPO_ROOT}/experiments/5090_schedule/blocks2_wide_compare1b_seed${seed}_v1"
  local run_specs

  require_dir "$shared_spec_dir"

  read -r -d '' run_specs <<EOF || true
blocks2_resid12_e8_h7000_1b_s${seed} 12 8.0 8 1 1 -3.0 0.003 100 7000 0.0003 8192 256 512
EOF

  echo
  echo "[blocks2] seed=${seed} model_root=${model_root}"
  env \
    SEED="${seed}" \
    SHARED_SPEC_DIR="${shared_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="blocks2_wide_compare1b_v1" \
    WANDB_TAGS="core_amp,5090,confirmation,multiseed,blocks2" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
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
    if [[ "${RUN_BLOCKS2}" == "1" ]]; then
      run_blocks2_seed "${seed}"
    fi
  done
}

main "$@"
