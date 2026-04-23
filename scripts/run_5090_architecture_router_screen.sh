#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"
source "${SCRIPT_DIR}/5090_common.sh"

SEEDS="${SEEDS:-1337}"
RUN_VERSION="${RUN_VERSION:-v2}"
ROUTER_MODES="${ROUTER_MODES:-softmax}"
GATE_MODE="${GATE_MODE:-none}"
TEMPORAL_MODE="${TEMPORAL_MODE:-ema}"
INCLUDE_BASELINE_NONE="${INCLUDE_BASELINE_NONE:-0}"
RUN_BLOCKS1="${RUN_BLOCKS1:-1}"
RUN_BLOCKS0="${RUN_BLOCKS0:-0}"
RUN_BLOCKS2="${RUN_BLOCKS2:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PRESET="${PRESET:-controller_default}"
export COMPILE="${COMPILE:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export REBUILD_SHARED="${REBUILD_SHARED:-0}"
export CORE_AMP_PHASE="${CORE_AMP_PHASE:-5090_architecture_router_screen}"
export TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-131072}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
export VAL_EVERY="${VAL_EVERY:-256}"
export VAL_STEPS="${VAL_STEPS:-8}"
export LOG_EVERY="${LOG_EVERY:-64}"
export LOG_STATE_EVERY="${LOG_STATE_EVERY:-256}"
export SAVE_EVERY="${SAVE_EVERY:-2048}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.98}"
export RESIDUAL_TOKEN_GATE_MODE="${RESIDUAL_TOKEN_GATE_MODE:-${GATE_MODE}}"
export BRANCH_TEMPORAL_MODE="${BRANCH_TEMPORAL_MODE:-${TEMPORAL_MODE}}"
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

mode_slug() {
  local mode="$1"
  printf '%s' "${mode//-/_}"
}

router_list() {
  if [[ "${INCLUDE_BASELINE_NONE}" == "1" ]]; then
    printf 'none %s\n' "${ROUTER_MODES}"
  else
    printf '%s\n' "${ROUTER_MODES}"
  fi
}

print_header() {
  echo "5090 architecture router screen"
  echo "repo_root=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "seeds=${SEEDS}"
  echo "run_version=${RUN_VERSION}"
  echo "router_modes=$(router_list)"
  echo "gate_mode=${RESIDUAL_TOKEN_GATE_MODE} temporal_mode=${BRANCH_TEMPORAL_MODE}"
  echo "include_baseline_none=${INCLUDE_BASELINE_NONE}"
  echo "run_blocks1=${RUN_BLOCKS1} run_blocks0=${RUN_BLOCKS0} run_blocks2=${RUN_BLOCKS2}"
  echo "compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}"
  echo "target_effective_step_tokens=${TARGET_EFFECTIVE_STEP_TOKENS}"
  echo "scan_backend=${SCAN_BACKEND}"
  echo "wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT} py_unbuffered=${PYTHONUNBUFFERED}"
  if [[ "${INCLUDE_BASELINE_NONE}" != "1" ]]; then
    echo "note=reuse router=none baseline from the matching temporal winner when available"
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

run_family_seed_mode() {
  local family="$1"
  local seed="$2"
  local router_mode="$3"
  local shared_spec_dir="$4"
  local model_root="$5"
  local wandb_group="$6"
  local wandb_tags="$7"
  local run_name="$8"
  local core_layers="$9"
  local core_expansion="${10}"
  local run_specs

  require_dir "${shared_spec_dir}"
  read -r -d '' run_specs <<EOF || true
${run_name} ${core_layers} ${core_expansion} 8 1 1 -3.0 0.003 100 3500 0.0003 4096 256 512
EOF

  echo
  echo "[${family}] seed=${seed} router=${router_mode} temporal=${BRANCH_TEMPORAL_MODE} gate=${RESIDUAL_TOKEN_GATE_MODE} model_root=${model_root}"
  env \
    SEED="${seed}" \
    BRANCH_ROUTER_MODE="${router_mode}" \
    SHARED_SPEC_DIR="${shared_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="${wandb_group}" \
    WANDB_TAGS="${wandb_tags}" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

run_blocks1_seed_mode() {
  local seed="$1"
  local router_mode="$2"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks1_router_screen_gate_${RESIDUAL_TOKEN_GATE_MODE}_temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}")_seed${seed}_${RUN_VERSION}"
  run_family_seed_mode \
    "blocks1" \
    "${seed}" \
    "${router_mode}" \
    "${shared_spec_dir}" \
    "${model_root}" \
    "blocks1_router_screen512m_${RUN_VERSION}" \
    "core_amp,5090,final_week,architecture,router,screening,blocks1,gate_${RESIDUAL_TOKEN_GATE_MODE},temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}"),router_$(mode_slug "${router_mode}")" \
    "blocks1_resid10_e12_gate_${RESIDUAL_TOKEN_GATE_MODE}_temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}")_router_$(mode_slug "${router_mode}")_h3500_512m_s${seed}" \
    "10" \
    "12.0"
}

run_blocks0_seed_mode() {
  local seed="$1"
  local router_mode="$2"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1/blocks0_resid12_e10_h7000_1b"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks0_router_replay_gate_${RESIDUAL_TOKEN_GATE_MODE}_temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}")_seed${seed}_${RUN_VERSION}"
  run_family_seed_mode \
    "blocks0" \
    "${seed}" \
    "${router_mode}" \
    "${shared_spec_dir}" \
    "${model_root}" \
    "blocks0_router_replay512m_${RUN_VERSION}" \
    "core_amp,5090,final_week,architecture,router,replay,blocks0,gate_${RESIDUAL_TOKEN_GATE_MODE},temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}"),router_$(mode_slug "${router_mode}")" \
    "blocks0_resid12_e10_gate_${RESIDUAL_TOKEN_GATE_MODE}_temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}")_router_$(mode_slug "${router_mode}")_h3500_512m_s${seed}" \
    "12" \
    "10.0"
}

run_blocks2_seed_mode() {
  local seed="$1"
  local router_mode="$2"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks2_wide_compare1b_seed1337_v1/blocks2_resid12_e8_h7000_1b_s1337"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks2_router_replay_gate_${RESIDUAL_TOKEN_GATE_MODE}_temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}")_seed${seed}_${RUN_VERSION}"
  run_family_seed_mode \
    "blocks2" \
    "${seed}" \
    "${router_mode}" \
    "${shared_spec_dir}" \
    "${model_root}" \
    "blocks2_router_replay512m_${RUN_VERSION}" \
    "core_amp,5090,final_week,architecture,router,replay,blocks2,gate_${RESIDUAL_TOKEN_GATE_MODE},temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}"),router_$(mode_slug "${router_mode}")" \
    "blocks2_resid12_e8_gate_${RESIDUAL_TOKEN_GATE_MODE}_temporal_$(mode_slug "${BRANCH_TEMPORAL_MODE}")_router_$(mode_slug "${router_mode}")_h3500_512m_s${seed}" \
    "12" \
    "8.0"
}

main() {
  local seed router_mode
  print_header

  for seed in ${SEEDS}; do
    for router_mode in $(router_list); do
      if [[ "${RUN_BLOCKS1}" == "1" ]]; then
        run_blocks1_seed_mode "${seed}" "${router_mode}"
      fi
      if [[ "${RUN_BLOCKS0}" == "1" ]]; then
        run_blocks0_seed_mode "${seed}" "${router_mode}"
      fi
      if [[ "${RUN_BLOCKS2}" == "1" ]]; then
        run_blocks2_seed_mode "${seed}" "${router_mode}"
      fi
    done
  done
}

main "$@"
