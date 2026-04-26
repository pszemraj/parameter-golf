#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"
source "${SCRIPT_DIR}/5090_common.sh"

SEEDS="${SEEDS:-1337}"
RUN_VERSION="${RUN_VERSION:-v1}"
RUN_BLOCKS1="${RUN_BLOCKS1:-0}"
RUN_BLOCKS0="${RUN_BLOCKS0:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0035}"
LEARNING_RATE_TAG="$(pg_5090_lr_slug "${LEARNING_RATE}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PRESET="${PRESET:-controller_default}"
export COMPILE="${COMPILE:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export REBUILD_SHARED="${REBUILD_SHARED:-0}"
export CORE_AMP_PHASE="${CORE_AMP_PHASE:-5090_trigram_memory_screen}"
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
export BASE_BIGRAM_DELTA="${BASE_BIGRAM_DELTA:-none}"
export TRIGRAM_MEMORY="${TRIGRAM_MEMORY:-frozen}"
export TRIGRAM_LOG_SCALE_INIT="${TRIGRAM_LOG_SCALE_INIT:-0.0}"
export TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"
export TRIGRAM_RESIDUAL_CLIP="${TRIGRAM_RESIDUAL_CLIP:-8.0}"
export TRIGRAM_CONFIDENCE_COUNT_CAP="${TRIGRAM_CONFIDENCE_COUNT_CAP:-4096}"
export TRIGRAM_CHUNK_SIZE="${TRIGRAM_CHUNK_SIZE:-50000000}"
export TRIGRAM_COUNT_WORKERS="${TRIGRAM_COUNT_WORKERS:-1}"
export TRIGRAM_MEMORY_SPEC_CACHE_ROOT="${TRIGRAM_MEMORY_SPEC_CACHE_ROOT:-${HOME}/.cache/experiments/param-golf-coreamp}"
export TRIGRAM_MEMORY_TABLE_CACHE_ROOT="${TRIGRAM_MEMORY_TABLE_CACHE_ROOT:-${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}/trigram_memory_tables}"
export REBUILD_TRIGRAM_MEMORY_TABLE_CACHE="${REBUILD_TRIGRAM_MEMORY_TABLE_CACHE:-0}"
export RESIDUAL_READOUT_DELTA_RANK="${RESIDUAL_READOUT_DELTA_RANK:-0}"
export RESIDUAL_READOUT_DELTA_INIT_STD="${RESIDUAL_READOUT_DELTA_INIT_STD:-0.02}"
export SCAN_BACKEND="${SCAN_BACKEND:-auto}"
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pg-hconv-ablations}"

pg_5090_require_serious_launcher_defaults "$(basename "$0")"

if [[ "${ALLOW_DEGRADED_5090_SERIOUS:-0}" != "1" && -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
  pg_5090_fail "$(basename "$0")" "TRIGRAM_MAX_TOKENS must be unset for serious runs"
fi

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
  if [[ ! -f "$path/spec.pt" ]]; then
    echo "Missing required spec: $path/spec.pt" >&2
    exit 1
  fi
  if [[ ! -f "$path/config.json" ]]; then
    echo "Missing required config: $path/config.json" >&2
    exit 1
  fi
}

resolve_trigram_memory_spec_dir() {
  local source_spec_dir="$1"
  local family="$2"
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/trigram_memory_spec_path.py"
    "${source_spec_dir}"
    --data "${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
    --family "${family}"
    --cache-root "${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}"
    --storage-dtype "${STORAGE_DTYPE:-uint16}"
    --top-k "${TRIGRAM_TOP_K}"
    --smoothing "${SMOOTHING:-0.25}"
    --residual-clip "${TRIGRAM_RESIDUAL_CLIP}"
    --confidence-count-cap "${TRIGRAM_CONFIDENCE_COUNT_CAP}"
  )
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cmd+=(--mkdir)
  fi
  if [[ -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
    cmd+=(--max-tokens "${TRIGRAM_MAX_TOKENS}")
  fi
  "${cmd[@]}"
}

ensure_trigram_memory_spec() {
  local source_spec_dir="$1"
  local out_spec_dir="$2"

  require_dir "${source_spec_dir}"
  if [[ "${REBUILD_TRIGRAM_MEMORY:-0}" != "1" && -f "${out_spec_dir}/spec.pt" && -f "${out_spec_dir}/config.json" ]]; then
    echo "Using cached trigram memory spec: ${out_spec_dir}"
    return 0
  fi

  echo "Building dense trigram top-${TRIGRAM_TOP_K} memory spec from full training shards ..."
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/build_trigram_memory_spec.py"
    "${source_spec_dir}"
    "${out_spec_dir}"
    --data "${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
    --storage-dtype "${STORAGE_DTYPE:-uint16}"
    --top-k "${TRIGRAM_TOP_K}"
    --smoothing "${SMOOTHING:-0.25}"
    --residual-clip "${TRIGRAM_RESIDUAL_CLIP}"
    --confidence-count-cap "${TRIGRAM_CONFIDENCE_COUNT_CAP}"
    --chunk-size "${TRIGRAM_CHUNK_SIZE}"
    --count-workers "${TRIGRAM_COUNT_WORKERS}"
    --table-cache-root "${TRIGRAM_MEMORY_TABLE_CACHE_ROOT}"
  )
  if [[ "${REBUILD_TRIGRAM_MEMORY:-0}" == "1" ]]; then
    cmd+=(--force)
  fi
  if [[ "${REBUILD_TRIGRAM_MEMORY_TABLE_CACHE}" == "1" ]]; then
    cmd+=(--rebuild-table-cache)
  fi
  if [[ -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
    cmd+=(--max-tokens "${TRIGRAM_MAX_TOKENS}")
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

print_header() {
  echo "5090 trigram memory screen"
  echo "repo_root=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "seeds=${SEEDS}"
  echo "run_version=${RUN_VERSION}"
  echo "trigram_memory=${TRIGRAM_MEMORY} top_k=${TRIGRAM_TOP_K} log_scale_init=${TRIGRAM_LOG_SCALE_INIT} count_workers=${TRIGRAM_COUNT_WORKERS}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "run_blocks1=${RUN_BLOCKS1} run_blocks0=${RUN_BLOCKS0}"
  echo "compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}"
  echo "target_effective_step_tokens=${TARGET_EFFECTIVE_STEP_TOKENS}"
  echo "trigram_memory_spec_cache_root=${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}"
  echo "trigram_memory_table_cache_root=${TRIGRAM_MEMORY_TABLE_CACHE_ROOT}"
  echo "branch_temporal_mode=${BRANCH_TEMPORAL_MODE} residual_token_gate_mode=${RESIDUAL_TOKEN_GATE_MODE} branch_router_mode=${BRANCH_ROUTER_MODE}"
  echo "scan_backend=${SCAN_BACKEND} wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT} py_unbuffered=${PYTHONUNBUFFERED}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

run_family_seed() {
  local family="$1"
  local seed="$2"
  local shared_spec_dir="$3"
  local memory_spec_dir="$4"
  local model_root="$5"
  local wandb_group="$6"
  local wandb_tags="$7"
  local run_name="$8"
  local core_layers="$9"
  local core_expansion="${10}"

  ensure_trigram_memory_spec "${shared_spec_dir}" "${memory_spec_dir}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    require_dir "${memory_spec_dir}"
  fi

  local run_specs
  read -r -d '' run_specs <<EOF || true
${run_name} ${core_layers} ${core_expansion} 8 1 1 -3.0 ${LEARNING_RATE} 100 3500 0.0003 4096 256 512
EOF

  echo
  echo "[${family}] seed=${seed} trigram_top_k=${TRIGRAM_TOP_K} lr=${LEARNING_RATE} model_root=${model_root}"
  env \
    SEED="${seed}" \
    SHARED_SPEC_DIR="${memory_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    WANDB_GROUP="${wandb_group}" \
    WANDB_TAGS="${wandb_tags}" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

run_blocks1_seed() {
  local seed="$1"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b"
  local memory_spec_dir
  memory_spec_dir="$(resolve_trigram_memory_spec_dir "${shared_spec_dir}" "blocks1")"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks1_trigram_seed${seed}_${RUN_VERSION}"
  run_family_seed \
    "blocks1" \
    "${seed}" \
    "${shared_spec_dir}" \
    "${memory_spec_dir}" \
    "${model_root}" \
    "blocks1_trigram512m_${RUN_VERSION}" \
    "core_amp,5090,architecture,trigram_memory,screening,blocks1,${LEARNING_RATE_TAG}" \
    "blocks1_resid10_e12_trigramk${TRIGRAM_TOP_K}_${LEARNING_RATE_TAG}_h3500_512m_s${seed}" \
    "10" \
    "12.0"
}

run_blocks0_seed() {
  local seed="$1"
  local shared_spec_dir="${REPO_ROOT}/experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1/blocks0_resid12_e10_h7000_1b"
  local memory_spec_dir
  memory_spec_dir="$(resolve_trigram_memory_spec_dir "${shared_spec_dir}" "blocks0")"
  local model_root="${REPO_ROOT}/experiments/5090_architecture/blocks0_trigram_seed${seed}_${RUN_VERSION}"
  run_family_seed \
    "blocks0" \
    "${seed}" \
    "${shared_spec_dir}" \
    "${memory_spec_dir}" \
    "${model_root}" \
    "blocks0_trigram512m_${RUN_VERSION}" \
    "core_amp,5090,architecture,trigram_memory,screening,blocks0,${LEARNING_RATE_TAG}" \
    "blocks0_resid12_e10_trigramk${TRIGRAM_TOP_K}_${LEARNING_RATE_TAG}_h3500_512m_s${seed}" \
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
