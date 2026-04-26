#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"
source "${SCRIPT_DIR}/5090_common.sh"

SEEDS="${SEEDS:-1337}"
RUN_VERSION="${RUN_VERSION:-v2}"
RUN_BLOCKS1="${RUN_BLOCKS1:-0}"
RUN_BLOCKS0="${RUN_BLOCKS0:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0035}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PRESET="${PRESET:-controller_default}"
COMPILE="${COMPILE:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
REBUILD_SHARED="${REBUILD_SHARED:-0}"
CORE_AMP_PHASE="${CORE_AMP_PHASE:-5090_trigram_memory_screen}"
TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-131072}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
STORAGE_DTYPE="${STORAGE_DTYPE:-uint16}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
HARD_LOSS_GAMMA="${HARD_LOSS_GAMMA:-0.5}"
HARD_LOSS_CAP="${HARD_LOSS_CAP:-5.0}"
VAL_EVERY="${VAL_EVERY:-256}"
VAL_STEPS="${VAL_STEPS:-8}"
LOG_EVERY="${LOG_EVERY:-64}"
LOG_STATE_EVERY="${LOG_STATE_EVERY:-256}"
SAVE_EVERY="${SAVE_EVERY:-2048}"
TRAIN_FRAC="${TRAIN_FRAC:-0.98}"
FULL_VAL_FINAL="${FULL_VAL_FINAL:-0}"
if [[ -z "${MMAP+x}" ]]; then
  if [[ "${NO_MMAP:-0}" == "1" ]]; then
    MMAP=0
  else
    MMAP=1
  fi
fi
if [[ -z "${AUTOCAST+x}" ]]; then
  if [[ "${NO_AUTOCAST:-0}" == "1" ]]; then
    AUTOCAST=0
  else
    AUTOCAST=1
  fi
fi
TOKENS_ON_DEVICE="${TOKENS_ON_DEVICE:-0}"
BRANCH_TEMPORAL_MODE="${BRANCH_TEMPORAL_MODE:-current}"
RESIDUAL_TOKEN_GATE_MODE="${RESIDUAL_TOKEN_GATE_MODE:-none}"
BRANCH_ROUTER_MODE="${BRANCH_ROUTER_MODE:-none}"
BASE_BIGRAM_DELTA="${BASE_BIGRAM_DELTA:-none}"
TRIGRAM_MEMORY="${TRIGRAM_MEMORY:-frozen}"
TRIGRAM_LOG_SCALE_INIT="${TRIGRAM_LOG_SCALE_INIT:-0.0}"
TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"
TRIGRAM_RESIDUAL_CLIP="${TRIGRAM_RESIDUAL_CLIP:-8.0}"
TRIGRAM_CONFIDENCE_COUNT_CAP="${TRIGRAM_CONFIDENCE_COUNT_CAP:-4096}"
TRIGRAM_CHUNK_SIZE="${TRIGRAM_CHUNK_SIZE:-50000000}"
TRIGRAM_COUNT_WORKERS="${TRIGRAM_COUNT_WORKERS:-1}"
TRIGRAM_MEMORY_SPEC_CACHE_ROOT="${TRIGRAM_MEMORY_SPEC_CACHE_ROOT:-${HOME}/.cache/experiments/param-golf-coreamp}"
TRIGRAM_MEMORY_TABLE_CACHE_ROOT="${TRIGRAM_MEMORY_TABLE_CACHE_ROOT:-${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}/trigram_memory_tables}"
REBUILD_TRIGRAM_MEMORY_TABLE_CACHE="${REBUILD_TRIGRAM_MEMORY_TABLE_CACHE:-0}"
RESIDUAL_READOUT_DELTA_RANK="${RESIDUAL_READOUT_DELTA_RANK:-0}"
RESIDUAL_READOUT_DELTA_INIT_STD="${RESIDUAL_READOUT_DELTA_INIT_STD:-0.02}"
SCAN_BACKEND="${SCAN_BACKEND:-auto}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pg-hconv-ablations}"
WANDB_WATCH="${WANDB_WATCH:-gradients}"
WANDB_WATCH_LOG_FREQ="${WANDB_WATCH_LOG_FREQ:-25}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --run-version VALUE
  --seeds VALUE
  --run-blocks1 | --no-run-blocks1
  --run-blocks0 | --no-run-blocks0
  --learning-rate VALUE
  --trigram-top-k VALUE
  --count-workers VALUE
  --full-val-final | --no-full-val-final
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-version) RUN_VERSION="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --run-blocks1) RUN_BLOCKS1=1; shift ;;
    --no-run-blocks1) RUN_BLOCKS1=0; shift ;;
    --run-blocks0) RUN_BLOCKS0=1; shift ;;
    --no-run-blocks0) RUN_BLOCKS0=0; shift ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --trigram-top-k) TRIGRAM_TOP_K="$2"; shift 2 ;;
    --trigram-count-workers|--count-workers) TRIGRAM_COUNT_WORKERS="$2"; shift 2 ;;
    --full-val-final) FULL_VAL_FINAL=1; shift ;;
    --no-full-val-final) FULL_VAL_FINAL=0; shift ;;
    --val-every) VAL_EVERY="$2"; shift 2 ;;
    --val-steps) VAL_STEPS="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --log-state-every) LOG_STATE_EVERY="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
LEARNING_RATE_TAG="$(pg_5090_lr_slug "${LEARNING_RATE}")"

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
    --data "${DATA_PATH}"
    --family "${family}"
    --cache-root "${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}"
    --storage-dtype "${STORAGE_DTYPE}"
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

  echo "Ensuring dense trigram top-${TRIGRAM_TOP_K} memory spec from full training shards ..."
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/build_trigram_memory_spec.py"
    "${source_spec_dir}"
    "${out_spec_dir}"
    --data "${DATA_PATH}"
    --storage-dtype "${STORAGE_DTYPE}"
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
  echo "lr_schedule=${LR_SCHEDULE} weight_decay=${WEIGHT_DECAY} grad_clip=${GRAD_CLIP} hard_loss_gamma=${HARD_LOSS_GAMMA} hard_loss_cap=${HARD_LOSS_CAP}"
  echo "val_every=${VAL_EVERY} val_steps=${VAL_STEPS} log_every=${LOG_EVERY} log_state_every=${LOG_STATE_EVERY} save_every=${SAVE_EVERY} full_val_final=${FULL_VAL_FINAL}"
  echo "train_frac=${TRAIN_FRAC} mmap=${MMAP} autocast=${AUTOCAST} tokens_on_device=${TOKENS_ON_DEVICE}"
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
  local sweep_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
    --preset "${PRESET}"
    --seed "${seed}"
    --data-path "${DATA_PATH}"
    --storage-dtype "${STORAGE_DTYPE}"
    --target-effective-step-tokens "${TARGET_EFFECTIVE_STEP_TOKENS}"
    --shared-spec-dir "${memory_spec_dir}"
    --model-root "${model_root}"
    --lr-schedule "${LR_SCHEDULE}"
    --weight-decay "${WEIGHT_DECAY}"
    --grad-clip "${GRAD_CLIP}"
    --hard-loss-gamma "${HARD_LOSS_GAMMA}"
    --hard-loss-cap "${HARD_LOSS_CAP}"
    --val-every "${VAL_EVERY}"
    --val-steps "${VAL_STEPS}"
    --log-every "${LOG_EVERY}"
    --log-state-every "${LOG_STATE_EVERY}"
    --save-every "${SAVE_EVERY}"
    --train-frac "${TRAIN_FRAC}"
    --branch-temporal-mode "${BRANCH_TEMPORAL_MODE}"
    --residual-token-gate-mode "${RESIDUAL_TOKEN_GATE_MODE}"
    --branch-router-mode "${BRANCH_ROUTER_MODE}"
    --base-bigram-delta "${BASE_BIGRAM_DELTA}"
    --trigram-memory "${TRIGRAM_MEMORY}"
    --trigram-log-scale-init "${TRIGRAM_LOG_SCALE_INIT}"
    --residual-readout-delta-rank "${RESIDUAL_READOUT_DELTA_RANK}"
    --residual-readout-delta-init-std "${RESIDUAL_READOUT_DELTA_INIT_STD}"
    --scan-backend "${SCAN_BACKEND}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-watch "${WANDB_WATCH}"
    --wandb-watch-log-freq "${WANDB_WATCH_LOG_FREQ}"
    --wandb-group "${wandb_group}"
    --wandb-tags "${wandb_tags}"
    --core-amp-phase "${CORE_AMP_PHASE}"
    --run-spec "${run_specs}"
  )
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "compile" "${COMPILE}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "gradient-checkpointing" "${GRADIENT_CHECKPOINTING}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "full-val-final" "${FULL_VAL_FINAL}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "mmap" "${MMAP}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "autocast" "${AUTOCAST}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "tokens-on-device" "${TOKENS_ON_DEVICE}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "skip-done" "${SKIP_DONE}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "rebuild-shared" "${REBUILD_SHARED}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "wandb" "${WANDB}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    sweep_cmd+=(--dry-run)
  fi
  "${sweep_cmd[@]}"
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
