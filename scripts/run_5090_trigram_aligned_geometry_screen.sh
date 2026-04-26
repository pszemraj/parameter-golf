#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/5090_common.sh"

PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export WANDB="${WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-pg-hconv-ablations}"
export WANDB_WATCH="${WANDB_WATCH:-gradients}"
export WANDB_WATCH_LOG_FREQ="${WANDB_WATCH_LOG_FREQ:-25}"
export PRESET="${PRESET:-controller_default}"
export COMPILE="${COMPILE:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export SCAN_BACKEND="${SCAN_BACKEND:-auto}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
export REBUILD_SHARED="${REBUILD_SHARED:-0}"

export RUN_VERSION="${RUN_VERSION:-v1}"
export SEEDS="${SEEDS:-1337}"
export SKIP_DONE="${SKIP_DONE:-1}"
export RUN_BLOCKS0="${RUN_BLOCKS0:-1}"

export LEARNING_RATE="${LEARNING_RATE:-0.0035}"
LEARNING_RATE_TAG="$(pg_5090_lr_slug "${LEARNING_RATE}")"

export COREAMP_SPEC_CACHE_ROOT="${COREAMP_SPEC_CACHE_ROOT:-${HOME}/.cache/experiments/param-golf-coreamp}"
export GEOMETRY_CORE_DIM="${GEOMETRY_CORE_DIM:-128}"
export GEOMETRY_CORE_LAYERS="${GEOMETRY_CORE_LAYERS:-4}"
export GEOMETRY_CORE_INNER_DIM="${GEOMETRY_CORE_INNER_DIM:-512}"
if [[ -z "${GEOMETRY_CORE_EXPANSION:-}" ]]; then
  GEOMETRY_CORE_EXPANSION="$(
    "${PYTHON_BIN}" - "${GEOMETRY_CORE_DIM}" "${GEOMETRY_CORE_INNER_DIM}" <<'PY'
import sys

core_dim = int(sys.argv[1])
inner_dim = int(sys.argv[2])
print(repr(float(inner_dim) / float(core_dim)))
PY
  )"
fi
export GEOMETRY_CORE_EXPANSION
GEOMETRY_CORE_INNER_DIM_RESOLVED="$(
  "${PYTHON_BIN}" - "${GEOMETRY_CORE_DIM}" "${GEOMETRY_CORE_EXPANSION}" <<'PY'
import sys

core_dim = int(sys.argv[1])
expansion = float(sys.argv[2])
print(int(core_dim * expansion))
PY
)"
export GEOMETRY_CORE_INNER_DIM_RESOLVED
if [[ -n "${GEOMETRY_CORE_INNER_DIM:-}" && "${GEOMETRY_CORE_INNER_DIM_RESOLVED}" != "${GEOMETRY_CORE_INNER_DIM}" ]]; then
  pg_5090_fail "$(basename "$0")" "GEOMETRY_CORE_DIM=${GEOMETRY_CORE_DIM} and GEOMETRY_CORE_EXPANSION=${GEOMETRY_CORE_EXPANSION} resolve to inner=${GEOMETRY_CORE_INNER_DIM_RESOLVED}, expected ${GEOMETRY_CORE_INNER_DIM}"
fi
export GEOMETRY_NUM_BLOCKS="${GEOMETRY_NUM_BLOCKS:-0}"
export GEOMETRY_BRANCH_LAGS="${GEOMETRY_BRANCH_LAGS:-1,2,3,4,6,8,12,16,24,32,48,64}"
export GEOMETRY_LABEL="${GEOMETRY_LABEL:-blocks0_core${GEOMETRY_CORE_DIM}_l${GEOMETRY_CORE_LAYERS}_i${GEOMETRY_CORE_INNER_DIM_RESOLVED}}"
export GEOMETRY_CARRY_CHUNKS="${GEOMETRY_CARRY_CHUNKS:-8}"
export GEOMETRY_BPTT_CHUNKS="${GEOMETRY_BPTT_CHUNKS:-1}"
export GEOMETRY_NUM_STEPS="${GEOMETRY_NUM_STEPS:-4096}"
export GEOMETRY_BATCH_SIZE="${GEOMETRY_BATCH_SIZE:-256}"
export GEOMETRY_SEQ_LEN="${GEOMETRY_SEQ_LEN:-512}"
export GEOMETRY_LR_WARMUP_STEPS="${GEOMETRY_LR_WARMUP_STEPS:-100}"
export GEOMETRY_LR_HOLD_STEPS="${GEOMETRY_LR_HOLD_STEPS:-3500}"
export GEOMETRY_MIN_LR="${GEOMETRY_MIN_LR:-0.0003}"
if [[ -z "${GEOMETRY_TRAIN_LABEL:-}" ]]; then
  if [[ "${GEOMETRY_NUM_STEPS}" == "4096" && "${TARGET_EFFECTIVE_STEP_TOKENS:-131072}" == "131072" ]]; then
    GEOMETRY_TRAIN_LABEL="512m"
  elif [[ "${GEOMETRY_NUM_STEPS}" == "8192" && "${TARGET_EFFECTIVE_STEP_TOKENS:-131072}" == "131072" ]]; then
    GEOMETRY_TRAIN_LABEL="1b"
  else
    GEOMETRY_TRAIN_LABEL="${GEOMETRY_NUM_STEPS}steps"
  fi
fi
export GEOMETRY_TRAIN_LABEL

export TRIGRAM_MEMORY="${TRIGRAM_MEMORY:-frozen}"
export TRIGRAM_LOG_SCALE_INIT="${TRIGRAM_LOG_SCALE_INIT:-0.0}"
export TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"
export TRIGRAM_RESIDUAL_CLIP="${TRIGRAM_RESIDUAL_CLIP:-8.0}"
export TRIGRAM_CONFIDENCE_COUNT_CAP="${TRIGRAM_CONFIDENCE_COUNT_CAP:-4096}"
export TRIGRAM_CHUNK_SIZE="${TRIGRAM_CHUNK_SIZE:-50000000}"
export TRIGRAM_MEMORY_SPEC_CACHE_ROOT="${TRIGRAM_MEMORY_SPEC_CACHE_ROOT:-${COREAMP_SPEC_CACHE_ROOT}}"
export TRIGRAM_MEMORY_TABLE_CACHE_ROOT="${TRIGRAM_MEMORY_TABLE_CACHE_ROOT:-${COREAMP_SPEC_CACHE_ROOT}/trigram_memory_tables}"
export REBUILD_TRIGRAM_MEMORY_TABLE_CACHE="${REBUILD_TRIGRAM_MEMORY_TABLE_CACHE:-0}"

export TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-131072}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export STORAGE_DTYPE="${STORAGE_DTYPE:-uint16}"
export LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
export VAL_EVERY="${VAL_EVERY:-256}"
export VAL_STEPS="${VAL_STEPS:-8}"
export LOG_EVERY="${LOG_EVERY:-64}"
export LOG_STATE_EVERY="${LOG_STATE_EVERY:-256}"
export SAVE_EVERY="${SAVE_EVERY:-2048}"
export TRAIN_FRAC="${TRAIN_FRAC:-0.98}"
export FULL_VAL_FINAL="${FULL_VAL_FINAL:-0}"
export BASE_BIGRAM_DELTA="${BASE_BIGRAM_DELTA:-none}"
export RESIDUAL_READOUT_DELTA_RANK="${RESIDUAL_READOUT_DELTA_RANK:-0}"
export RESIDUAL_READOUT_DELTA_INIT_STD="${RESIDUAL_READOUT_DELTA_INIT_STD:-0.02}"

pg_5090_require_serious_launcher_defaults "$(basename "$0")"
if [[ "${ALLOW_DEGRADED_5090_SERIOUS:-0}" != "1" && -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
  pg_5090_fail "$(basename "$0")" "TRIGRAM_MAX_TOKENS must be unset for serious runs"
fi

slugify() {
  local raw="$1"
  raw="${raw//./p}"
  raw="${raw//,/m}"
  raw="${raw// /_}"
  printf '%s' "${raw}"
}

shared_spec_dir() {
  local label
  label="$(slugify "${GEOMETRY_LABEL}_branches_${GEOMETRY_BRANCH_LAGS}_blocks${GEOMETRY_NUM_BLOCKS}")"
  printf '%s/shared_specs/%s_full' "${COREAMP_SPEC_CACHE_ROOT}" "${label}"
}

ensure_shared_spec() {
  local out_dir="$1"
  if [[ "${REBUILD_GEOMETRY_SPEC:-0}" != "1" && -f "${out_dir}/spec.pt" && -f "${out_dir}/config.json" ]]; then
    echo "Using cached aligned shared spec: ${out_dir}"
    return 0
  fi

  echo "Building aligned shared spec from full training shards: ${out_dir}"
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/inspect_model.py" init "${out_dir}"
    --data "${DATA_PATH}"
    --storage-dtype "${STORAGE_DTYPE}"
    --vocab-size 1024
    --core-dim "${GEOMETRY_CORE_DIM}"
    --branch-lags "${GEOMETRY_BRANCH_LAGS}"
    --num-blocks "${GEOMETRY_NUM_BLOCKS}"
    --fixed-dtype bfloat16
    --embedding-init spectral
    --spectral-neighbors 64
    --lag-identity-base 0.15
    --spec-strategy auto
    --spec-workers -1
    --core-layers "${GEOMETRY_CORE_LAYERS}"
    --core-expansion "${GEOMETRY_CORE_EXPANSION}"
    --residual-core 1
    --residual-core-init -3.0
    --branch-temporal-mode current
    --residual-token-gate-mode none
    --branch-router-mode none
    --base-bigram-delta none
    --trigram-memory none
    --scan-backend auto
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

resolve_trigram_memory_spec_dir() {
  local source_spec_dir="$1"
  local family="$2"
  if [[ "${DRY_RUN:-0}" == "1" && ! -f "${source_spec_dir}/spec.pt" ]]; then
    printf '%s/trigram_memory_specs/%s_k%s_dryrun' \
      "${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}" \
      "$(slugify "${family}")" \
      "${TRIGRAM_TOP_K}"
    return 0
  fi
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/trigram_memory_spec_path.py"
    "${source_spec_dir}"
    --data "${DATA_PATH}"
    --family "${family}"
    --cache-root "${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}"
    --storage-dtype "${STORAGE_DTYPE}"
    --top-k "${TRIGRAM_TOP_K}"
    --smoothing 0.25
    --residual-clip "${TRIGRAM_RESIDUAL_CLIP}"
    --confidence-count-cap "${TRIGRAM_CONFIDENCE_COUNT_CAP}"
    --mkdir
  )
  if [[ -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
    cmd+=(--max-tokens "${TRIGRAM_MAX_TOKENS}")
  fi
  "${cmd[@]}"
}

ensure_trigram_memory_spec() {
  local source_spec_dir="$1"
  local out_spec_dir="$2"
  if [[ "${REBUILD_TRIGRAM_MEMORY:-0}" != "1" && -f "${out_spec_dir}/spec.pt" && -f "${out_spec_dir}/config.json" ]]; then
    echo "Using cached trigram memory spec: ${out_spec_dir}"
    return 0
  fi

  echo "Building dense trigram top-${TRIGRAM_TOP_K} memory spec from full training shards ..."
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/build_trigram_memory_spec.py"
    "${source_spec_dir}"
    "${out_spec_dir}"
    --data "${DATA_PATH}"
    --storage-dtype "${STORAGE_DTYPE}"
    --top-k "${TRIGRAM_TOP_K}"
    --smoothing 0.25
    --residual-clip "${TRIGRAM_RESIDUAL_CLIP}"
    --confidence-count-cap "${TRIGRAM_CONFIDENCE_COUNT_CAP}"
    --chunk-size "${TRIGRAM_CHUNK_SIZE}"
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
    return 0
  fi
  "${cmd[@]}"
}

print_header() {
  cat <<EOF
5090 trigram aligned-geometry screen
repo_root=${REPO_ROOT}
python=${PYTHON_BIN}
seeds=${SEEDS}
run_version=${RUN_VERSION}
geometry_core_dim=${GEOMETRY_CORE_DIM} layers=${GEOMETRY_CORE_LAYERS} inner_dim=${GEOMETRY_CORE_INNER_DIM_RESOLVED} expansion=${GEOMETRY_CORE_EXPANSION} blocks=${GEOMETRY_NUM_BLOCKS}
geometry_branch_lags=${GEOMETRY_BRANCH_LAGS}
trigram_memory=${TRIGRAM_MEMORY} top_k=${TRIGRAM_TOP_K} log_scale_init=${TRIGRAM_LOG_SCALE_INIT}
learning_rate=${LEARNING_RATE}
compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}
num_steps=${GEOMETRY_NUM_STEPS} lr_hold_steps=${GEOMETRY_LR_HOLD_STEPS} carry_chunks=${GEOMETRY_CARRY_CHUNKS} bptt_chunks=${GEOMETRY_BPTT_CHUNKS}
batch_size=${GEOMETRY_BATCH_SIZE} seq_len=${GEOMETRY_SEQ_LEN} target_effective_step_tokens=${TARGET_EFFECTIVE_STEP_TOKENS}
coreamp_spec_cache_root=${COREAMP_SPEC_CACHE_ROOT}
trigram_memory_spec_cache_root=${TRIGRAM_MEMORY_SPEC_CACHE_ROOT}
trigram_memory_table_cache_root=${TRIGRAM_MEMORY_TABLE_CACHE_ROOT}
scan_backend=${SCAN_BACKEND} wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT} py_unbuffered=${PYTHONUNBUFFERED}
EOF
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

run_blocks0_seed() {
  local seed="$1"
  local source_spec_dir
  source_spec_dir="$(shared_spec_dir)"
  ensure_shared_spec "${source_spec_dir}"
  local memory_spec_dir
  memory_spec_dir="$(resolve_trigram_memory_spec_dir "${source_spec_dir}" "${GEOMETRY_LABEL}")"
  ensure_trigram_memory_spec "${source_spec_dir}" "${memory_spec_dir}"

  local model_root="${REPO_ROOT}/experiments/5090_architecture/${GEOMETRY_LABEL}_trigram_seed${seed}_${RUN_VERSION}"
  local run_name="${GEOMETRY_LABEL}_trigramk${TRIGRAM_TOP_K}_${LEARNING_RATE_TAG}_h${GEOMETRY_LR_HOLD_STEPS}_${GEOMETRY_TRAIN_LABEL}_s${seed}"
  local run_specs
  read -r -d '' run_specs <<EOF || true
${run_name} ${GEOMETRY_CORE_LAYERS} ${GEOMETRY_CORE_EXPANSION} ${GEOMETRY_CARRY_CHUNKS} ${GEOMETRY_BPTT_CHUNKS} 1 -3.0 ${LEARNING_RATE} ${GEOMETRY_LR_WARMUP_STEPS} ${GEOMETRY_LR_HOLD_STEPS} ${GEOMETRY_MIN_LR} ${GEOMETRY_NUM_STEPS} ${GEOMETRY_BATCH_SIZE} ${GEOMETRY_SEQ_LEN}
EOF

  echo
  echo "[blocks0_aligned] seed=${seed} trigram_top_k=${TRIGRAM_TOP_K} lr=${LEARNING_RATE} inner_dim=${GEOMETRY_CORE_INNER_DIM_RESOLVED} model_root=${model_root}"
  env \
    SEED="${seed}" \
    SHARED_SPEC_DIR="${memory_spec_dir}" \
    MODEL_ROOT="${model_root}" \
    CORE_DIM="${GEOMETRY_CORE_DIM}" \
    CORE_LAYERS="${GEOMETRY_CORE_LAYERS}" \
    CORE_EXPANSION="${GEOMETRY_CORE_EXPANSION}" \
    RESIDUAL_CORE="1" \
    RESIDUAL_CORE_INIT="-3.0" \
    NUM_BLOCKS="${GEOMETRY_NUM_BLOCKS}" \
    BRANCH_LAGS="${GEOMETRY_BRANCH_LAGS}" \
    WANDB_GROUP="${GEOMETRY_LABEL}_trigram512m_${RUN_VERSION}" \
    WANDB_TAGS="core_amp,5090,architecture,trigram_memory,aligned_geometry,screening,${LEARNING_RATE_TAG}" \
    CORE_AMP_PHASE="5090_trigram_aligned_geometry_screen" \
    RUN_SPECS="${run_specs}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
}

main() {
  local seed
  print_header
  for seed in ${SEEDS}; do
    if [[ "${RUN_BLOCKS0}" == "1" ]]; then
      run_blocks0_seed "${seed}"
    fi
  done
}

main "$@"
