#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP="${WANDB_GROUP:-allama-sharedblock-nearcap-v4}"
export WANDB_TAGS="${WANDB_TAGS:-5090,allama,nearcap,finalsize,behavior,blocked,aligned,sharedblocks,clipped_exporter}"
export WANDB_WATCH="${WANDB_WATCH:-all}"
export WANDB_WATCH_LOG_FREQ="${WANDB_WATCH_LOG_FREQ:-100}"
SDPA_BACKEND_VALUE="${SDPA_BACKEND:-auto}"
export SDPA_BACKEND="${SDPA_BACKEND_VALUE}"

SWEEP_PROFILE="${SWEEP_PROFILE:-ablate}"

case "${SWEEP_PROFILE}" in
  ablate)
    DEFAULT_TRAIN_BATCH_TOKENS=262144
    DEFAULT_GRAD_ACCUM_STEPS=64
    DEFAULT_MAX_STEPS=750
    DEFAULT_VAL_LOSS_EVERY=100
    DEFAULT_MAX_WALLCLOCK_SECONDS=0
    DEFAULT_RUN_BASELINE_BLOCK=1
    DEFAULT_RUN_SHARE_BLOCK=1
    DEFAULT_RUN_NORM_BLOCK=1
    DEFAULT_RUN_SHORTCUT_BLOCK=1
    ;;
  screen)
    DEFAULT_TRAIN_BATCH_TOKENS=262144
    DEFAULT_GRAD_ACCUM_STEPS=128
    DEFAULT_MAX_STEPS=750
    DEFAULT_VAL_LOSS_EVERY=250
    DEFAULT_MAX_WALLCLOCK_SECONDS=0
    DEFAULT_RUN_BASELINE_BLOCK=1
    DEFAULT_RUN_SHARE_BLOCK=0
    DEFAULT_RUN_NORM_BLOCK=0
    DEFAULT_RUN_SHORTCUT_BLOCK=0
    ;;
  explore)
    DEFAULT_TRAIN_BATCH_TOKENS=262144
    DEFAULT_GRAD_ACCUM_STEPS=128
    DEFAULT_MAX_STEPS=750
    DEFAULT_VAL_LOSS_EVERY=250
    DEFAULT_MAX_WALLCLOCK_SECONDS=0
    DEFAULT_RUN_BASELINE_BLOCK=1
    DEFAULT_RUN_SHARE_BLOCK=1
    DEFAULT_RUN_NORM_BLOCK=1
    DEFAULT_RUN_SHORTCUT_BLOCK=1
    ;;
  full)
    DEFAULT_TRAIN_BATCH_TOKENS=262144
    DEFAULT_GRAD_ACCUM_STEPS=128
    DEFAULT_MAX_STEPS=2000
    DEFAULT_VAL_LOSS_EVERY=500
    DEFAULT_MAX_WALLCLOCK_SECONDS=0
    DEFAULT_RUN_BASELINE_BLOCK=1
    DEFAULT_RUN_SHARE_BLOCK=1
    DEFAULT_RUN_NORM_BLOCK=1
    DEFAULT_RUN_SHORTCUT_BLOCK=1
    ;;
  *)
    echo "unknown SWEEP_PROFILE=${SWEEP_PROFILE}" >&2
    echo "expected SWEEP_PROFILE=ablate, screen, explore, or full" >&2
    exit 1
    ;;
esac

TRAIN_SEQ_LEN_VALUE="${TRAIN_SEQ_LEN:-1024}"
EVAL_SEQ_LEN_VALUE="${EVAL_SEQ_LEN:-1024}"
TRAIN_BATCH_TOKENS_VALUE="${TRAIN_BATCH_TOKENS:-${DEFAULT_TRAIN_BATCH_TOKENS}}"
GRAD_ACCUM_STEPS_VALUE="${GRAD_ACCUM_STEPS:-${DEFAULT_GRAD_ACCUM_STEPS}}"
MAX_STEPS_VALUE="${MAX_STEPS:-${ITERATIONS:-${DEFAULT_MAX_STEPS}}}"
VAL_LOSS_EVERY_VALUE="${VAL_LOSS_EVERY:-${DEFAULT_VAL_LOSS_EVERY}}"
TRAIN_LOG_EVERY_VALUE="${TRAIN_LOG_EVERY:-25}"
EVAL_MODE_VALUE="${EVAL_MODE:-sampled}"
VAL_BATCH_SIZE_VALUE="${VAL_BATCH_SIZE:-8}"
VAL_BATCHES_VALUE="${VAL_BATCHES:-8}"
EVAL_BATCH_TOKENS_VALUE="${EVAL_BATCH_TOKENS:-0}"
MLP_MULTIPLE_OF_VALUE="${MLP_MULTIPLE_OF:-128}"
OUT_DIR_VALUE="${OUT_DIR:-./runs_allama}"
DATA_PATH_VALUE="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH_VALUE="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE_VALUE="${VOCAB_SIZE:-1024}"
DEVICE_VALUE="${DEVICE:-cuda}"
DTYPE_VALUE="${DTYPE:-bf16}"
NUM_EPOCHS_VALUE="${NUM_EPOCHS:-1}"
MAX_WALLCLOCK_SECONDS_VALUE="${MAX_WALLCLOCK_SECONDS:-${DEFAULT_MAX_WALLCLOCK_SECONDS}}"
COMPILE_WARMUP_STEPS_VALUE="${COMPILE_WARMUP_STEPS:-20}"
CONTROL_TENSOR_NAME_PATTERNS_VALUE="${CONTROL_TENSOR_NAME_PATTERNS:-attn_scale,mlp_scale,q_gain,x0_gate,norm}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-sbcal_v4_}"

BASE_ENV=(
  OUT_DIR="${OUT_DIR_VALUE}"
  DATA_PATH="${DATA_PATH_VALUE}"
  TOKENIZER_PATH="${TOKENIZER_PATH_VALUE}"
  DEVICE="${DEVICE_VALUE}"
  DTYPE="${DTYPE_VALUE}"
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN_VALUE}"
  EVAL_SEQ_LEN="${EVAL_SEQ_LEN_VALUE}"
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS_VALUE}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS_VALUE}"
  MAX_STEPS="${MAX_STEPS_VALUE}"
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY_VALUE}"
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY_VALUE}"
  EVAL_MODE="${EVAL_MODE_VALUE}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE_VALUE}"
  VAL_BATCHES="${VAL_BATCHES_VALUE}"
  EVAL_BATCH_TOKENS="${EVAL_BATCH_TOKENS_VALUE}"
  VOCAB_SIZE="${VOCAB_SIZE_VALUE}"
  MLP_MULTIPLE_OF="${MLP_MULTIPLE_OF_VALUE}"
  NUM_EPOCHS="${NUM_EPOCHS_VALUE}"
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS_VALUE}"
  COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS_VALUE}"
  CONTROL_TENSOR_NAME_PATTERNS="${CONTROL_TENSOR_NAME_PATTERNS_VALUE}"
)

TRAINER_ENV_TO_CLEAR=(
  ADAM_EPS
  ATTN_DROPOUT
  BETA1
  BETA2
  COMPILE_WARMUP_STEPS
  CONTROL_TENSOR_NAME_PATTERNS
  DATA_BACKEND
  DATA_BYTES_LIMIT
  DATA_PATH
  DEVICE
  DTYPE
  EMBED_DIM
  EMBED_LR
  ENWIK8_PATH
  EVAL_BATCH_TOKENS
  EVAL_MODE
  EVAL_SEQ_LEN
  EXPORT_INT8_PATH
  GRAD_ACCUM_STEPS
  GRAD_CLIP_NORM
  HEAD_LR
  ITERATIONS
  LAYER_SCALE_INIT
  LEARNING_RATE
  LOAD_PATH
  LOGIT_SOFTCAP
  MATRIX_LR
  MAX_STEPS
  MAX_WALLCLOCK_SECONDS
  MIN_LR
  MLP_MULT
  MLP_MULTIPLE_OF
  MODEL_DIM
  MODEL_SUMMARY_MAX_DEPTH
  MODEL_SUMMARY_SHOW_SHAPES
  MUON_BACKEND_STEPS
  MUON_MOMENTUM
  MUON_MOMENTUM_WARMUP_START
  MUON_MOMENTUM_WARMUP_STEPS
  NORM_EPS
  NORM_KIND
  NORM_LAYOUT
  NUM_EPOCHS
  NUM_HEADS
  NUM_KV_HEADS
  NUM_LAYERS
  NUM_SHARED_BLOCKS
  OUT_DIR
  PRINT_MODEL_SUMMARY
  QUANT_KEEP_FLOAT_NUMEL
  QK_GAIN_INIT
  QK_NORM
  REPORT_ARTIFACT
  RESID_DROPOUT
  RESID_MIX_INIT
  ROPE_BASE
  RUN_ID
  SAVE_PATH
  SCALAR_LR
  SDPA_BACKEND
  SEED
  SHARE_PATTERN
  STRICT_LOAD
  TIED_EMBED_INIT_STD
  TIED_EMBED_LR
  TIE_EMBEDDINGS
  X0_GATE_INIT
  TOKENIZER_PATH
  TRAIN_BATCH_TOKENS
  TRAIN_FILES
  TRAIN_LOG_EVERY
  TRAIN_SEQ_LEN
  TRAIN_SPLIT
  USE_BIAS
  USE_FINAL_NORM
  USE_X0_SHORTCUT
  VAL_BATCHES
  VAL_BATCH_SIZE
  VAL_FILES
  VAL_LOSS_EVERY
  VOCAB_SIZE
  WARMDOWN_ITERS
  WARMUP_STEPS
  WEIGHT_DECAY
  ZERO_INIT_RESIDUAL
)
TRAINER_ENV_CLEAR_ARGS=()
for TRAINER_ENV_KEY in "${TRAINER_ENV_TO_CLEAR[@]}"; do
  TRAINER_ENV_CLEAR_ARGS+=(-u "${TRAINER_ENV_KEY}")
done

FAMILIES=(
  wide_s4_e384_ff10
  shortfat_s4_ff15
  balanced_s4_e1472_ff175
  tall_s4_e832_ff2125
)

RUN_BASELINE_BLOCK="${RUN_BASELINE_BLOCK:-${DEFAULT_RUN_BASELINE_BLOCK}}"
RUN_SHARE_BLOCK="${RUN_SHARE_BLOCK:-${DEFAULT_RUN_SHARE_BLOCK}}"
RUN_NORM_BLOCK="${RUN_NORM_BLOCK:-${DEFAULT_RUN_NORM_BLOCK}}"
RUN_SHORTCUT_BLOCK="${RUN_SHORTCUT_BLOCK:-${DEFAULT_RUN_SHORTCUT_BLOCK}}"
RUN_GPT_BASELINE="${RUN_GPT_BASELINE:-1}"
RUN_COMPILE="${RUN_COMPILE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

case "${RUN_GPT_BASELINE}" in
  0|1)
    ;;
  *)
    echo "expected RUN_GPT_BASELINE=0 or 1, got ${RUN_GPT_BASELINE}" >&2
    exit 1
    ;;
esac

VARIANT_COUNT=0
if [[ "${RUN_BASELINE_BLOCK}" == "1" ]]; then
  ((VARIANT_COUNT += 1))
fi
if [[ "${RUN_SHARE_BLOCK}" == "1" ]]; then
  ((VARIANT_COUNT += 2))
fi
if [[ "${RUN_NORM_BLOCK}" == "1" ]]; then
  ((VARIANT_COUNT += 2))
fi
if [[ "${RUN_SHORTCUT_BLOCK}" == "1" ]]; then
  ((VARIANT_COUNT += 2))
fi
ALLAMA_RUNS=$(( VARIANT_COUNT * ${#FAMILIES[@]} ))
TOTAL_RUNS=$(( ALLAMA_RUNS + RUN_GPT_BASELINE ))
LOCAL_BATCH_SIZE=$(( TRAIN_BATCH_TOKENS_VALUE / (GRAD_ACCUM_STEPS_VALUE * TRAIN_SEQ_LEN_VALUE) ))
PLANNED_TRAIN_TOKENS=$(( TRAIN_BATCH_TOKENS_VALUE * MAX_STEPS_VALUE ))
SCHEDULED_ALLAMA_RUNS="${ALLAMA_RUNS}"
SCHEDULED_TOTAL_RUNS=$(( SCHEDULED_ALLAMA_RUNS + RUN_GPT_BASELINE ))

if [[ -n "${ITERATIONS:-}" && -z "${MAX_STEPS:-}" ]]; then
  echo "note=ITERATIONS override is deprecated here; use MAX_STEPS instead"
fi

echo "sweep_profile=${SWEEP_PROFILE} run_id_prefix=${RUN_ID_PREFIX} compile=${RUN_COMPILE} force_rerun=${FORCE_RERUN} out_dir=${OUT_DIR_VALUE} data_path=${DATA_PATH_VALUE} tokenizer_path=${TOKENIZER_PATH_VALUE} vocab_size=${VOCAB_SIZE_VALUE} device=${DEVICE_VALUE} dtype=${DTYPE_VALUE} eval_mode=${EVAL_MODE_VALUE} val_batch_size=${VAL_BATCH_SIZE_VALUE} val_batches=${VAL_BATCHES_VALUE} eval_batch_tokens=${EVAL_BATCH_TOKENS_VALUE} train_batch_tokens=${TRAIN_BATCH_TOKENS_VALUE} grad_accum_steps=${GRAD_ACCUM_STEPS_VALUE} local_batch_size=${LOCAL_BATCH_SIZE} max_steps=${MAX_STEPS_VALUE} planned_train_tokens=${PLANNED_TRAIN_TOKENS} val_loss_every=${VAL_LOSS_EVERY_VALUE} train_log_every=${TRAIN_LOG_EVERY_VALUE} num_epochs=${NUM_EPOCHS_VALUE} max_wallclock_seconds=${MAX_WALLCLOCK_SECONDS_VALUE} compile_warmup_steps=${COMPILE_WARMUP_STEPS_VALUE} sdpa_backend=${SDPA_BACKEND_VALUE} mlp_multiple_of=${MLP_MULTIPLE_OF_VALUE} control_tensor_name_patterns=${CONTROL_TENSOR_NAME_PATTERNS_VALUE} wandb_watch=${WANDB_WATCH} wandb_watch_log_freq=${WANDB_WATCH_LOG_FREQ} trainer_env_sanitized=1 run_gpt_baseline=${RUN_GPT_BASELINE} allama_runs_planned=${ALLAMA_RUNS} allama_runs_scheduled=${SCHEDULED_ALLAMA_RUNS} total_runs_planned=${TOTAL_RUNS} total_runs_scheduled=${SCHEDULED_TOTAL_RUNS}"
echo "family_wide_s4_e384_ff10 model_dim=1024 embed_dim=384 num_layers=16 num_heads=16 num_kv_heads=2 num_shared_blocks=4 mlp_mult=1.0 raw_payload_bytes=23502151 predicted_artifact_bytes_conservative=15987999 predicted_headroom_bytes_conservative=12001"
echo "family_shortfat_s4_ff15 model_dim=896 embed_dim=896 num_layers=20 num_heads=14 num_kv_heads=2 num_shared_blocks=4 mlp_mult=1.5 raw_payload_bytes=23711251 predicted_artifact_bytes_conservative=16128842 predicted_headroom_bytes_conservative=-128842"
echo "family_balanced_s4_e1472_ff175 model_dim=768 embed_dim=1472 num_layers=24 num_heads=12 num_kv_heads=4 num_shared_blocks=4 mlp_mult=1.75 raw_payload_bytes=23356487 predicted_artifact_bytes_conservative=15889885 predicted_headroom_bytes_conservative=110115"
echo "family_tall_s4_e832_ff2125 model_dim=768 embed_dim=832 num_layers=32 num_heads=12 num_kv_heads=2 num_shared_blocks=4 mlp_mult=2.125 raw_payload_bytes=23365831 predicted_artifact_bytes_conservative=15896178 predicted_headroom_bytes_conservative=103822"

run_is_complete () {
  local RUN_DIR="$1"
  local EXPECTED_MAX_STEPS="$2"
  local EXPECTED_MAX_WALLCLOCK_SECONDS="$3"
  local EXPECTED_RUN_SPEC="$4"
  local REQUIRED_LOG_PATTERN_1="$5"
  local REQUIRED_LOG_PATTERN_2="$6"
  shift 6
  local LOG_PATH="${RUN_DIR}/train.log"
  local RUN_SPEC_PATH="${RUN_DIR}/.run_spec"

  for ARTIFACT_REL in "$@"; do
    [[ -f "${RUN_DIR}/${ARTIFACT_REL}" ]] || return 1
  done
  [[ -f "${LOG_PATH}" ]] || return 1
  [[ -f "${RUN_SPEC_PATH}" ]] || return 1

  if [[ "$(<"${RUN_SPEC_PATH}")" != "${EXPECTED_RUN_SPEC}" ]]; then
    return 1
  fi

  if ! rg -q "train_stop step=${EXPECTED_MAX_STEPS} reason=|step=${EXPECTED_MAX_STEPS}/${EXPECTED_MAX_STEPS}" "${LOG_PATH}"; then
    if [[ "${EXPECTED_MAX_WALLCLOCK_SECONDS}" == "0" ]]; then
      return 1
    fi
    local WALLCLOCK_REASON_PATTERN="${EXPECTED_MAX_WALLCLOCK_SECONDS//./\\.}"
    if [[ "${EXPECTED_MAX_WALLCLOCK_SECONDS}" != *.* ]]; then
      WALLCLOCK_REASON_PATTERN="${WALLCLOCK_REASON_PATTERN}(\\.0+)?"
    fi
    if ! rg -q "train_stop step=[0-9]+ reason=max_wallclock_seconds=${WALLCLOCK_REASON_PATTERN}" "${LOG_PATH}"; then
      return 1
    fi
  fi

  if [[ -n "${REQUIRED_LOG_PATTERN_1}" ]] && ! rg -q "${REQUIRED_LOG_PATTERN_1}" "${LOG_PATH}"; then
    return 1
  fi
  if [[ -n "${REQUIRED_LOG_PATTERN_2}" ]] && ! rg -q "${REQUIRED_LOG_PATTERN_2}" "${LOG_PATH}"; then
    return 1
  fi

  if [[ ! -f "${RUN_DIR}/.complete" ]]; then
    touch "${RUN_DIR}/.complete"
  fi
}

run_gpt_reference () {
  local RUN_ID="${RUN_ID_PREFIX}gpt_baseline_reference"
  local RUN_DIR="${OUT_DIR_VALUE}/${RUN_ID}"
  local SAVE_PATH_FILE="${RUN_DIR}/model.pt"
  local EXPORT_INT8_PATH_FILE="${RUN_DIR}/model_int8.ptz"
  local RUN_SPEC_PATH="${RUN_DIR}/.run_spec"
  local RUN_SPEC="family_set=sharedblock_nearcap_v4 kind=train_gpt_reference out_dir=${OUT_DIR_VALUE} data_path=${DATA_PATH_VALUE} tokenizer_path=${TOKENIZER_PATH_VALUE} vocab_size=${VOCAB_SIZE_VALUE} train_seq_len=${TRAIN_SEQ_LEN_VALUE} eval_seq_len=${EVAL_SEQ_LEN_VALUE} train_batch_tokens=${TRAIN_BATCH_TOKENS_VALUE} grad_accum_steps=${GRAD_ACCUM_STEPS_VALUE} max_steps=${MAX_STEPS_VALUE} val_loss_every=${VAL_LOSS_EVERY_VALUE} train_log_every=${TRAIN_LOG_EVERY_VALUE} eval_mode=${EVAL_MODE_VALUE} val_batch_size=${VAL_BATCH_SIZE_VALUE} val_batches=${VAL_BATCHES_VALUE} eval_batch_tokens=${EVAL_BATCH_TOKENS_VALUE} max_wallclock_seconds=${MAX_WALLCLOCK_SECONDS_VALUE} compile_warmup_steps=${COMPILE_WARMUP_STEPS_VALUE} compile=${RUN_COMPILE} wandb_watch=${WANDB_WATCH} wandb_watch_log_freq=${WANDB_WATCH_LOG_FREQ}"

  if [[ "${FORCE_RERUN}" != "1" ]]; then
    if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${MAX_WALLCLOCK_SECONDS_VALUE}" "${RUN_SPEC}" "size_final " "final_int8_zlib_roundtrip_exact " "model.pt" "model_int8.ptz"; then
      echo "skip_existing run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} checkpoint=${SAVE_PATH_FILE} int8=${EXPORT_INT8_PATH_FILE}"
      return 0
    fi
    if [[ -e "${RUN_DIR}" ]]; then
      echo "existing_run_conflict run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} note=directory_exists_but_does_not_match_current_completion_check use_FORCE_RERUN=1_to_overwrite" >&2
      return 1
    fi
  fi

  env "${TRAINER_ENV_CLEAR_ARGS[@]}" \
    OUT_DIR="${OUT_DIR_VALUE}" \
    DATA_PATH="${DATA_PATH_VALUE}" \
    TOKENIZER_PATH="${TOKENIZER_PATH_VALUE}" \
    VOCAB_SIZE="${VOCAB_SIZE_VALUE}" \
    RUN_ID="${RUN_ID}" \
    TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN_VALUE}" \
    EVAL_SEQ_LEN="${EVAL_SEQ_LEN_VALUE}" \
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS_VALUE}" \
    GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS_VALUE}" \
    ITERATIONS="${MAX_STEPS_VALUE}" \
    VAL_LOSS_EVERY="${VAL_LOSS_EVERY_VALUE}" \
    TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY_VALUE}" \
    EVAL_MODE="${EVAL_MODE_VALUE}" \
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE_VALUE}" \
    VAL_BATCHES="${VAL_BATCHES_VALUE}" \
    EVAL_BATCH_TOKENS="${EVAL_BATCH_TOKENS_VALUE}" \
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS_VALUE}" \
    COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS_VALUE}" \
    COMPILE="${RUN_COMPILE}" \
    WANDB_RUN_NAME="${RUN_ID}" \
    WANDB_TAGS="${WANDB_TAGS},baseline,reference,train_gpt" \
    python train_gpt.py

  printf '%s\n' "${RUN_SPEC}" > "${RUN_SPEC_PATH}"

  if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${MAX_WALLCLOCK_SECONDS_VALUE}" "${RUN_SPEC}" "size_final " "final_int8_zlib_roundtrip_exact " "model.pt" "model_int8.ptz"; then
    return 0
  fi

  echo "run_failed_completion_check run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE}" >&2
  return 1
}

run_one () {
  local RUN_ID="$1"
  local FAMILY="$2"
  local VARIANT="$3"
  local MODEL_DIM EMBED_DIM NUM_LAYERS NUM_HEADS NUM_KV_HEADS NUM_SHARED_BLOCKS MLP_MULT
  local NORM_LAYOUT NORM_KIND SHARE_PATTERN USE_X0_SHORTCUT X0_GATE_INIT

  case "$FAMILY" in
    wide_s4_e384_ff10)
      MODEL_DIM=1024
      EMBED_DIM=384
      NUM_LAYERS=16
      NUM_HEADS=16
      NUM_KV_HEADS=2
      NUM_SHARED_BLOCKS=4
      MLP_MULT=1.0
      ;;
    shortfat_s4_ff15)
      MODEL_DIM=896
      EMBED_DIM=896
      NUM_LAYERS=20
      NUM_HEADS=14
      NUM_KV_HEADS=2
      NUM_SHARED_BLOCKS=4
      MLP_MULT=1.5
      ;;
    balanced_s4_e1472_ff175)
      MODEL_DIM=768
      EMBED_DIM=1472
      NUM_LAYERS=24
      NUM_HEADS=12
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=4
      MLP_MULT=1.75
      ;;
    tall_s4_e832_ff2125)
      MODEL_DIM=768
      EMBED_DIM=832
      NUM_LAYERS=32
      NUM_HEADS=12
      NUM_KV_HEADS=2
      NUM_SHARED_BLOCKS=4
      MLP_MULT=2.125
      ;;
    *)
      echo "unknown FAMILY=$FAMILY" >&2
      return 1
      ;;
  esac

  case "$VARIANT" in
    baseline)
      NORM_LAYOUT=postnorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      X0_GATE_INIT=-6.0
      ;;
    share_chunk)
      NORM_LAYOUT=postnorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=chunk
      USE_X0_SHORTCUT=1
      X0_GATE_INIT=-6.0
      ;;
    share_repeat2)
      NORM_LAYOUT=postnorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=repeat_2
      USE_X0_SHORTCUT=1
      X0_GATE_INIT=-6.0
      ;;
    norm_prenorm_rms)
      NORM_LAYOUT=prenorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      X0_GATE_INIT=-6.0
      ;;
    norm_post_rms)
      NORM_LAYOUT=postnorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      X0_GATE_INIT=-6.0
      ;;
    shortcut_gate005)
      NORM_LAYOUT=postnorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      X0_GATE_INIT=-2.9444389792
      ;;
    shortcut_no_x0)
      NORM_LAYOUT=postnorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=0
      X0_GATE_INIT=-6.0
      ;;
    *)
      echo "unknown VARIANT=$VARIANT" >&2
      return 1
      ;;
  esac

  local PYTHON_FLAGS=()
  local RUN_DIR="${OUT_DIR_VALUE}/${RUN_ID}"
  local SAVE_PATH_FILE="${RUN_DIR}/model.pt"
  local EXPORT_INT8_PATH_FILE="${RUN_DIR}/model_int8.pt"
  local RUN_SPEC_PATH="${RUN_DIR}/.run_spec"
  local RUN_SPEC="family_set=sharedblock_nearcap_v4 out_dir=${OUT_DIR_VALUE} data_path=${DATA_PATH_VALUE} tokenizer_path=${TOKENIZER_PATH_VALUE} vocab_size=${VOCAB_SIZE_VALUE} device=${DEVICE_VALUE} dtype=${DTYPE_VALUE} num_epochs=${NUM_EPOCHS_VALUE} max_wallclock_seconds=${MAX_WALLCLOCK_SECONDS_VALUE} family=${FAMILY} variant=${VARIANT} model_dim=${MODEL_DIM} embed_dim=${EMBED_DIM} num_layers=${NUM_LAYERS} num_heads=${NUM_HEADS} num_kv_heads=${NUM_KV_HEADS} num_shared_blocks=${NUM_SHARED_BLOCKS} mlp_mult=${MLP_MULT} mlp_multiple_of=${MLP_MULTIPLE_OF_VALUE} norm_layout=${NORM_LAYOUT} norm_kind=${NORM_KIND} share_pattern=${SHARE_PATTERN} use_x0_shortcut=${USE_X0_SHORTCUT} x0_gate_init=${X0_GATE_INIT} train_seq_len=${TRAIN_SEQ_LEN_VALUE} eval_seq_len=${EVAL_SEQ_LEN_VALUE} train_batch_tokens=${TRAIN_BATCH_TOKENS_VALUE} grad_accum_steps=${GRAD_ACCUM_STEPS_VALUE} max_steps=${MAX_STEPS_VALUE} val_loss_every=${VAL_LOSS_EVERY_VALUE} train_log_every=${TRAIN_LOG_EVERY_VALUE} eval_mode=${EVAL_MODE_VALUE} val_batch_size=${VAL_BATCH_SIZE_VALUE} val_batches=${VAL_BATCHES_VALUE} eval_batch_tokens=${EVAL_BATCH_TOKENS_VALUE} compile_warmup_steps=${COMPILE_WARMUP_STEPS_VALUE} sdpa_backend=${SDPA_BACKEND_VALUE} run_compile=${RUN_COMPILE} control_tensor_name_patterns=${CONTROL_TENSOR_NAME_PATTERNS_VALUE} wandb_watch=${WANDB_WATCH} wandb_watch_log_freq=${WANDB_WATCH_LOG_FREQ}"

  case "${RUN_COMPILE}" in
    1)
      PYTHON_FLAGS+=(--compile)
      ;;
    0)
      PYTHON_FLAGS+=(--no-compile)
      ;;
    *)
      echo "expected RUN_COMPILE=0 or 1, got ${RUN_COMPILE}" >&2
      return 1
      ;;
  esac

  if [[ "${FORCE_RERUN}" != "1" ]]; then
    if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${MAX_WALLCLOCK_SECONDS_VALUE}" "${RUN_SPEC}" "size_final " "" "model.pt" "model_int8.pt"; then
      echo "skip_existing run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} checkpoint=${SAVE_PATH_FILE} int8=${EXPORT_INT8_PATH_FILE}"
      return 0
    fi
    if [[ -e "${RUN_DIR}" ]]; then
      echo "existing_run_conflict run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} note=directory_exists_but_does_not_match_current_completion_check use_FORCE_RERUN=1_to_overwrite" >&2
      return 1
    fi
  fi

  env "${TRAINER_ENV_CLEAR_ARGS[@]}" "${BASE_ENV[@]}" \
    RUN_ID="$RUN_ID" \
    MODEL_DIM="$MODEL_DIM" \
    EMBED_DIM="$EMBED_DIM" \
    NUM_LAYERS="$NUM_LAYERS" \
    NUM_HEADS="$NUM_HEADS" \
    NUM_KV_HEADS="$NUM_KV_HEADS" \
    NUM_SHARED_BLOCKS="$NUM_SHARED_BLOCKS" \
    MLP_MULT="$MLP_MULT" \
    NORM_LAYOUT="$NORM_LAYOUT" \
    NORM_KIND="$NORM_KIND" \
    SHARE_PATTERN="$SHARE_PATTERN" \
    USE_X0_SHORTCUT="$USE_X0_SHORTCUT" \
    X0_GATE_INIT="$X0_GATE_INIT" \
    SAVE_PATH="${SAVE_PATH_FILE}" \
    EXPORT_INT8_PATH="${EXPORT_INT8_PATH_FILE}" \
    python train_allama.py "${PYTHON_FLAGS[@]}"

  printf '%s\n' "${RUN_SPEC}" > "${RUN_SPEC_PATH}"

  if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${MAX_WALLCLOCK_SECONDS_VALUE}" "${RUN_SPEC}" "size_final " "" "model.pt" "model_int8.pt"; then
    return 0
  fi

  echo "run_failed_completion_check run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE}" >&2
  return 1
}

run_variant_block() {
  local VARIANT="$1"
  for FAMILY in "${FAMILIES[@]}"; do
    run_one "${RUN_ID_PREFIX}${FAMILY}_${VARIANT}" "${FAMILY}" "${VARIANT}"
  done
}

# These anchors were re-calibrated under the corrected clipped exporter. They keep
# head_dim=64 and spend size budget primarily via NUM_SHARED_BLOCKS, because
# logical NUM_LAYERS changes runtime much more than final artifact size.

if [[ "${RUN_GPT_BASELINE}" == "1" ]]; then
  run_gpt_reference
fi

if [[ "${RUN_BASELINE_BLOCK}" == "1" ]]; then
  run_variant_block baseline
fi

if [[ "${RUN_SHARE_BLOCK}" == "1" ]]; then
  run_variant_block share_chunk
  run_variant_block share_repeat2
fi

if [[ "${RUN_NORM_BLOCK}" == "1" ]]; then
  run_variant_block norm_prenorm_rms
  run_variant_block norm_post_rms
fi

if [[ "${RUN_SHORTCUT_BLOCK}" == "1" ]]; then
  run_variant_block shortcut_gate005
  run_variant_block shortcut_no_x0
fi
