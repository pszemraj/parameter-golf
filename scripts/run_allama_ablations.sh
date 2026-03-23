#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP="${WANDB_GROUP:-allama-aligned-blocked-ablations}"
export WANDB_TAGS="${WANDB_TAGS:-5090,allama,nearcap,finalsize,behavior,blocked,aligned}"
export WANDB_WATCH="${WANDB_WATCH:-all}"
export WANDB_WATCH_LOG_FREQ="${WANDB_WATCH_LOG_FREQ:-100}"
export SDPA_BACKEND="${SDPA_BACKEND:-auto}"
export TORCH_BLAS_PREFER_CUBLASLT=1

SWEEP_PROFILE="${SWEEP_PROFILE:-explore}"

case "${SWEEP_PROFILE}" in
  screen)
    DEFAULT_TRAIN_BATCH_TOKENS=262144
    DEFAULT_GRAD_ACCUM_STEPS=128
    DEFAULT_MAX_STEPS=750
    DEFAULT_VAL_LOSS_EVERY=250
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
    DEFAULT_RUN_BASELINE_BLOCK=1
    DEFAULT_RUN_SHARE_BLOCK=1
    DEFAULT_RUN_NORM_BLOCK=1
    DEFAULT_RUN_SHORTCUT_BLOCK=1
    ;;
  *)
    echo "unknown SWEEP_PROFILE=${SWEEP_PROFILE}" >&2
    echo "expected SWEEP_PROFILE=screen, explore, or full" >&2
    exit 1
    ;;
esac

TRAIN_SEQ_LEN_VALUE="${TRAIN_SEQ_LEN:-1024}"
EVAL_SEQ_LEN_VALUE="${EVAL_SEQ_LEN:-1024}"
TRAIN_BATCH_TOKENS_VALUE="${TRAIN_BATCH_TOKENS:-${DEFAULT_TRAIN_BATCH_TOKENS}}"
GRAD_ACCUM_STEPS_VALUE="${GRAD_ACCUM_STEPS:-${DEFAULT_GRAD_ACCUM_STEPS}}"
MAX_STEPS_VALUE="${MAX_STEPS:-${ITERATIONS:-${DEFAULT_MAX_STEPS}}}"
VAL_LOSS_EVERY_VALUE="${VAL_LOSS_EVERY:-${DEFAULT_VAL_LOSS_EVERY}}"
CONTROL_TENSOR_NAME_PATTERNS_VALUE="${CONTROL_TENSOR_NAME_PATTERNS:-depth_gains}"

BASE_ENV=(
  OUT_DIR=./runs_allama
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  DEVICE=cuda
  DTYPE=bf16
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN_VALUE}"
  EVAL_SEQ_LEN="${EVAL_SEQ_LEN_VALUE}"
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS_VALUE}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS_VALUE}"
  MAX_STEPS="${MAX_STEPS_VALUE}"
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY_VALUE}"
  TRAIN_LOG_EVERY=25
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
  MLP_MULTIPLE_OF=128
  CONTROL_TENSOR_NAME_PATTERNS="${CONTROL_TENSOR_NAME_PATTERNS_VALUE}"
)

FAMILIES=(
  wide_ff15
  shortfat_ff20
  balanced_ff25
  tall_ff30
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
ALLAMA_STALE_BLOCKED=0
if [[ "${ALLOW_STALE_FAMILY_SET:-0}" != "1" && "${ALLAMA_RUNS}" -gt 0 ]]; then
  ALLAMA_STALE_BLOCKED=1
fi
SCHEDULED_ALLAMA_RUNS="${ALLAMA_RUNS}"
if [[ "${ALLAMA_STALE_BLOCKED}" == "1" ]]; then
  SCHEDULED_ALLAMA_RUNS=0
fi
SCHEDULED_TOTAL_RUNS=$(( SCHEDULED_ALLAMA_RUNS + RUN_GPT_BASELINE ))

if [[ -n "${ITERATIONS:-}" && -z "${MAX_STEPS:-}" ]]; then
  echo "note=ITERATIONS override is deprecated here; use MAX_STEPS instead"
fi

echo "sweep_profile=${SWEEP_PROFILE} compile=${RUN_COMPILE} force_rerun=${FORCE_RERUN} train_batch_tokens=${TRAIN_BATCH_TOKENS_VALUE} grad_accum_steps=${GRAD_ACCUM_STEPS_VALUE} local_batch_size=${LOCAL_BATCH_SIZE} max_steps=${MAX_STEPS_VALUE} val_loss_every=${VAL_LOSS_EVERY_VALUE} control_tensor_name_patterns=${CONTROL_TENSOR_NAME_PATTERNS_VALUE} run_gpt_baseline=${RUN_GPT_BASELINE} allama_runs_planned=${ALLAMA_RUNS} allama_runs_scheduled=${SCHEDULED_ALLAMA_RUNS} total_runs_planned=${TOTAL_RUNS} total_runs_scheduled=${SCHEDULED_TOTAL_RUNS} stale_allama_guard=${ALLAMA_STALE_BLOCKED}"

run_is_complete () {
  local RUN_DIR="$1"
  local EXPECTED_MAX_STEPS="$2"
  local EXPECTED_RUN_SPEC="$3"
  shift 3
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
    return 1
  fi

  if [[ ! -f "${RUN_DIR}/.complete" ]]; then
    touch "${RUN_DIR}/.complete"
  fi
}

run_gpt_reference () {
  local RUN_ID="gpt_baseline_reference"
  local RUN_DIR="./runs_allama/${RUN_ID}"
  local SAVE_PATH_FILE="${RUN_DIR}/model.pt"
  local EXPORT_INT8_PATH_FILE="${RUN_DIR}/model_int8.ptz"
  local RUN_SPEC_PATH="${RUN_DIR}/.run_spec"
  local RUN_SPEC="kind=train_gpt_reference train_seq_len=${TRAIN_SEQ_LEN_VALUE} eval_seq_len=${EVAL_SEQ_LEN_VALUE} train_batch_tokens=${TRAIN_BATCH_TOKENS_VALUE} max_steps=${MAX_STEPS_VALUE} val_loss_every=${VAL_LOSS_EVERY_VALUE} train_log_every=25 eval_mode=sampled val_batch_size=8 val_batches=8 max_wallclock_seconds=0"

  if [[ "${FORCE_RERUN}" != "1" ]]; then
    if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${RUN_SPEC}" "model.pt" "model_int8.ptz"; then
      echo "skip_existing run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} checkpoint=${SAVE_PATH_FILE} int8=${EXPORT_INT8_PATH_FILE}"
      return 0
    fi
    if [[ -e "${RUN_DIR}" ]]; then
      echo "existing_run_conflict run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} note=directory_exists_but_does_not_match_current_completion_check use_FORCE_RERUN=1_to_overwrite" >&2
      return 1
    fi
  fi

  env \
    OUT_DIR=./runs_allama \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    RUN_ID="${RUN_ID}" \
    TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN_VALUE}" \
    EVAL_SEQ_LEN="${EVAL_SEQ_LEN_VALUE}" \
    TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS_VALUE}" \
    ITERATIONS="${MAX_STEPS_VALUE}" \
    VAL_LOSS_EVERY="${VAL_LOSS_EVERY_VALUE}" \
    TRAIN_LOG_EVERY=25 \
    EVAL_MODE=sampled \
    VAL_BATCH_SIZE=8 \
    VAL_BATCHES=8 \
    MAX_WALLCLOCK_SECONDS=0 \
    WANDB_RUN_NAME="${RUN_ID}" \
    WANDB_TAGS="${WANDB_TAGS},baseline,reference,train_gpt" \
    python train_gpt.py

  printf '%s\n' "${RUN_SPEC}" > "${RUN_SPEC_PATH}"

  if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${RUN_SPEC}" "model.pt" "model_int8.ptz"; then
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
  local NORM_LAYOUT NORM_KIND SHARE_PATTERN USE_X0_SHORTCUT RESID_MIX_INIT

  case "$FAMILY" in
    wide_ff15)
      MODEL_DIM=1024
      EMBED_DIM=256
      NUM_LAYERS=16
      NUM_HEADS=8
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=1.5
      ;;
    shortfat_ff20)
      MODEL_DIM=896
      EMBED_DIM=1152
      NUM_LAYERS=20
      NUM_HEADS=14
      NUM_KV_HEADS=2
      NUM_SHARED_BLOCKS=2
      MLP_MULT=2.0
      ;;
    balanced_ff25)
      MODEL_DIM=768
      EMBED_DIM=1728
      NUM_LAYERS=24
      NUM_HEADS=12
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=2.5
      ;;
    tall_ff30)
      MODEL_DIM=768
      EMBED_DIM=1088
      NUM_LAYERS=32
      NUM_HEADS=12
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=3.0
      ;;
    *)
      echo "unknown FAMILY=$FAMILY" >&2
      return 1
      ;;
  esac

  case "$VARIANT" in
    baseline)
      NORM_LAYOUT=postnorm
      NORM_KIND=layernorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      RESID_MIX_INIT=0.10
      ;;
    share_chunk)
      NORM_LAYOUT=postnorm
      NORM_KIND=layernorm
      SHARE_PATTERN=chunk
      USE_X0_SHORTCUT=1
      RESID_MIX_INIT=0.10
      ;;
    share_repeat2)
      NORM_LAYOUT=postnorm
      NORM_KIND=layernorm
      SHARE_PATTERN=repeat_2
      USE_X0_SHORTCUT=1
      RESID_MIX_INIT=0.10
      ;;
    norm_prenorm_ln)
      NORM_LAYOUT=prenorm
      NORM_KIND=layernorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      RESID_MIX_INIT=0.10
      ;;
    norm_post_rms)
      NORM_LAYOUT=postnorm
      NORM_KIND=rmsnorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      RESID_MIX_INIT=0.10
      ;;
    shortcut_mix005)
      NORM_LAYOUT=postnorm
      NORM_KIND=layernorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=1
      RESID_MIX_INIT=0.05
      ;;
    shortcut_no_x0)
      NORM_LAYOUT=postnorm
      NORM_KIND=layernorm
      SHARE_PATTERN=cycle
      USE_X0_SHORTCUT=0
      RESID_MIX_INIT=0.00
      ;;
    *)
      echo "unknown VARIANT=$VARIANT" >&2
      return 1
      ;;
  esac

  RUN_SPEC="family=${FAMILY} variant=${VARIANT} model_dim=${MODEL_DIM} embed_dim=${EMBED_DIM} num_layers=${NUM_LAYERS} num_heads=${NUM_HEADS} num_kv_heads=${NUM_KV_HEADS} num_shared_blocks=${NUM_SHARED_BLOCKS} mlp_mult=${MLP_MULT} norm_layout=${NORM_LAYOUT} norm_kind=${NORM_KIND} share_pattern=${SHARE_PATTERN} use_x0_shortcut=${USE_X0_SHORTCUT} resid_mix_init=${RESID_MIX_INIT} train_seq_len=${TRAIN_SEQ_LEN_VALUE} eval_seq_len=${EVAL_SEQ_LEN_VALUE} train_batch_tokens=${TRAIN_BATCH_TOKENS_VALUE} grad_accum_steps=${GRAD_ACCUM_STEPS_VALUE} max_steps=${MAX_STEPS_VALUE} val_loss_every=${VAL_LOSS_EVERY_VALUE} run_compile=${RUN_COMPILE} control_tensor_name_patterns=${CONTROL_TENSOR_NAME_PATTERNS_VALUE}"

  local PYTHON_FLAGS=()
  local RUN_DIR="./runs_allama/${RUN_ID}"
  local SAVE_PATH_FILE="${RUN_DIR}/model.pt"
  local EXPORT_INT8_PATH_FILE="${RUN_DIR}/model_int8.pt"
  local RUN_SPEC_PATH="${RUN_DIR}/.run_spec"
  local RUN_SPEC

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
    if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${RUN_SPEC}" "model.pt" "model_int8.pt"; then
      echo "skip_existing run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} checkpoint=${SAVE_PATH_FILE} int8=${EXPORT_INT8_PATH_FILE}"
      return 0
    fi
    if [[ -e "${RUN_DIR}" ]]; then
      echo "existing_run_conflict run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE} note=directory_exists_but_does_not_match_current_completion_check use_FORCE_RERUN=1_to_overwrite" >&2
      return 1
    fi
  fi

  env "${BASE_ENV[@]}" \
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
    RESID_MIX_INIT="$RESID_MIX_INIT" \
    SAVE_PATH="${SAVE_PATH_FILE}" \
    EXPORT_INT8_PATH="${EXPORT_INT8_PATH_FILE}" \
    python train_allama_reborn.py "${PYTHON_FLAGS[@]}"

  printf '%s\n' "${RUN_SPEC}" > "${RUN_SPEC_PATH}"

  if run_is_complete "${RUN_DIR}" "${MAX_STEPS_VALUE}" "${RUN_SPEC}" "model.pt" "model_int8.pt"; then
    return 0
  fi

  echo "run_failed_completion_check run_id=${RUN_ID} max_steps=${MAX_STEPS_VALUE}" >&2
  return 1
}

run_variant_block() {
  local VARIANT="$1"
  for FAMILY in "${FAMILIES[@]}"; do
    run_one "${FAMILY}_${VARIANT}" "${FAMILY}" "${VARIANT}"
  done
}

# This family set is intentionally blocked by the stale-family guard above.
# After switching to the clipped exporter, these anchors are no longer near-cap
# and must be recalibrated before this script should be used for real sweeps.
# Cold compile speed audit on the 5090 favored these aligned shapes over the old
# 1056/960/864/832 anchors while the Inductor online-softmax warning persisted.

if [[ "${RUN_GPT_BASELINE}" == "1" ]]; then
  run_gpt_reference
fi

if [[ "${ALLAMA_STALE_BLOCKED}" == "1" ]]; then
  echo "ERROR current allama family anchors are stale under the clipped exporter policy." >&2
  echo "ERROR finished baselines now re-export around 7.4-7.7 MB, so these allama runs would waste budget and compute." >&2
  echo "ERROR recalibrate family sizes first, or set ALLOW_STALE_FAMILY_SET=1 only if you intentionally want stale undersized runs." >&2
  exit 1
fi

if [[ "${RUN_BASELINE_BLOCK}" == "1" ]]; then
  run_variant_block baseline
fi

if [[ "${RUN_SHARE_BLOCK}" == "1" ]]; then
  run_variant_block share_chunk
  run_variant_block share_repeat2
fi

if [[ "${RUN_NORM_BLOCK}" == "1" ]]; then
  run_variant_block norm_prenorm_ln
  run_variant_block norm_post_rms
fi

if [[ "${RUN_SHORTCUT_BLOCK}" == "1" ]]; then
  run_variant_block shortcut_mix005
  run_variant_block shortcut_no_x0
fi
