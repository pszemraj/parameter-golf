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
    DEFAULT_ITERATIONS=750
    DEFAULT_VAL_LOSS_EVERY=250
    DEFAULT_RUN_BASELINE_BLOCK=1
    DEFAULT_RUN_SHARE_BLOCK=0
    DEFAULT_RUN_NORM_BLOCK=0
    DEFAULT_RUN_SHORTCUT_BLOCK=0
    ;;
  explore)
    DEFAULT_TRAIN_BATCH_TOKENS=262144
    DEFAULT_GRAD_ACCUM_STEPS=128
    DEFAULT_ITERATIONS=750
    DEFAULT_VAL_LOSS_EVERY=250
    DEFAULT_RUN_BASELINE_BLOCK=1
    DEFAULT_RUN_SHARE_BLOCK=1
    DEFAULT_RUN_NORM_BLOCK=1
    DEFAULT_RUN_SHORTCUT_BLOCK=1
    ;;
  full)
    DEFAULT_TRAIN_BATCH_TOKENS=262144
    DEFAULT_GRAD_ACCUM_STEPS=128
    DEFAULT_ITERATIONS=2000
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
ITERATIONS_VALUE="${ITERATIONS:-${DEFAULT_ITERATIONS}}"
VAL_LOSS_EVERY_VALUE="${VAL_LOSS_EVERY:-${DEFAULT_VAL_LOSS_EVERY}}"

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
  ITERATIONS="${ITERATIONS_VALUE}"
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY_VALUE}"
  TRAIN_LOG_EVERY=25
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
  MLP_MULTIPLE_OF=128
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
RUN_COMPILE="${RUN_COMPILE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

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
TOTAL_RUNS=$(( VARIANT_COUNT * ${#FAMILIES[@]} ))
LOCAL_BATCH_SIZE=$(( TRAIN_BATCH_TOKENS_VALUE / (GRAD_ACCUM_STEPS_VALUE * TRAIN_SEQ_LEN_VALUE) ))

echo "sweep_profile=${SWEEP_PROFILE} compile=${RUN_COMPILE} force_rerun=${FORCE_RERUN} train_batch_tokens=${TRAIN_BATCH_TOKENS_VALUE} grad_accum_steps=${GRAD_ACCUM_STEPS_VALUE} local_batch_size=${LOCAL_BATCH_SIZE} iterations=${ITERATIONS_VALUE} val_loss_every=${VAL_LOSS_EVERY_VALUE} total_runs=${TOTAL_RUNS}"

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
      EMBED_DIM=1792
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

  local PYTHON_FLAGS=()
  local RUN_DIR="./runs_allama/${RUN_ID}"
  local SAVE_PATH_FILE="${RUN_DIR}/model.pt"
  local EXPORT_INT8_PATH_FILE="${RUN_DIR}/model_int8.pt"
  if [[ "${RUN_COMPILE}" == "1" ]]; then
    PYTHON_FLAGS+=(--compile)
  fi

  if [[ "${FORCE_RERUN}" != "1" && -f "${SAVE_PATH_FILE}" && -f "${EXPORT_INT8_PATH_FILE}" ]]; then
    echo "skip_existing run_id=${RUN_ID} checkpoint=${SAVE_PATH_FILE} int8=${EXPORT_INT8_PATH_FILE}"
    return 0
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

  if [[ -f "${SAVE_PATH_FILE}" && -f "${EXPORT_INT8_PATH_FILE}" ]]; then
    touch "${RUN_DIR}/.complete"
  fi
}

run_variant_block() {
  local VARIANT="$1"
  for FAMILY in "${FAMILIES[@]}"; do
    run_one "${FAMILY}_${VARIANT}" "${FAMILY}" "${VARIANT}"
  done
}

# Final-size-aware aligned anchors selected from short-horizon sizing checks.
# Safe proxy: code_bytes + 0.945 * int8_payload_bytes_init
# All families use MLP_MULTIPLE_OF=128 and 64/128-friendly dimensions.
# - wide_ff15     1024/256/16  h8/kv4  hd128 hidden1536 qkv2048 -> proxy945 = 15,896,523
# - shortfat_ff20  896/1152/20 h14/kv2 hd64  hidden1792 qkv1152 -> proxy945 = 15,935,714
# - balanced_ff25  768/1792/24 h12/kv4 hd64  hidden1920 qkv1280 -> proxy945 = 15,969,643
# - tall_ff30      768/1088/32 h12/kv4 hd64  hidden2304 qkv1280 -> proxy945 = 15,986,759
# 20-step checked-size sanity at the actual sweep batch:
# - wide_ff15     -> 14,959,072 bytes
# - shortfat_ff20 -> 14,990,094 bytes
# - balanced_ff25 -> 15,236,675 bytes
# - tall_ff30     -> 15,115,258 bytes
# Closest layer-upsized aligned variants were rejected because they added very
# little checked size for a real speed hit:
# - wide 16->24:   +41,622 bytes, about -33% eager tok/s
# - shortfat 20->24: +12,186 bytes, about -16% eager tok/s
# - balanced 24->28: +3,018 bytes, about -14% eager tok/s
# - tall 32->34:   +7,811 bytes, about -6% eager tok/s
# Cold compile speed audit on the 5090 favored these aligned shapes over the old
# 1056/960/864/832 anchors while the Inductor online-softmax warning persisted.

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
