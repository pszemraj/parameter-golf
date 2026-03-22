#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP="${WANDB_GROUP:-allama-blocked-ablations}"
export WANDB_TAGS="${WANDB_TAGS:-5090,allama,nearcap,finalsize,behavior,blocked}"
export TORCH_BLAS_PREFER_CUBLASLT=1

BASE_ENV=(
  OUT_DIR=./runs_allama
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  DEVICE=cuda
  DTYPE=bf16
  TRAIN_SEQ_LEN=1024
  EVAL_SEQ_LEN=1024
  TRAIN_BATCH_TOKENS=65536
  GRAD_ACCUM_STEPS=4
  ITERATIONS=2000
  VAL_LOSS_EVERY=250
  TRAIN_LOG_EVERY=25
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
)

FAMILIES=(
  wide_ff15
  shortfat_ff20
  balanced_ff25
  tall_ff30
)

RUN_BASELINE_BLOCK="${RUN_BASELINE_BLOCK:-1}"
RUN_SHARE_BLOCK="${RUN_SHARE_BLOCK:-1}"
RUN_NORM_BLOCK="${RUN_NORM_BLOCK:-1}"
RUN_SHORTCUT_BLOCK="${RUN_SHORTCUT_BLOCK:-1}"
RUN_COMPILE="${RUN_COMPILE:-1}"

run_one () {
  local RUN_ID="$1"
  local FAMILY="$2"
  local VARIANT="$3"
  local MODEL_DIM EMBED_DIM NUM_LAYERS NUM_HEADS NUM_KV_HEADS NUM_SHARED_BLOCKS MLP_MULT
  local NORM_LAYOUT NORM_KIND SHARE_PATTERN USE_X0_SHORTCUT RESID_MIX_INIT

  case "$FAMILY" in
    wide_ff15)
      MODEL_DIM=1056
      EMBED_DIM=264
      NUM_LAYERS=16
      NUM_HEADS=16
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=1.5
      ;;
    shortfat_ff20)
      MODEL_DIM=960
      EMBED_DIM=240
      NUM_LAYERS=20
      NUM_HEADS=16
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=2.0
      ;;
    balanced_ff25)
      MODEL_DIM=864
      EMBED_DIM=216
      NUM_LAYERS=24
      NUM_HEADS=16
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=2.5
      ;;
    tall_ff30)
      MODEL_DIM=832
      EMBED_DIM=208
      NUM_LAYERS=32
      NUM_HEADS=16
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
  if [[ "${RUN_COMPILE}" == "1" ]]; then
    PYTHON_FLAGS+=(--compile)
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
    SAVE_PATH="./runs_allama/${RUN_ID}/model.pt" \
    EXPORT_INT8_PATH="./runs_allama/${RUN_ID}/model_int8.pt" \
    python train_allama_reborn.py "${PYTHON_FLAGS[@]}"
}

run_variant_block() {
  local VARIANT="$1"
  for FAMILY in "${FAMILIES[@]}"; do
    run_one "${FAMILY}_${VARIANT}" "${FAMILY}" "${VARIANT}"
  done
}

# Final-size-aware anchors selected from short-horizon sizing checks.
# Safe proxy: code_bytes + 0.945 * int8_payload_bytes_init
# - wide_ff15     1056/264/16  -> size_final@50 = 15,826,095
# - shortfat_ff20  960/240/20  -> proxy945      = 15,777,074
# - balanced_ff25  864/216/24  -> proxy945      = 15,062,987
#   note: 896/224/24 reached size_final@50 = 16,048,727, so the next valid 16-head step down is used
# - tall_ff30      832/208/32  -> size_final@50 = 15,869,614

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
