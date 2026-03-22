#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP=allama-nearcap-train
export WANDB_TAGS=5090,allama,nearcap,behavior

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

run_one () {
  local RUN_ID="$1"
  local FAMILY="$2"
  local NORM_LAYOUT="$3"
  local NORM_KIND="$4"
  local SHARE_PATTERN="$5"
  local USE_X0_SHORTCUT="$6"
  local RESID_MIX_INIT="$7"

  local MODEL_DIM EMBED_DIM NUM_LAYERS NUM_HEADS NUM_KV_HEADS NUM_SHARED_BLOCKS MLP_MULT

  case "$FAMILY" in
    wide_ff15)
      MODEL_DIM=1280
      EMBED_DIM=320
      NUM_LAYERS=16
      NUM_HEADS=16
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=1.5
      ;;
    shortfat_ff20)
      MODEL_DIM=1152
      EMBED_DIM=288
      NUM_LAYERS=20
      NUM_HEADS=16
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=2.0
      ;;
    balanced_ff25)
      MODEL_DIM=1056
      EMBED_DIM=264
      NUM_LAYERS=24
      NUM_HEADS=16
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=2.5
      ;;
    tall_ff30)
      MODEL_DIM=992
      EMBED_DIM=248
      NUM_LAYERS=32
      NUM_HEADS=16
      NUM_KV_HEADS=4
      NUM_SHARED_BLOCKS=2
      MLP_MULT=3.0
      ;;
    tallmulti_ff25)
      MODEL_DIM=896
      EMBED_DIM=224
      NUM_LAYERS=40
      NUM_HEADS=14
      NUM_KV_HEADS=2
      NUM_SHARED_BLOCKS=3
      MLP_MULT=2.5
      ;;
    *)
      echo "unknown FAMILY=$FAMILY" >&2
      return 1
      ;;
  esac

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
    python train_allama_reborn.py
}

# Measured compressed artifact sizes from direct EVAL_ONLY probes:
# - wide_ff15:      15,994,336 bytes
# - shortfat_ff20:  15,659,236 bytes
# - balanced_ff25:  15,496,683 bytes
# - tall_ff30:      15,614,669 bytes
# - tallmulti_ff25: 15,882,909 bytes

run_one wide_ff15_cycle_x0_010      wide_ff15      postnorm layernorm cycle    1 0.10
run_one shortfat_ff20_cycle_x0_010  shortfat_ff20  postnorm layernorm cycle    1 0.10
run_one balanced_ff25_cycle_x0_010  balanced_ff25  postnorm layernorm cycle    1 0.10
run_one tall_ff30_cycle_x0_010      tall_ff30      postnorm layernorm cycle    1 0.10
run_one tallmulti_ff25_cycle_x0_010 tallmulti_ff25 postnorm layernorm cycle    1 0.10

run_one balanced_ff25_pre_ln        balanced_ff25  prenorm  layernorm cycle    1 0.10
run_one balanced_ff25_post_rms      balanced_ff25  postnorm rmsnorm   cycle    1 0.10
run_one tall_ff30_repeat2           tall_ff30      postnorm layernorm repeat_2 1 0.10
run_one shortfat_ff20_no_x0         shortfat_ff20  postnorm layernorm cycle    0 0.00
run_one wide_ff15_chunk             wide_ff15      postnorm layernorm chunk    1 0.10
