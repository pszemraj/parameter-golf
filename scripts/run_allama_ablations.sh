#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP=allama-budget-matched-ablations
export WANDB_TAGS=5090,allama,budget-matched

BASE_ENV=(
  OUT_DIR=./runs_allama
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  DEVICE=cuda
  DTYPE=bf16
  MODEL_DIM=1024
  EMBED_DIM=256
  NUM_LAYERS=24
  NUM_SHARED_BLOCKS=2
  NUM_HEADS=16
  NUM_KV_HEADS=4
  MLP_MULT=2.5
  TRAIN_SEQ_LEN=1024
  EVAL_SEQ_LEN=1024
  TRAIN_BATCH_TOKENS=65536
  GRAD_ACCUM_STEPS=4
  NUM_EPOCHS=1
  VAL_LOSS_EVERY=250
  TRAIN_LOG_EVERY=25
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
  USE_X0_SHORTCUT=1
  RESID_MIX_INIT=0.1
)

run_one () {
  local RUN_ID="$1"
  local NORM_LAYOUT="$2"
  local NORM_KIND="$3"
  local SHARE_PATTERN="$4"
  local USE_X0_SHORTCUT="$5"
  local RESID_MIX_INIT="$6"

  env "${BASE_ENV[@]}" \
    RUN_ID="$RUN_ID" \
    NORM_LAYOUT="$NORM_LAYOUT" \
    NORM_KIND="$NORM_KIND" \
    SHARE_PATTERN="$SHARE_PATTERN" \
    USE_X0_SHORTCUT="$USE_X0_SHORTCUT" \
    RESID_MIX_INIT="$RESID_MIX_INIT" \
    SAVE_PATH="./runs_allama/${RUN_ID}/model.pt" \
    EXPORT_INT8_PATH="./runs_allama/${RUN_ID}/model_int8.pt" \
    python train_allama_reborn.py
}

run_one post_ln_chunk_x0_010   postnorm layernorm chunk    1 0.10
run_one post_ln_cycle_x0_010   postnorm layernorm cycle    1 0.10
run_one post_ln_repeat2_x0_010 postnorm layernorm repeat_2 1 0.10
run_one pre_ln_cycle_x0_010    prenorm  layernorm cycle    1 0.10
run_one post_rms_cycle_x0_010  postnorm rmsnorm   cycle    1 0.10
run_one pre_rms_cycle_x0_010   prenorm  rmsnorm   cycle    1 0.10
run_one post_ln_cycle_x0_005   postnorm layernorm cycle    1 0.05
run_one post_ln_cycle_no_x0    postnorm layernorm cycle    0 0.00
