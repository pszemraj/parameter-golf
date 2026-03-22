set -e
# run this from the repo root

export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP=allama-short-5090
export WANDB_TAGS=5090,allama,core,short-train

BASE_ENV=(
  OUT_DIR=./runs_allama_reborn
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  DEVICE=cuda
  DTYPE=bf16
  MODEL_DIM=512
  EMBED_DIM=128
  NUM_LAYERS=12
  NUM_HEADS=8
  NUM_KV_HEADS=4
  MLP_MULT=2.0
  TRAIN_SEQ_LEN=1024
  EVAL_SEQ_LEN=1024
  TRAIN_BATCH_TOKENS=65536
  ITERATIONS=2000
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
  local NUM_SHARED_BLOCKS="$4"
  local SHARE_PATTERN="$5"

  env "${BASE_ENV[@]}" \
    RUN_ID="$RUN_ID" \
    NORM_LAYOUT="$NORM_LAYOUT" \
    NORM_KIND="$NORM_KIND" \
    NUM_SHARED_BLOCKS="$NUM_SHARED_BLOCKS" \
    SHARE_PATTERN="$SHARE_PATTERN" \
    SAVE_PATH="./runs_allama_reborn/${RUN_ID}/model.pt" \
    EXPORT_INT8_PATH="./runs_allama_reborn/${RUN_ID}/model_int8.pt" \
    python train_allama_reborn.py
}

run_one post_ln_share1_chunk   postnorm layernorm 1 chunk
run_one post_ln_share2_chunk   postnorm layernorm 2 chunk
run_one post_ln_share2_cycle   postnorm layernorm 2 cycle
run_one post_ln_share4_chunk   postnorm layernorm 4 chunk
run_one post_ln_share4_cycle   postnorm layernorm 4 cycle
run_one post_ln_share2_repeat2 postnorm layernorm 2 repeat_2
run_one pre_ln_share1_chunk    prenorm  layernorm 1 chunk
run_one pre_ln_share2_cycle    prenorm  layernorm 2 cycle
