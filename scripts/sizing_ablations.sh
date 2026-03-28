#!/usr/bin/env bash
set -euo pipefail

export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP="${WANDB_GROUP:-allama-size-probes-v4}"
export WANDB_TAGS="${WANDB_TAGS:-5090,allama,size-probe,nearcap,v4}"
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
  NUM_EPOCHS=1
  EVAL_ONLY=1
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
  MLP_MULTIPLE_OF=128
  USE_X0_SHORTCUT=1
  X0_GATE_INIT=-6.0
  NORM_LAYOUT=postnorm
  NORM_KIND=rmsnorm
)

probe_one () {
  local RUN_ID="$1"
  local MODEL_DIM="$2"
  local EMBED_DIM="$3"
  local NUM_LAYERS="$4"
  local NUM_HEADS="$5"
  local NUM_KV_HEADS="$6"
  local MLP_MULT="$7"
  local NUM_SHARED_BLOCKS="$8"
  local SHARE_PATTERN="$9"

  env "${BASE_ENV[@]}" \
    RUN_ID="${RUN_ID}" \
    MODEL_DIM="${MODEL_DIM}" \
    EMBED_DIM="${EMBED_DIM}" \
    NUM_LAYERS="${NUM_LAYERS}" \
    NUM_HEADS="${NUM_HEADS}" \
    NUM_KV_HEADS="${NUM_KV_HEADS}" \
    MLP_MULT="${MLP_MULT}" \
    NUM_SHARED_BLOCKS="${NUM_SHARED_BLOCKS}" \
    SHARE_PATTERN="${SHARE_PATTERN}" \
    python train_allama.py
}

# current sweep anchors
probe_one probe_wide_s4_e384_ff10 1024 384 16 16 2 1.0 4 cycle
probe_one probe_shortfat_s4_ff15 896 896 20 14 2 1.5 4 cycle
probe_one probe_balanced_s4_e1472_ff175 768 1472 24 12 4 1.75 4 cycle
probe_one probe_tall_s4_e832_ff2125 768 832 32 12 2 2.125 4 cycle

# immediate over-cap challengers
probe_one probe_wide_s4_e448_ff10 1024 448 16 16 2 1.0 4 cycle
probe_one probe_shortfat_s5_ff1125 896 896 20 14 2 1.125 5 cycle
probe_one probe_balanced_s4_e1536_ff175 768 1536 24 12 4 1.75 4 cycle
probe_one probe_tall_s4_e896_ff2125 768 896 32 12 2 2.125 4 cycle
