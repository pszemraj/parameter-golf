#!/bin/bash
# Called by wandb sweep agent. Env vars are set by wandb from sweep_config.yaml.
# This script just forwards them to the training script via torchrun.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

NGPU="${NGPU:-1}"
export RUN_ID="${RUN_ID:-sweep_${WANDB_RUN_ID:-$(date +%s)}}"
export WANDB_PROJECT="${WANDB_PROJECT:-param-golf-hybrid}"
export DATA_PATH="${DATA_PATH:-$repo_root/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$repo_root/data/tokenizers/fineweb_1024_bpe.model}"
export USE_WANDB="${USE_WANDB:-1}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

exec torchrun --standalone --nproc_per_node="$NGPU" "$repo_root/train_gpt_hybrid.py"
