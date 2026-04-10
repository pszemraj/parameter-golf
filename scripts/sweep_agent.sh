#!/bin/bash
# Called by wandb sweep agent. Env vars are set by wandb from sweep_config.yaml.
# This script just forwards them to the training script via torchrun.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

NGPU="${NGPU:-1}"
export RUN_ID="${RUN_ID:-sweep_${WANDB_RUN_ID:-$(date +%s)}}"
export WANDB_PROJECT="${WANDB_PROJECT:-pg-hgdn-ablations}"
export DATA_PATH="${DATA_PATH:-$HGDN_REPO_ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$HGDN_REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export USE_WANDB="${USE_WANDB:-1}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

exec torchrun --standalone --nproc_per_node="$NGPU" "$HGDN_REPO_ROOT/train_gpt_hybrid.py"
