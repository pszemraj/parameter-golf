#!/bin/bash
# Called by wandb sweep agent. Env vars are set by wandb from sweep_config.yaml.
# This script just forwards them to the training script via torchrun.
set -euo pipefail

NGPU="${NGPU:-1}"
RUN_ID="sweep_${WANDB_RUN_ID:-$(date +%s)}"

exec torchrun --standalone --nproc_per_node="$NGPU" train_gpt_hybrid.py
