#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

usage() {
    cat <<'EOF'
Usage: scripts/run_hgdn_cuda_preflight.sh

Purpose:
  Run the tiny single-process CUDA HGDN preflight suite. This catches FLA and
  compile regressions before longer H100 or sweep commands.

Defaults:
  - USE_WANDB=0
  - WANDB_MODE=offline
  - PYTHON_BIN=python
  - SEED=1337
  - PYTHONHASHSEED=1337
  - CUDNN_BENCHMARK=0

Example:
  scripts/run_hgdn_cuda_preflight.sh
  PYTHON_BIN=/home/pszemraj/miniforge3/envs/pg/bin/python scripts/run_hgdn_cuda_preflight.sh
EOF
}

case "${1:-run}" in
help|-h|--help)
    usage
    exit 0
    ;;
run)
    ;;
*)
    echo "Unknown mode: $1" >&2
    usage >&2
    exit 1
    ;;
esac

export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-1337}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
"${PYTHON_BIN:-python}" "$HGDN_REPO_ROOT/scripts/hgdn_cuda_preflight.py"
