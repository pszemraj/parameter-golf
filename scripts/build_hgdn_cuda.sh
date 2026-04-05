#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${MAX_JOBS:=}"
export MAX_JOBS

python setup_hgdn_cuda.py build_ext --inplace
