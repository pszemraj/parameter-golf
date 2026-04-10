#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

: "${MAX_JOBS:=}"
export MAX_JOBS

"${PYTHON_BIN:-python}" setup_hgdn_cuda.py build_ext --inplace
