#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SEEDS="${SEEDS:-1337 2027 3141}"

export FINALIST_SPECS="${FINALIST_SPECS:-$'blocks1_resid10_e12_lr0035_final '"${REPO_ROOT}"$'/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b 10 12.0 none current none 0.0035\nblocks0_resid12_e10_lr0035_final '"${REPO_ROOT}"$'/experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1/blocks0_resid12_e10_h7000_1b 12 10.0 none current none 0.0035'}"
export SEEDS

echo "5090 post-temporal queue"
echo "repo_root=${REPO_ROOT}"
echo "seeds=${SEEDS}"
echo "finalist_specs:"
printf '%s\n' "${FINALIST_SPECS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
fi

exec bash "${SCRIPT_DIR}/run_5090_finalist_confirm1b.sh"
