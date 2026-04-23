#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUN_VERSION="${RUN_VERSION:-v2}"
FINALIST_SEEDS="${FINALIST_SEEDS:-1337 2027 3141}"
SIDECAR_SEEDS="${SIDECAR_SEEDS:-1337 2027}"
RUN_GATE_LR_SIDECAR_AFTER_FINALISTS="${RUN_GATE_LR_SIDECAR_AFTER_FINALISTS:-0}"

export FINALIST_SPECS="${FINALIST_SPECS:-$'blocks1_resid10_e12_lr0035_final '"${REPO_ROOT}"$'/experiments/5090_schedule/blocks1_hold_confirm1b_v1/blocks1_resid10_e12_h7000_1b 10 12.0 none current none 0.0035\nblocks0_resid12_e10_lr0035_final '"${REPO_ROOT}"$'/experiments/5090_schedule/blocks0_12x10_hold_confirm1b_v1/blocks0_resid12_e10_h7000_1b 12 10.0 none current none 0.0035'}"
export RUN_VERSION

echo "5090 post-temporal queue"
echo "repo_root=${REPO_ROOT}"
echo "run_version=${RUN_VERSION}"
echo "finalist_seeds=${FINALIST_SEEDS}"
echo "sidecar_seeds=${SIDECAR_SEEDS}"
echo "run_gate_lr_sidecar_after_finalists=${RUN_GATE_LR_SIDECAR_AFTER_FINALISTS}"
echo "finalist_specs:"
printf '%s\n' "${FINALIST_SPECS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
fi

SEEDS="${FINALIST_SEEDS}" bash "${SCRIPT_DIR}/run_5090_finalist_confirm1b.sh"

if [[ "${RUN_GATE_LR_SIDECAR_AFTER_FINALISTS}" == "1" ]]; then
  echo
  echo "running gate x lr sidecar after finalists"
  SEEDS="${SIDECAR_SEEDS}" bash "${SCRIPT_DIR}/run_5090_gate_lr_sidecar.sh"
fi
