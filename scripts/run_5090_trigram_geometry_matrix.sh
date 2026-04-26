#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_VERSION="${RUN_VERSION:-geom1}"
SEEDS="${SEEDS:-1337}"
DRY_RUN="${DRY_RUN:-0}"
TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"
TRIGRAM_COUNT_WORKERS="${TRIGRAM_COUNT_WORKERS:-1}"
INCLUDE_EXTENDED_GEOMETRY="${INCLUDE_EXTENDED_GEOMETRY:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
GEOMETRY_NUM_STEPS="${GEOMETRY_NUM_STEPS:-}"
GEOMETRY_LR_HOLD_STEPS="${GEOMETRY_LR_HOLD_STEPS:-}"
GEOMETRY_BATCH_SIZE="${GEOMETRY_BATCH_SIZE:-}"
GEOMETRY_SEQ_LEN="${GEOMETRY_SEQ_LEN:-}"
TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-}"
VAL_EVERY="${VAL_EVERY:-}"
VAL_STEPS="${VAL_STEPS:-}"
LOG_EVERY="${LOG_EVERY:-}"
LOG_STATE_EVERY="${LOG_STATE_EVERY:-}"
SAVE_EVERY="${SAVE_EVERY:-}"
TRIGRAM_MAX_TOKENS="${TRIGRAM_MAX_TOKENS:-}"
DATA_MAX_TOKENS="${DATA_MAX_TOKENS:-}"
GEOMETRY_TRAIN_LABEL="${GEOMETRY_TRAIN_LABEL:-}"
WANDB="${WANDB:-}"

read -r -d '' DEFAULT_GEOMETRY_SPECS <<'EOF' || true
blocks0_d96_l6_i512 96 6 512
blocks0_d64_l10_i512 64 10 512
blocks0_d128_l4_i512 128 4 512
blocks0_d128_l5_i512 128 5 512
EOF

read -r -d '' EXTENDED_GEOMETRY_SPECS <<'EOF' || true
blocks0_d64_l8_i512 64 8 512
blocks0_d96_l8_i512 96 8 512
blocks0_d128_l6_i384 128 6 384
blocks0_d160_l4_i512 160 4 512
EOF

GEOMETRY_SPECS="${GEOMETRY_SPECS:-${DEFAULT_GEOMETRY_SPECS}}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --run-version VALUE
  --seeds VALUE
  --trigram-top-k VALUE
  --count-workers VALUE
  --geometry-specs VALUE
  --num-steps VALUE
  --lr-hold-steps VALUE
  --batch-size VALUE
  --seq-len VALUE
  --target-effective-step-tokens VALUE
  --val-every VALUE
  --val-steps VALUE
  --log-every VALUE
  --log-state-every VALUE
  --save-every VALUE
  --trigram-max-tokens VALUE
  --data-max-tokens VALUE
  --geometry-train-label VALUE
  --no-wandb
  --smoke-test
  --include-extended-geometry
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-version) RUN_VERSION="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --trigram-top-k) TRIGRAM_TOP_K="$2"; shift 2 ;;
    --trigram-count-workers|--count-workers) TRIGRAM_COUNT_WORKERS="$2"; shift 2 ;;
    --geometry-specs) GEOMETRY_SPECS="$2"; shift 2 ;;
    --num-steps|--geometry-num-steps) GEOMETRY_NUM_STEPS="$2"; shift 2 ;;
    --lr-hold-steps|--geometry-lr-hold-steps) GEOMETRY_LR_HOLD_STEPS="$2"; shift 2 ;;
    --batch-size|--geometry-batch-size) GEOMETRY_BATCH_SIZE="$2"; shift 2 ;;
    --seq-len|--geometry-seq-len) GEOMETRY_SEQ_LEN="$2"; shift 2 ;;
    --target-effective-step-tokens) TARGET_EFFECTIVE_STEP_TOKENS="$2"; shift 2 ;;
    --val-every) VAL_EVERY="$2"; shift 2 ;;
    --val-steps) VAL_STEPS="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --log-state-every) LOG_STATE_EVERY="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --trigram-max-tokens) TRIGRAM_MAX_TOKENS="$2"; shift 2 ;;
    --data-max-tokens) DATA_MAX_TOKENS="$2"; shift 2 ;;
    --geometry-train-label) GEOMETRY_TRAIN_LABEL="$2"; shift 2 ;;
    --no-wandb) WANDB=0; shift ;;
    --smoke-test) SMOKE_TEST=1; shift ;;
    --include-extended-geometry) INCLUDE_EXTENDED_GEOMETRY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${INCLUDE_EXTENDED_GEOMETRY}" == "1" ]]; then
  GEOMETRY_SPECS="${GEOMETRY_SPECS}"$'\n'"${EXTENDED_GEOMETRY_SPECS}"
fi

echo "5090 trigram geometry matrix"
echo "run_version=${RUN_VERSION}"
echo "seeds=${SEEDS}"
echo "spec_columns=label core_dim core_layers inner_dim"
echo "smoke_test=${SMOKE_TEST}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry_run=1"
fi

append_if_set() {
  local flag="$1"
  local value="$2"
  if [[ -n "${value:-}" ]]; then
    cmd+=("${flag}" "${value}")
  fi
}

while read -r label core_dim core_layers core_inner_dim extra; do
  if [[ -z "${label:-}" || "${label:0:1}" == "#" ]]; then
    continue
  fi
  if [[ -n "${extra:-}" ]]; then
    echo "Invalid GEOMETRY_SPECS row: ${label} ${core_dim} ${core_layers} ${core_inner_dim} ${extra}" >&2
    exit 1
  fi
  echo
  echo "=== geometry ${label}: core_dim=${core_dim} layers=${core_layers} inner_dim=${core_inner_dim} ==="
  cmd=(
    bash "${SCRIPT_DIR}/run_5090_trigram_aligned_geometry_screen.sh"
    --run-version "${RUN_VERSION}"
    --seeds "${SEEDS}"
    --geometry-label "${label}"
    --geometry-core-dim "${core_dim}"
    --geometry-core-layers "${core_layers}"
    --geometry-core-inner-dim "${core_inner_dim}"
    --geometry-core-expansion ""
    --geometry-num-blocks "0"
    --trigram-top-k "${TRIGRAM_TOP_K}"
    --count-workers "${TRIGRAM_COUNT_WORKERS}"
  )
  append_if_set "--num-steps" "${GEOMETRY_NUM_STEPS}"
  append_if_set "--lr-hold-steps" "${GEOMETRY_LR_HOLD_STEPS}"
  append_if_set "--batch-size" "${GEOMETRY_BATCH_SIZE}"
  append_if_set "--seq-len" "${GEOMETRY_SEQ_LEN}"
  append_if_set "--target-effective-step-tokens" "${TARGET_EFFECTIVE_STEP_TOKENS}"
  append_if_set "--val-every" "${VAL_EVERY}"
  append_if_set "--val-steps" "${VAL_STEPS}"
  append_if_set "--log-every" "${LOG_EVERY}"
  append_if_set "--log-state-every" "${LOG_STATE_EVERY}"
  append_if_set "--save-every" "${SAVE_EVERY}"
  append_if_set "--trigram-max-tokens" "${TRIGRAM_MAX_TOKENS}"
  append_if_set "--data-max-tokens" "${DATA_MAX_TOKENS}"
  append_if_set "--geometry-train-label" "${GEOMETRY_TRAIN_LABEL}"
  if [[ "${WANDB:-}" == "0" ]]; then
    cmd+=(--no-wandb)
  fi
  if [[ "${SMOKE_TEST}" == "1" ]]; then
    cmd+=(--smoke-test)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  "${cmd[@]}"
done <<<"${GEOMETRY_SPECS}"
