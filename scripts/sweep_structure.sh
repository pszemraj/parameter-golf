#!/usr/bin/env bash
set -euo pipefail

# Structural sweep launcher for the frozen Core/Amplifier spec.
#
# This answers questions like:
#   - Are the amplifier blocks earning their bytes?
#   - How many lag-derived branches are actually needed?
#   - Does a low-rank readout keep quality while saving artifact bytes?
#
# Each run line is:
#   name branch_lags num_blocks readout_rank
#
# Global controller/training hyperparameters are shared via env vars.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

bool_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  echo "+ $*"
  if bool_true "${DRY_RUN}"; then
    return 0
  fi
  "$@"
}

now_stamp() { date +%Y%m%d_%H%M%S; }

PRESET="${PRESET:-cpu_structure}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_CONVERT_PARQUET="${AUTO_CONVERT_PARQUET:-1}"
RUN_FILTER="${RUN_FILTER:-}"
SKIP_DONE="${SKIP_DONE:-1}"
BLAS_THREADS="${BLAS_THREADS:-1}"

DEFAULT_DATA_PATH="${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
DATA_PATH="${DATA_PATH:-$DEFAULT_DATA_PATH}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/experiments/structure_sweeps/$(now_stamp)_${PRESET}}"
SUMMARY_TSV="${SUMMARY_TSV:-${MODEL_ROOT}/summary.tsv}"
COMMANDS_TXT="${COMMANDS_TXT:-${MODEL_ROOT}/commands.txt}"

# Shared frozen-spec config unless overridden per run via RUN_SPECS
STORAGE_DTYPE="${STORAGE_DTYPE:-uint16}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
CORE_DIM="${CORE_DIM:-16}"
FIXED_DTYPE="${FIXED_DTYPE:-float16}"
EMBEDDING_INIT="${EMBEDDING_INIT:-svd}"
SPECTRAL_NEIGHBORS="${SPECTRAL_NEIGHBORS:-64}"
LAG_IDENTITY_BASE="${LAG_IDENTITY_BASE:-0.15}"
SPEC_STRATEGY="${SPEC_STRATEGY:-stream}"
SPEC_WORKERS="${SPEC_WORKERS:--1}"
SPEC_MAX_TOKENS="${SPEC_MAX_TOKENS:-500000}"

# Shared controller/training config
CORE_LAYERS="${CORE_LAYERS:-5}"
CORE_EXPANSION="${CORE_EXPANSION:-2.0}"
RESIDUAL_CORE="${RESIDUAL_CORE:-1}"
RESIDUAL_CORE_INIT="${RESIDUAL_CORE_INIT:--2.0}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_STEPS="${NUM_STEPS:-80}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
MIN_LR="${MIN_LR:-0.0003}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
LR_HOLD_STEPS="${LR_HOLD_STEPS:-20}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
HARD_LOSS_GAMMA="${HARD_LOSS_GAMMA:-0.5}"
HARD_LOSS_CAP="${HARD_LOSS_CAP:-5.0}"
CARRY_CHUNKS="${CARRY_CHUNKS:-16}"
BPTT_CHUNKS="${BPTT_CHUNKS:-2}"
VAL_EVERY="${VAL_EVERY:-20}"
VAL_STEPS="${VAL_STEPS:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
DATA_MAX_TOKENS="${DATA_MAX_TOKENS:-500000}"
FORCE_DEVICE="${FORCE_DEVICE:-cpu}"
NO_MMAP="${NO_MMAP:-1}"
COMPILE="${COMPILE:-0}"

if [[ -z "${RUN_SPECS:-}" ]]; then
  case "$PRESET" in
    cpu_structure)
      RUN_SPECS="$(cat <<'EOF'
blocks0      1,2,3,4,6,8,12,16,24,32,48,64                    0   0
blocks3      1,2,3,4,6,8,12,16,24,32,48,64                    3   0
blocks6      1,2,3,4,6,8,12,16,24,32,48,64                    6   0
blocks9      1,2,3,4,6,8,12,16,24,32,48,64                    9   0
branches8    1,2,4,8,16,32,64,128                            9   0
readout128   1,2,3,4,6,8,12,16,24,32,48,64                    9 128
EOF
)"
      ;;
    *)
      echo "ERROR: unknown PRESET='$PRESET' and RUN_SPECS not provided" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$MODEL_ROOT"

if [[ "$DATA_PATH" == *.parquet ]]; then
  if bool_true "$AUTO_CONVERT_PARQUET"; then
    mkdir -p "$MODEL_ROOT/data"
    PARQUET_OUT="${MODEL_ROOT}/data/$(basename "${DATA_PATH%.parquet}").${STORAGE_DTYPE}.bin"
    if [[ ! -f "$PARQUET_OUT" ]]; then
      run_cmd "$PYTHON_BIN" "$REPO_ROOT/tools/parquet_tokens_to_bin.py" "$DATA_PATH" "$PARQUET_OUT" --dtype "$STORAGE_DTYPE"
    fi
    DATA_PATH="$PARQUET_OUT"
  else
    echo "ERROR: DATA_PATH points to a parquet file and AUTO_CONVERT_PARQUET=0" >&2
    exit 1
  fi
fi

append_summary() {
  "$PYTHON_BIN" - "$SUMMARY_TSV" "$1" "$2" "$3" "$4" "$5" "$6" "$7" <<'PY'
import json, math, pathlib, sys
summary_path = pathlib.Path(sys.argv[1])
run_name, run_dir, branch_lags, num_blocks, readout_rank, status, log_path = sys.argv[2:9]
metrics_path = pathlib.Path(run_dir) / "metrics.jsonl"
rows = []
if metrics_path.exists():
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
best = {}
last = {}
if rows:
    def _loss(row):
        v = row.get("val_loss")
        try:
            return float(v)
        except Exception:
            return math.inf
    best = min(rows, key=_loss)
    last = rows[-1]
header = [
    "run_name", "status", "branch_lags", "num_blocks", "readout_rank",
    "best_step", "best_val_loss", "best_val_bpb", "last_step", "last_val_loss", "last_val_bpb",
    "run_dir", "log_path"
]
row = [
    run_name, status, branch_lags, num_blocks, readout_rank,
    str(best.get("step", "")), str(best.get("val_loss", "")), str(best.get("val_bpb", "")),
    str(last.get("step", "")), str(last.get("val_loss", "")), str(last.get("val_bpb", "")),
    run_dir, log_path,
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
write_header = (not summary_path.exists()) or summary_path.stat().st_size == 0
with summary_path.open("a", encoding="utf-8") as f:
    if write_header:
        f.write("\t".join(header) + "\n")
    f.write("\t".join(row) + "\n")
PY
}

while read -r RUN_NAME BRANCH_LAGS NUM_BLOCKS READOUT_RANK; do
  [[ -z "${RUN_NAME:-}" ]] && continue
  [[ "${RUN_NAME:0:1}" == "#" ]] && continue
  if [[ -n "$RUN_FILTER" && "$RUN_NAME" != *"$RUN_FILTER"* ]]; then
    continue
  fi

  RUN_DIR="${MODEL_ROOT}/${RUN_NAME}"
  LOG_PATH="${MODEL_ROOT}/${RUN_NAME}.log"
  if [[ -f "$RUN_DIR/metrics.jsonl" && "$SKIP_DONE" == "1" ]]; then
    echo "Skipping existing run: $RUN_NAME"
    append_summary "$RUN_NAME" "$RUN_DIR" "$BRANCH_LAGS" "$NUM_BLOCKS" "$READOUT_RANK" "skipped" "$LOG_PATH"
    continue
  fi

  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"

  INIT_CMD=(
    "$PYTHON_BIN" "$REPO_ROOT/inspect_model.py" init "$RUN_DIR"
    --data "$DATA_PATH"
    --storage-dtype "$STORAGE_DTYPE"
    --vocab-size "$VOCAB_SIZE"
    --core-dim "$CORE_DIM"
    --branch-lags "$BRANCH_LAGS"
    --num-blocks "$NUM_BLOCKS"
    --fixed-dtype "$FIXED_DTYPE"
    --embedding-init "$EMBEDDING_INIT"
    --spectral-neighbors "$SPECTRAL_NEIGHBORS"
    --lag-identity-base "$LAG_IDENTITY_BASE"
    --spec-strategy "$SPEC_STRATEGY"
    --spec-workers "$SPEC_WORKERS"
    --core-layers "$CORE_LAYERS"
    --core-expansion "$CORE_EXPANSION"
    --residual-core "$RESIDUAL_CORE"
    --residual-core-init "$RESIDUAL_CORE_INIT"
  )
  if [[ -n "$SPEC_MAX_TOKENS" ]]; then
    INIT_CMD+=(--max-tokens "$SPEC_MAX_TOKENS")
  fi
  if [[ "$READOUT_RANK" != "0" ]]; then
    INIT_CMD+=(--readout-rank "$READOUT_RANK")
  fi

  TRAIN_CMD=(
    "$PYTHON_BIN" "$REPO_ROOT/train_core_amplifier.py" "$RUN_DIR"
    --data "$DATA_PATH"
    --storage-dtype "$STORAGE_DTYPE"
    --seq-len "$SEQ_LEN"
    --batch-size "$BATCH_SIZE"
    --num-steps "$NUM_STEPS"
    --learning-rate "$LEARNING_RATE"
    --lr-schedule cosine
    --warmup-steps "$WARMUP_STEPS"
    --lr-hold-steps "$LR_HOLD_STEPS"
    --min-lr "$MIN_LR"
    --weight-decay "$WEIGHT_DECAY"
    --hard-loss-gamma "$HARD_LOSS_GAMMA"
    --hard-loss-cap "$HARD_LOSS_CAP"
    --carry-chunks "$CARRY_CHUNKS"
    --bptt-chunks "$BPTT_CHUNKS"
    --core-layers "$CORE_LAYERS"
    --core-expansion "$CORE_EXPANSION"
    --residual-core "$RESIDUAL_CORE"
    --residual-core-init "$RESIDUAL_CORE_INIT"
    --val-every "$VAL_EVERY"
    --val-steps "$VAL_STEPS"
    --log-every "$LOG_EVERY"
    --save-every "$SAVE_EVERY"
    --data-max-tokens "$DATA_MAX_TOKENS"
    --force-device "$FORCE_DEVICE"
  )
  if bool_true "$NO_MMAP"; then
    TRAIN_CMD+=(--no-mmap)
  fi
  if bool_true "$COMPILE"; then
    TRAIN_CMD+=(--compile)
  fi

  printf '%q ' "${INIT_CMD[@]}" >> "$COMMANDS_TXT"; echo >> "$COMMANDS_TXT"
  printf '%q ' "${TRAIN_CMD[@]}" >> "$COMMANDS_TXT"; echo >> "$COMMANDS_TXT"

  {
    echo "=== [$RUN_NAME] init ==="
    OMP_NUM_THREADS="$BLAS_THREADS" MKL_NUM_THREADS="$BLAS_THREADS" OPENBLAS_NUM_THREADS="$BLAS_THREADS" "${INIT_CMD[@]}"
    echo
    echo "=== [$RUN_NAME] train ==="
    OMP_NUM_THREADS="$BLAS_THREADS" MKL_NUM_THREADS="$BLAS_THREADS" OPENBLAS_NUM_THREADS="$BLAS_THREADS" "${TRAIN_CMD[@]}"
  } | tee "$LOG_PATH"

  append_summary "$RUN_NAME" "$RUN_DIR" "$BRANCH_LAGS" "$NUM_BLOCKS" "$READOUT_RANK" "done" "$LOG_PATH"

done <<< "$RUN_SPECS"

if [[ -x "$REPO_ROOT/tools/rebuild_summary.py" ]] && ! bool_true "$DRY_RUN"; then
  echo "Rebuilding clean summary from completed run dirs ..."
  "$PYTHON_BIN" "$REPO_ROOT/tools/rebuild_summary.py" "$MODEL_ROOT" --out "$SUMMARY_TSV" >/dev/null
fi

echo
echo "Summary: $SUMMARY_TSV"
