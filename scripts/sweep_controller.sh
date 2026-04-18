#!/usr/bin/env bash
set -euo pipefail

# Controller-only sweep launcher for the Core/Amplifier LM.
#
# Default behavior:
#   - build one shared frozen spec
#   - clone that spec into one run directory per controller setting
#   - train each run sequentially
#   - append a compact TSV summary
#
# Typical usage:
#   bash scripts/sweep_controller.sh
#
# GPU controller sweep with the recommended default grid:
#   DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
#   MODEL_ROOT=experiments/sweeps/controller_main \
#   bash scripts/sweep_controller.sh
#
# CPU smoke on the uploaded FineWeb parquet sample:
#   PRESET=cpu_smoke \
#   DATA_PATH=/path/to/fineweb_sample.parquet \
#   FORCE_DEVICE=cpu COMPILE=0 \
#   MODEL_ROOT=experiments/sweeps/cpu_smoke \
#   bash scripts/sweep_controller.sh
#
# Custom run grid. Each non-empty, non-comment line is:
#   name core_layers core_expansion carry_chunks bptt_chunks residual_core
#        residual_core_init learning_rate lr_hold_steps min_lr
#        num_steps batch_size seq_len
#
# Example:
#   RUN_SPECS=$'d5_e20 5 2.0 16 2 1 -2.0 0.003 1500 0.0003 7000 256 512\n\
# d5_e30 5 3.0 16 2 1 -2.0 0.003 1500 0.0003 7000 256 512' \
#   bash scripts/sweep_controller.sh

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

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path not found: $path" >&2
    exit 1
  fi
}

now_stamp() {
  date +%Y%m%d_%H%M%S
}

PRESET="${PRESET:-controller_default}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
REBUILD_SHARED="${REBUILD_SHARED:-0}"
AUTO_CONVERT_PARQUET="${AUTO_CONVERT_PARQUET:-1}"
RUN_FILTER="${RUN_FILTER:-}"
USER_SET_COMPILE="${COMPILE+x}"

DEFAULT_DATA_PATH="${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
DATA_PATH="${DATA_PATH:-$DEFAULT_DATA_PATH}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/experiments/sweeps/$(now_stamp)_${PRESET}}"
SHARED_SPEC_DIR="${SHARED_SPEC_DIR:-${MODEL_ROOT}/_shared_spec}"
SUMMARY_TSV="${SUMMARY_TSV:-${MODEL_ROOT}/summary.tsv}"
COMMANDS_TXT="${COMMANDS_TXT:-${MODEL_ROOT}/commands.txt}"

# Frozen-spec settings (shared across the sweep)
STORAGE_DTYPE="${STORAGE_DTYPE:-uint16}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
CORE_DIM="${CORE_DIM:-48}"
BRANCH_LAGS="${BRANCH_LAGS:-1,2,3,4,6,8,12,16,24,32,48,64}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
FIXED_DTYPE="${FIXED_DTYPE:-bfloat16}"
EMBEDDING_INIT="${EMBEDDING_INIT:-spectral}"
SPECTRAL_NEIGHBORS="${SPECTRAL_NEIGHBORS:-64}"
LAG_IDENTITY_BASE="${LAG_IDENTITY_BASE:-0.15}"
SPEC_STRATEGY="${SPEC_STRATEGY:-auto}"
SPEC_WORKERS="${SPEC_WORKERS:--1}"
SPEC_MAX_TOKENS="${SPEC_MAX_TOKENS:-}"

# Global training defaults (can be overridden per run via RUN_SPECS columns)
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
HARD_LOSS_GAMMA="${HARD_LOSS_GAMMA:-0.5}"
HARD_LOSS_CAP="${HARD_LOSS_CAP:-5.0}"
VAL_EVERY="${VAL_EVERY:-200}"
VAL_STEPS="${VAL_STEPS:-20}"
LOG_EVERY="${LOG_EVERY:-20}"
LOG_STATE_EVERY="${LOG_STATE_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DATA_MAX_TOKENS="${DATA_MAX_TOKENS:-}"
TRAIN_FRAC="${TRAIN_FRAC:-0.9}"
NO_MMAP="${NO_MMAP:-0}"
TOKENS_ON_DEVICE="${TOKENS_ON_DEVICE:-0}"
FORCE_DEVICE="${FORCE_DEVICE:-}"
NO_AUTOCAST="${NO_AUTOCAST:-0}"
COMPILE="${COMPILE:-1}"
COMPILE_AFTER="${COMPILE_AFTER:-200}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
COMPILE_BASE_PATH="${COMPILE_BASE_PATH:-1}"

if [[ "${PRESET}" == "cpu_smoke" && -z "$USER_SET_COMPILE" ]]; then
  COMPILE=0
fi

if [[ -z "${RUN_SPECS:-}" ]]; then
  case "$PRESET" in
    controller_default)
      RUN_SPECS="$(cat <<'EOF'
d5_e20   5 2.0 16 2 1 -2.0 0.003 1500 0.0003 7000 256 512
d4_e25   4 2.5 16 2 1 -2.0 0.003 1500 0.0003 7000 256 512
d5_e30   5 3.0 16 2 1 -2.0 0.003 1500 0.0003 7000 256 512
d6_e25   6 2.5 16 2 1 -2.0 0.003 1500 0.0003 7000 256 512
EOF
)"
      ;;
    cpu_smoke)
      RUN_SPECS="$(cat <<'EOF'
plain3_e20         3 2.0  8 1 0 -2.0 0.003 20 0.0003  80  8 128
resid5_e20         5 2.0 16 1 1 -2.0 0.003 20 0.0003  80  8 128
resid5_e20_tbptt2  5 2.0 16 2 1 -2.0 0.003 20 0.0003  80  8 128
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

ORIGINAL_DATA_PATH="$DATA_PATH"
if [[ "$DATA_PATH" == *.parquet ]]; then
  if bool_true "$AUTO_CONVERT_PARQUET"; then
    mkdir -p "$MODEL_ROOT/data"
    PARQUET_OUT="${MODEL_ROOT}/data/$(basename "${DATA_PATH%.parquet}").uint16.bin"
    if [[ ! -f "$PARQUET_OUT" ]]; then
      run_cmd "$PYTHON_BIN" "$REPO_ROOT/tools/parquet_tokens_to_bin.py" "$DATA_PATH" "$PARQUET_OUT" --dtype "$STORAGE_DTYPE"
    else
      echo "Using existing converted parquet token file: $PARQUET_OUT"
    fi
    DATA_PATH="$PARQUET_OUT"
  else
    echo "ERROR: DATA_PATH points to a parquet file and AUTO_CONVERT_PARQUET=0" >&2
    exit 1
  fi
fi

require_file "$REPO_ROOT/inspect_model.py"
require_file "$REPO_ROOT/train_core_amplifier.py"
if bool_true "$DRY_RUN"; then
  require_file "$ORIGINAL_DATA_PATH"
else
  require_file "$DATA_PATH"
fi

append_summary() {
  local run_name="$1"
  local run_dir="$2"
  local log_path="$3"
  local status="$4"
  local core_layers="$5"
  local core_expansion="$6"
  local carry_chunks="$7"
  local bptt_chunks="$8"
  local residual_core="$9"
  local residual_core_init="${10}"
  local learning_rate="${11}"
  local lr_hold_steps="${12}"
  local min_lr="${13}"
  local num_steps="${14}"
  local batch_size="${15}"
  local seq_len="${16}"

  "$PYTHON_BIN" - "$SUMMARY_TSV" "$run_name" "$run_dir" "$log_path" "$status" \
    "$core_layers" "$core_expansion" "$carry_chunks" "$bptt_chunks" "$residual_core" "$residual_core_init" \
    "$learning_rate" "$lr_hold_steps" "$min_lr" "$num_steps" "$batch_size" "$seq_len" <<'PY'
import json, math, pathlib, sys
summary_path = pathlib.Path(sys.argv[1])
run_name, run_dir, log_path, status = sys.argv[2:6]
fields = sys.argv[6:]
(core_layers, core_expansion, carry_chunks, bptt_chunks, residual_core, residual_core_init,
 learning_rate, lr_hold_steps, min_lr, num_steps, batch_size, seq_len) = fields
metrics_path = pathlib.Path(run_dir) / "metrics.jsonl"
rows = []
if metrics_path.exists():
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
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
    def _loss(x):
        v = x.get("val_loss")
        try:
            return float(v)
        except Exception:
            return math.inf
    best = min(rows, key=_loss)
    last = rows[-1]
header = [
    "run_name", "status", "core_layers", "core_expansion", "carry_chunks", "bptt_chunks",
    "residual_core", "residual_core_init", "learning_rate", "lr_hold_steps", "min_lr",
    "num_steps", "batch_size", "seq_len", "best_step", "best_val_loss", "best_val_bpb",
    "last_step", "last_val_loss", "last_val_bpb", "run_dir", "log_path"
]
row = [
    run_name, status, core_layers, core_expansion, carry_chunks, bptt_chunks,
    residual_core, residual_core_init, learning_rate, lr_hold_steps, min_lr,
    num_steps, batch_size, seq_len,
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

build_shared_spec() {
  if [[ -f "$SHARED_SPEC_DIR/spec.pt" && -f "$SHARED_SPEC_DIR/config.json" ]] && ! bool_true "$REBUILD_SHARED"; then
    echo "Using existing shared spec: $SHARED_SPEC_DIR"
    return 0
  fi

  mkdir -p "$SHARED_SPEC_DIR"
  local cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/inspect_model.py" init "$SHARED_SPEC_DIR"
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
    --core-layers 5
    --core-expansion 2.0
    --residual-core 1
    --residual-core-init -2.0
  )
  if [[ -n "$SPEC_MAX_TOKENS" ]]; then
    cmd+=(--max-tokens "$SPEC_MAX_TOKENS")
  fi
  printf '%q ' "${cmd[@]}" >> "$COMMANDS_TXT"
  printf '\n' >> "$COMMANDS_TXT"
  run_cmd "${cmd[@]}"
}

prepare_run_dir() {
  local run_dir="$1"
  local run_name="$2"
  local core_layers="$3"
  local core_expansion="$4"
  local carry_chunks="$5"
  local bptt_chunks="$6"
  local residual_core="$7"
  local residual_core_init="$8"
  local learning_rate="$9"
  local lr_hold_steps="${10}"
  local min_lr="${11}"
  local num_steps="${12}"
  local batch_size="${13}"
  local seq_len="${14}"

  mkdir -p "$run_dir"
  if bool_true "$DRY_RUN"; then
    return 0
  fi
  cp "$SHARED_SPEC_DIR/spec.pt" "$run_dir/spec.pt"
  cp "$SHARED_SPEC_DIR/config.json" "$run_dir/config.json"
  shopt -s nullglob
  local tok
  for tok in "$SHARED_SPEC_DIR"/*.model "$SHARED_SPEC_DIR"/*.vocab; do
    cp "$tok" "$run_dir/"
  done
  shopt -u nullglob

  "$PYTHON_BIN" - "$run_dir/config.json" \
    "$run_name" "$DATA_PATH" "$STORAGE_DTYPE" "$core_layers" "$core_expansion" "$residual_core" "$residual_core_init" \
    "$seq_len" "$batch_size" "$num_steps" "$learning_rate" "$LR_SCHEDULE" "$min_lr" "$WARMUP_STEPS" "$lr_hold_steps" \
    "$WEIGHT_DECAY" "$GRAD_CLIP" "$HARD_LOSS_GAMMA" "$HARD_LOSS_CAP" "$carry_chunks" "$bptt_chunks" "$VAL_EVERY" "$VAL_STEPS" \
    "$LOG_EVERY" "$LOG_STATE_EVERY" "$SAVE_EVERY" "$GRAD_ACCUM" "$TRAIN_FRAC" "$DATA_MAX_TOKENS" <<'PY'
import json, pathlib, sys
cfg_path = pathlib.Path(sys.argv[1])
(
    run_name, data_path, storage_dtype, core_layers, core_expansion, residual_core, residual_core_init,
    seq_len, batch_size, num_steps, learning_rate, lr_schedule, min_lr, warmup_steps, lr_hold_steps,
    weight_decay, grad_clip, hard_loss_gamma, hard_loss_cap, carry_chunks, bptt_chunks, val_every, val_steps,
    log_every, log_state_every, save_every, grad_accum, train_frac, data_max_tokens
) = sys.argv[2:]
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg.setdefault("meta", {})["run_name"] = run_name
cfg.setdefault("data", {})["source"] = data_path
cfg["data"]["storage_dtype"] = storage_dtype
cfg.setdefault("model", {})["core_layers"] = int(core_layers)
cfg["model"]["core_expansion"] = float(core_expansion)
cfg["model"]["residual_core"] = bool(int(residual_core))
cfg["model"]["residual_core_init"] = float(residual_core_init)
tr = cfg.setdefault("training", {})
tr["seq_len"] = int(seq_len)
tr["batch_size"] = int(batch_size)
tr["grad_accum"] = int(grad_accum)
tr["carry_chunks"] = int(carry_chunks)
tr["bptt_chunks"] = int(bptt_chunks)
tr["num_steps"] = int(num_steps)
tr["learning_rate"] = float(learning_rate)
tr["lr_schedule"] = lr_schedule
tr["min_lr"] = float(min_lr)
tr["warmup_steps"] = int(warmup_steps)
tr["lr_hold_steps"] = int(lr_hold_steps)
tr["weight_decay"] = float(weight_decay)
tr["grad_clip"] = float(grad_clip)
tr["hard_loss_gamma"] = float(hard_loss_gamma)
tr["hard_loss_cap"] = float(hard_loss_cap)
tr["val_every"] = int(val_every)
tr["val_steps"] = int(val_steps)
tr["log_every"] = int(log_every)
tr["log_state_every"] = int(log_state_every)
tr["save_every"] = int(save_every)
tr["train_frac"] = float(train_frac)
if data_max_tokens:
    tr["data_max_tokens"] = int(data_max_tokens)
with cfg_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
}

build_shared_spec

echo "Sweep root: $MODEL_ROOT"
echo "Summary TSV: $SUMMARY_TSV"
echo "Shared spec: $SHARED_SPEC_DIR"

while read -r run_name core_layers core_expansion carry_chunks bptt_chunks residual_core residual_core_init learning_rate lr_hold_steps min_lr num_steps batch_size seq_len; do
  if [[ -z "${run_name:-}" ]] || [[ "$run_name" == \#* ]]; then
    continue
  fi
  if [[ -n "$RUN_FILTER" ]] && [[ "$run_name" != *"$RUN_FILTER"* ]]; then
    continue
  fi

  run_dir="${MODEL_ROOT}/${run_name}"
  log_path="${run_dir}/train.log"

  if bool_true "$SKIP_DONE" && [[ -f "$run_dir/final.pt" ]]; then
    echo "Skipping completed run: $run_name"
    append_summary "$run_name" "$run_dir" "$log_path" "skipped_done" \
      "$core_layers" "$core_expansion" "$carry_chunks" "$bptt_chunks" "$residual_core" "$residual_core_init" \
      "$learning_rate" "$lr_hold_steps" "$min_lr" "$num_steps" "$batch_size" "$seq_len"
    continue
  fi

  prepare_run_dir "$run_dir" "$run_name" "$core_layers" "$core_expansion" "$carry_chunks" "$bptt_chunks" "$residual_core" "$residual_core_init" "$learning_rate" "$lr_hold_steps" "$min_lr" "$num_steps" "$batch_size" "$seq_len"

  cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/train_core_amplifier.py" "$run_dir"
    --data "$DATA_PATH"
    --storage-dtype "$STORAGE_DTYPE"
    --seq-len "$seq_len"
    --batch-size "$batch_size"
    --grad-accum "$GRAD_ACCUM"
    --carry-chunks "$carry_chunks"
    --bptt-chunks "$bptt_chunks"
    --num-steps "$num_steps"
    --learning-rate "$learning_rate"
    --lr-schedule "$LR_SCHEDULE"
    --min-lr "$min_lr"
    --warmup-steps "$WARMUP_STEPS"
    --lr-hold-steps "$lr_hold_steps"
    --weight-decay "$WEIGHT_DECAY"
    --hard-loss-gamma "$HARD_LOSS_GAMMA"
    --hard-loss-cap "$HARD_LOSS_CAP"
    --grad-clip "$GRAD_CLIP"
    --core-layers "$core_layers"
    --core-expansion "$core_expansion"
    --residual-core "$residual_core"
    --residual-core-init "$residual_core_init"
    --val-every "$VAL_EVERY"
    --val-steps "$VAL_STEPS"
    --save-every "$SAVE_EVERY"
    --log-every "$LOG_EVERY"
    --log-state-every "$LOG_STATE_EVERY"
    --train-frac "$TRAIN_FRAC"
    --seed 1337
  )
  if [[ -n "$DATA_MAX_TOKENS" ]]; then
    cmd+=(--data-max-tokens "$DATA_MAX_TOKENS")
  fi
  if bool_true "$NO_MMAP"; then
    cmd+=(--no-mmap)
  fi
  if bool_true "$TOKENS_ON_DEVICE"; then
    cmd+=(--tokens-on-device)
  fi
  if [[ -n "$FORCE_DEVICE" ]]; then
    cmd+=(--force-device "$FORCE_DEVICE")
  fi
  if bool_true "$NO_AUTOCAST"; then
    cmd+=(--no-autocast)
  fi
  if bool_true "$COMPILE"; then
    cmd+=(--compile --compile-after "$COMPILE_AFTER" --compile-mode "$COMPILE_MODE")
    if bool_true "$COMPILE_BASE_PATH"; then
      cmd+=(--compile-base-path)
    fi
  fi

  mkdir -p "$run_dir"
  printf '%q ' "${cmd[@]}" | tee -a "$COMMANDS_TXT"
  printf '\n' | tee -a "$COMMANDS_TXT"

  status="ok"
  if bool_true "$DRY_RUN"; then
    status="dry_run"
    echo "[dry-run] not executing training for $run_name"
  else
    set +e
    "${cmd[@]}" 2>&1 | tee "$log_path"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]]; then
      status="failed"
      echo "Run failed: $run_name (exit $rc)" >&2
    fi
  fi

  append_summary "$run_name" "$run_dir" "$log_path" "$status" \
    "$core_layers" "$core_expansion" "$carry_chunks" "$bptt_chunks" "$residual_core" "$residual_core_init" \
    "$learning_rate" "$lr_hold_steps" "$min_lr" "$num_steps" "$batch_size" "$seq_len"

done <<< "$RUN_SPECS"

if [[ -x "$REPO_ROOT/tools/rebuild_summary.py" ]] && ! bool_true "$DRY_RUN"; then
  echo "Rebuilding clean summary from completed run dirs ..."
  "$PYTHON_BIN" "$REPO_ROOT/tools/rebuild_summary.py" "$MODEL_ROOT" --out "$SUMMARY_TSV" >/dev/null
fi

echo "Done. Summary: $SUMMARY_TSV"
