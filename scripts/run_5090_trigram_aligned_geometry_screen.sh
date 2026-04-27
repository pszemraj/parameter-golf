#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/5090_common.sh"

PYTHON_BIN="${PYTHON:-/home/pszemraj/miniforge3/envs/train/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pg-core-amp}"
WANDB_WATCH="${WANDB_WATCH:-gradients}"
WANDB_WATCH_LOG_FREQ="${WANDB_WATCH_LOG_FREQ:-25}"
PRESET="${PRESET:-controller_default}"
COMPILE="${COMPILE:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SCAN_BACKEND="${SCAN_BACKEND:-auto}"
export TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
REBUILD_SHARED="${REBUILD_SHARED:-0}"
ALLOW_REBUILD_SHARED="${ALLOW_REBUILD_SHARED:-0}"

RUN_VERSION="${RUN_VERSION:-geom1}"
SEEDS="${SEEDS:-1337}"
SKIP_DONE="${SKIP_DONE:-1}"
RUN_BLOCKS0="${RUN_BLOCKS0:-1}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"

LEARNING_RATE="${LEARNING_RATE:-0.0035}"

COREAMP_SPEC_CACHE_ROOT="${COREAMP_SPEC_CACHE_ROOT:-${HOME}/.cache/experiments/param-golf-coreamp}"
GEOMETRY_CORE_DIM="${GEOMETRY_CORE_DIM:-128}"
GEOMETRY_CORE_LAYERS="${GEOMETRY_CORE_LAYERS:-4}"
GEOMETRY_CORE_INNER_DIM="${GEOMETRY_CORE_INNER_DIM:-512}"
GEOMETRY_CORE_EXPANSION="${GEOMETRY_CORE_EXPANSION:-}"
GEOMETRY_NUM_BLOCKS="${GEOMETRY_NUM_BLOCKS:-0}"
GEOMETRY_BRANCH_LAGS="${GEOMETRY_BRANCH_LAGS:-1,2,3,4,6,8,12,16,24,32,48,64}"
GEOMETRY_LABEL="${GEOMETRY_LABEL:-}"
GEOMETRY_CARRY_CHUNKS="${GEOMETRY_CARRY_CHUNKS:-8}"
GEOMETRY_BPTT_CHUNKS="${GEOMETRY_BPTT_CHUNKS:-1}"
GEOMETRY_NUM_STEPS="${GEOMETRY_NUM_STEPS:-4096}"
GEOMETRY_BATCH_SIZE="${GEOMETRY_BATCH_SIZE:-256}"
GEOMETRY_SEQ_LEN="${GEOMETRY_SEQ_LEN:-512}"
GEOMETRY_LR_WARMUP_STEPS="${GEOMETRY_LR_WARMUP_STEPS:-100}"
GEOMETRY_LR_HOLD_STEPS="${GEOMETRY_LR_HOLD_STEPS:-3500}"
GEOMETRY_MIN_LR="${GEOMETRY_MIN_LR:-0.0003}"
GEOMETRY_TRAIN_LABEL="${GEOMETRY_TRAIN_LABEL:-}"

TRIGRAM_MEMORY="${TRIGRAM_MEMORY:-frozen}"
TRIGRAM_LOG_SCALE_INIT="${TRIGRAM_LOG_SCALE_INIT:-0.0}"
TRIGRAM_TOP_K="${TRIGRAM_TOP_K:-2}"
TRIGRAM_RESIDUAL_CLIP="${TRIGRAM_RESIDUAL_CLIP:-8.0}"
TRIGRAM_CONFIDENCE_COUNT_CAP="${TRIGRAM_CONFIDENCE_COUNT_CAP:-4096}"
TRIGRAM_CHUNK_SIZE="${TRIGRAM_CHUNK_SIZE:-50000000}"
TRIGRAM_COUNT_WORKERS="${TRIGRAM_COUNT_WORKERS:-1}"
TRIGRAM_MEMORY_TABLE_CACHE_ROOT="${TRIGRAM_MEMORY_TABLE_CACHE_ROOT:-${COREAMP_SPEC_CACHE_ROOT}/trigram_memory_tables}"
REBUILD_TRIGRAM_MEMORY_TABLE_CACHE="${REBUILD_TRIGRAM_MEMORY_TABLE_CACHE:-0}"
TRIGRAM_MAX_TOKENS="${TRIGRAM_MAX_TOKENS:-}"
SPEC_MAX_TOKENS="${SPEC_MAX_TOKENS:-}"
ARTIFACT_PREFLIGHT="${ARTIFACT_PREFLIGHT:-1}"
PREFLIGHT_TRAINABLE_PAYLOAD_BYTES="${PREFLIGHT_TRAINABLE_PAYLOAD_BYTES:-2000000}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

TARGET_EFFECTIVE_STEP_TOKENS="${TARGET_EFFECTIVE_STEP_TOKENS:-131072}"
DATA_MAX_TOKENS="${DATA_MAX_TOKENS:-}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
STORAGE_DTYPE="${STORAGE_DTYPE:-uint16}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
HARD_LOSS_GAMMA="${HARD_LOSS_GAMMA:-0.5}"
HARD_LOSS_CAP="${HARD_LOSS_CAP:-5.0}"
VAL_EVERY="${VAL_EVERY:-256}"
VAL_STEPS="${VAL_STEPS:-8}"
LOG_EVERY="${LOG_EVERY:-64}"
LOG_STATE_EVERY="${LOG_STATE_EVERY:-256}"
SAVE_EVERY="${SAVE_EVERY:-2048}"
FULL_VAL_FINAL="${FULL_VAL_FINAL:-0}"
if [[ -z "${MMAP+x}" ]]; then
  if [[ "${NO_MMAP:-0}" == "1" ]]; then
    MMAP=0
  else
    MMAP=1
  fi
fi
if [[ -z "${AUTOCAST+x}" ]]; then
  if [[ "${NO_AUTOCAST:-0}" == "1" ]]; then
    AUTOCAST=0
  else
    AUTOCAST=1
  fi
fi
TOKENS_ON_DEVICE="${TOKENS_ON_DEVICE:-0}"
BASE_BIGRAM_DELTA="${BASE_BIGRAM_DELTA:-none}"
RESIDUAL_READOUT_DELTA_RANK="${RESIDUAL_READOUT_DELTA_RANK:-0}"
RESIDUAL_READOUT_DELTA_INIT_STD="${RESIDUAL_READOUT_DELTA_INIT_STD:-0.02}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Active 5090 aligned-geometry screen. CLI flags are the experiment protocol;
environment variables are retained only as defaults/legacy convenience.

Options:
  --run-version VALUE
  --seeds VALUE
  --geometry-label VALUE
  --geometry-core-dim VALUE
  --geometry-core-layers VALUE
  --geometry-core-inner-dim VALUE
  --geometry-core-expansion VALUE
  --geometry-num-blocks VALUE
  --geometry-branch-lags VALUE
  --geometry-batch-size VALUE | --batch-size VALUE
  --geometry-seq-len VALUE | --seq-len VALUE
  --geometry-bptt-chunks VALUE
  --geometry-carry-chunks VALUE
  --num-steps VALUE
  --lr-warmup-steps VALUE
  --lr-hold-steps VALUE
  --min-lr VALUE
  --trigram-top-k VALUE
  --trigram-max-tokens VALUE
  --spec-max-tokens VALUE
  --data-max-tokens VALUE
  --count-workers VALUE
  --rebuild-shared | --no-rebuild-shared
  --artifact-preflight | --no-artifact-preflight
  --preflight-trainable-payload-bytes VALUE
  --preflight-only
  --full-val-final | --no-full-val-final
  --smoke-test
  --dry-run
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-version) RUN_VERSION="$2"; shift 2 ;;
      --seeds) SEEDS="$2"; shift 2 ;;
      --skip-done) SKIP_DONE="$2"; shift 2 ;;
      --run-blocks0) RUN_BLOCKS0="$2"; shift 2 ;;
      --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
      --coreamp-spec-cache-root) COREAMP_SPEC_CACHE_ROOT="$2"; shift 2 ;;
      --geometry-label) GEOMETRY_LABEL="$2"; shift 2 ;;
      --geometry-core-dim) GEOMETRY_CORE_DIM="$2"; shift 2 ;;
      --geometry-core-layers) GEOMETRY_CORE_LAYERS="$2"; shift 2 ;;
      --geometry-core-inner-dim) GEOMETRY_CORE_INNER_DIM="$2"; shift 2 ;;
      --geometry-core-expansion) GEOMETRY_CORE_EXPANSION="$2"; shift 2 ;;
      --geometry-num-blocks) GEOMETRY_NUM_BLOCKS="$2"; shift 2 ;;
      --geometry-branch-lags) GEOMETRY_BRANCH_LAGS="$2"; shift 2 ;;
      --geometry-carry-chunks) GEOMETRY_CARRY_CHUNKS="$2"; shift 2 ;;
      --geometry-bptt-chunks) GEOMETRY_BPTT_CHUNKS="$2"; shift 2 ;;
      --geometry-num-steps|--num-steps) GEOMETRY_NUM_STEPS="$2"; shift 2 ;;
      --geometry-batch-size|--batch-size) GEOMETRY_BATCH_SIZE="$2"; shift 2 ;;
      --geometry-seq-len|--seq-len) GEOMETRY_SEQ_LEN="$2"; shift 2 ;;
      --geometry-lr-warmup-steps|--lr-warmup-steps) GEOMETRY_LR_WARMUP_STEPS="$2"; shift 2 ;;
      --geometry-lr-hold-steps|--lr-hold-steps) GEOMETRY_LR_HOLD_STEPS="$2"; shift 2 ;;
      --geometry-min-lr|--min-lr) GEOMETRY_MIN_LR="$2"; shift 2 ;;
      --geometry-train-label) GEOMETRY_TRAIN_LABEL="$2"; shift 2 ;;
      --trigram-memory) TRIGRAM_MEMORY="$2"; shift 2 ;;
      --trigram-log-scale-init) TRIGRAM_LOG_SCALE_INIT="$2"; shift 2 ;;
      --trigram-top-k) TRIGRAM_TOP_K="$2"; shift 2 ;;
      --trigram-max-tokens) TRIGRAM_MAX_TOKENS="$2"; shift 2 ;;
      --spec-max-tokens) SPEC_MAX_TOKENS="$2"; shift 2 ;;
      --trigram-residual-clip) TRIGRAM_RESIDUAL_CLIP="$2"; shift 2 ;;
      --trigram-confidence-count-cap) TRIGRAM_CONFIDENCE_COUNT_CAP="$2"; shift 2 ;;
      --trigram-chunk-size) TRIGRAM_CHUNK_SIZE="$2"; shift 2 ;;
      --trigram-count-workers|--count-workers) TRIGRAM_COUNT_WORKERS="$2"; shift 2 ;;
      --rebuild-shared) REBUILD_SHARED=1; ALLOW_REBUILD_SHARED=1; shift ;;
      --no-rebuild-shared) REBUILD_SHARED=0; shift ;;
      --artifact-preflight) ARTIFACT_PREFLIGHT=1; shift ;;
      --no-artifact-preflight) ARTIFACT_PREFLIGHT=0; shift ;;
      --preflight-trainable-payload-bytes) PREFLIGHT_TRAINABLE_PAYLOAD_BYTES="$2"; shift 2 ;;
      --preflight-only) PREFLIGHT_ONLY=1; shift ;;
      --target-effective-step-tokens) TARGET_EFFECTIVE_STEP_TOKENS="$2"; shift 2 ;;
      --data-max-tokens) DATA_MAX_TOKENS="$2"; shift 2 ;;
      --data-path) DATA_PATH="$2"; shift 2 ;;
      --storage-dtype) STORAGE_DTYPE="$2"; shift 2 ;;
      --lr-schedule) LR_SCHEDULE="$2"; shift 2 ;;
      --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
      --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
      --hard-loss-gamma) HARD_LOSS_GAMMA="$2"; shift 2 ;;
      --hard-loss-cap) HARD_LOSS_CAP="$2"; shift 2 ;;
      --val-every) VAL_EVERY="$2"; shift 2 ;;
      --val-steps) VAL_STEPS="$2"; shift 2 ;;
      --log-every) LOG_EVERY="$2"; shift 2 ;;
      --log-state-every) LOG_STATE_EVERY="$2"; shift 2 ;;
      --save-every) SAVE_EVERY="$2"; shift 2 ;;
      --full-val-final) FULL_VAL_FINAL=1; shift ;;
      --no-full-val-final) FULL_VAL_FINAL=0; shift ;;
      --mmap) MMAP=1; shift ;;
      --no-mmap) MMAP=0; shift ;;
      --autocast) AUTOCAST=1; shift ;;
      --no-autocast) AUTOCAST=0; shift ;;
      --tokens-on-device) TOKENS_ON_DEVICE=1; shift ;;
      --no-tokens-on-device) TOKENS_ON_DEVICE=0; shift ;;
      --compile) COMPILE=1; shift ;;
      --no-compile) COMPILE=0; shift ;;
      --gradient-checkpointing) GRADIENT_CHECKPOINTING=1; shift ;;
      --no-gradient-checkpointing) GRADIENT_CHECKPOINTING=0; shift ;;
      --wandb) WANDB=1; shift ;;
      --no-wandb) WANDB=0; shift ;;
      --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
      --wandb-watch) WANDB_WATCH="$2"; shift 2 ;;
      --wandb-watch-log-freq) WANDB_WATCH_LOG_FREQ="$2"; shift 2 ;;
      --scan-backend) SCAN_BACKEND="$2"; shift 2 ;;
      --smoke-test) SMOKE_TEST=1; shift ;;
      --base-bigram-delta) BASE_BIGRAM_DELTA="$2"; shift 2 ;;
      --residual-readout-delta-rank) RESIDUAL_READOUT_DELTA_RANK="$2"; shift 2 ;;
      --residual-readout-delta-init-std) RESIDUAL_READOUT_DELTA_INIT_STD="$2"; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
  done
}

parse_args "$@"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  export ALLOW_DEGRADED_5090_SERIOUS=1
  WANDB=0
fi

if [[ -z "${GEOMETRY_CORE_EXPANSION:-}" ]]; then
  GEOMETRY_CORE_EXPANSION="$(
    "${PYTHON_BIN}" - "${GEOMETRY_CORE_DIM}" "${GEOMETRY_CORE_INNER_DIM}" <<'PY'
import sys

core_dim = int(sys.argv[1])
inner_dim = int(sys.argv[2])
print(repr(float(inner_dim) / float(core_dim)))
PY
  )"
fi
GEOMETRY_CORE_EXPANSION="${GEOMETRY_CORE_EXPANSION}"
GEOMETRY_CORE_INNER_DIM_RESOLVED="$(
  "${PYTHON_BIN}" - "${GEOMETRY_CORE_DIM}" "${GEOMETRY_CORE_EXPANSION}" <<'PY'
import sys

core_dim = int(sys.argv[1])
expansion = float(sys.argv[2])
print(int(core_dim * expansion))
PY
)"
GEOMETRY_CORE_INNER_DIM_RESOLVED="${GEOMETRY_CORE_INNER_DIM_RESOLVED}"
if [[ -n "${GEOMETRY_CORE_INNER_DIM:-}" && "${GEOMETRY_CORE_INNER_DIM_RESOLVED}" != "${GEOMETRY_CORE_INNER_DIM}" ]]; then
  pg_5090_fail "$(basename "$0")" "GEOMETRY_CORE_DIM=${GEOMETRY_CORE_DIM} and GEOMETRY_CORE_EXPANSION=${GEOMETRY_CORE_EXPANSION} resolve to inner=${GEOMETRY_CORE_INNER_DIM_RESOLVED}, expected ${GEOMETRY_CORE_INNER_DIM}"
fi
if [[ -z "${GEOMETRY_LABEL:-}" ]]; then
  GEOMETRY_LABEL="blocks0_core${GEOMETRY_CORE_DIM}_l${GEOMETRY_CORE_LAYERS}_i${GEOMETRY_CORE_INNER_DIM_RESOLVED}"
fi
if [[ -z "${GEOMETRY_TRAIN_LABEL:-}" ]]; then
  if [[ "${GEOMETRY_NUM_STEPS}" == "4096" && "${TARGET_EFFECTIVE_STEP_TOKENS}" == "131072" ]]; then
    GEOMETRY_TRAIN_LABEL="512m"
  elif [[ "${GEOMETRY_NUM_STEPS}" == "8192" && "${TARGET_EFFECTIVE_STEP_TOKENS}" == "131072" ]]; then
    GEOMETRY_TRAIN_LABEL="1b"
  else
    GEOMETRY_TRAIN_LABEL="${GEOMETRY_NUM_STEPS}steps"
  fi
fi
LEARNING_RATE_TAG="$(pg_5090_lr_slug "${LEARNING_RATE}")"

pg_5090_require_serious_launcher_defaults "$(basename "$0")"
if [[ "${ALLOW_DEGRADED_5090_SERIOUS:-0}" != "1" && -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
  pg_5090_fail "$(basename "$0")" "TRIGRAM_MAX_TOKENS must be unset for serious runs"
fi
if [[ "${ALLOW_DEGRADED_5090_SERIOUS:-0}" != "1" && -n "${SPEC_MAX_TOKENS:-}" ]]; then
  pg_5090_fail "$(basename "$0")" "SPEC_MAX_TOKENS must be unset for serious runs"
fi
if [[ "${ALLOW_DEGRADED_5090_SERIOUS:-0}" != "1" && -n "${DATA_MAX_TOKENS:-}" ]]; then
  pg_5090_fail "$(basename "$0")" "DATA_MAX_TOKENS must be unset for serious runs"
fi

slugify() {
  local raw="$1"
  raw="${raw//./p}"
  raw="${raw//,/m}"
  raw="${raw// /_}"
  printf '%s' "${raw}"
}

shared_spec_dir() {
  local label
  local token_scope
  token_scope="full"
  if [[ -n "${SPEC_MAX_TOKENS:-}" || -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
    token_scope="spec${SPEC_MAX_TOKENS:-full}_trigram${TRIGRAM_MAX_TOKENS:-full}"
  fi
  label="$(slugify "${GEOMETRY_LABEL}_branches_${GEOMETRY_BRANCH_LAGS}_blocks${GEOMETRY_NUM_BLOCKS}_${TRIGRAM_MEMORY}_trigramk${TRIGRAM_TOP_K}_clip${TRIGRAM_RESIDUAL_CLIP}_cap${TRIGRAM_CONFIDENCE_COUNT_CAP}_${token_scope}")"
  printf '%s/shared_specs/%s' "${COREAMP_SPEC_CACHE_ROOT}" "${label}"
}

shared_spec_manifest() {
  local mode="$1"
  local out_dir="$2"
  SPEC_MANIFEST_MODE="${mode}" \
  SPEC_MANIFEST_PATH="${out_dir}/shared_spec_manifest.json" \
  SPEC_REPO_ROOT="${REPO_ROOT}" \
  SPEC_DATA_PATH="${DATA_PATH}" \
  SPEC_STORAGE_DTYPE="${STORAGE_DTYPE}" \
  SPEC_CORE_DIM="${GEOMETRY_CORE_DIM}" \
  SPEC_CORE_LAYERS="${GEOMETRY_CORE_LAYERS}" \
  SPEC_CORE_EXPANSION="${GEOMETRY_CORE_EXPANSION}" \
  SPEC_CORE_INNER_DIM="${GEOMETRY_CORE_INNER_DIM_RESOLVED}" \
  SPEC_BRANCH_LAGS="${GEOMETRY_BRANCH_LAGS}" \
  SPEC_NUM_BLOCKS="${GEOMETRY_NUM_BLOCKS}" \
  SPEC_TRIGRAM_MEMORY="${TRIGRAM_MEMORY}" \
  SPEC_TRIGRAM_LOG_SCALE_INIT="${TRIGRAM_LOG_SCALE_INIT}" \
  SPEC_TRIGRAM_TOP_K="${TRIGRAM_TOP_K}" \
  SPEC_TRIGRAM_RESIDUAL_CLIP="${TRIGRAM_RESIDUAL_CLIP}" \
  SPEC_TRIGRAM_CONFIDENCE_COUNT_CAP="${TRIGRAM_CONFIDENCE_COUNT_CAP}" \
  SPEC_TRIGRAM_CHUNK_SIZE="${TRIGRAM_CHUNK_SIZE}" \
  SPEC_TRIGRAM_MAX_TOKENS="${TRIGRAM_MAX_TOKENS:-}" \
  SPEC_MAX_TOKENS="${SPEC_MAX_TOKENS:-}" \
  SPEC_TRIGRAM_TABLE_CACHE_ROOT="${TRIGRAM_MEMORY_TABLE_CACHE_ROOT}" \
  SPEC_SCAN_BACKEND="${SCAN_BACKEND}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["SPEC_REPO_ROOT"]).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core_amplifier_lm.spec_builder import training_token_file_fingerprint

def optional_int(raw):
    if raw in (None, ""):
        return None
    return int(raw)


def normalize_optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


data_path = Path(os.environ["SPEC_DATA_PATH"]).expanduser().resolve()
manifest_path = Path(os.environ["SPEC_MANIFEST_PATH"])
expected = {
    "data_path": str(data_path),
    "train_fingerprint": training_token_file_fingerprint(data_path),
    "storage_dtype": os.environ["SPEC_STORAGE_DTYPE"],
    "vocab_size": 1024,
    "core_dim": int(os.environ["SPEC_CORE_DIM"]),
    "core_layers": int(os.environ["SPEC_CORE_LAYERS"]),
    "core_expansion": os.environ["SPEC_CORE_EXPANSION"],
    "core_inner_dim": int(os.environ["SPEC_CORE_INNER_DIM"]),
    "branch_lags": os.environ["SPEC_BRANCH_LAGS"],
    "num_blocks": int(os.environ["SPEC_NUM_BLOCKS"]),
    "fixed_dtype": "bfloat16",
    "embedding_init": "spectral",
    "spectral_neighbors": 64,
    "lag_identity_base": "0.15",
    "residual_core": 1,
    "residual_core_init": "-3.0",
    "branch_temporal_mode": "current",
    "residual_token_gate_mode": "none",
    "branch_router_mode": "none",
    "base_bigram_delta": "none",
    "trigram_memory": os.environ["SPEC_TRIGRAM_MEMORY"],
    "trigram_log_scale_init": os.environ["SPEC_TRIGRAM_LOG_SCALE_INIT"],
    "trigram_top_k": int(os.environ["SPEC_TRIGRAM_TOP_K"]),
    "trigram_smoothing": "0.25",
    "trigram_residual_clip": os.environ["SPEC_TRIGRAM_RESIDUAL_CLIP"],
    "trigram_confidence_count_cap": int(os.environ["SPEC_TRIGRAM_CONFIDENCE_COUNT_CAP"]),
    "trigram_chunk_size": int(os.environ["SPEC_TRIGRAM_CHUNK_SIZE"]),
    "trigram_max_tokens": optional_int(os.environ["SPEC_TRIGRAM_MAX_TOKENS"]),
    "spec_max_tokens": optional_int(os.environ["SPEC_MAX_TOKENS"]),
    "trigram_table_cache_root": str(
        Path(os.environ["SPEC_TRIGRAM_TABLE_CACHE_ROOT"]).expanduser().resolve()
    ),
    "scan_backend": os.environ["SPEC_SCAN_BACKEND"],
}
mode = os.environ["SPEC_MANIFEST_MODE"]
if mode == "write":
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
elif mode == "validate":
    if not manifest_path.exists():
        raise SystemExit(f"missing shared spec manifest: {manifest_path}")
    found = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("trigram_max_tokens", "spec_max_tokens"):
        found[key] = normalize_optional_int(found.get(key))
    mismatches = {
        key: (found.get(key), value)
        for key, value in expected.items()
        if found.get(key) != value
    }
    if mismatches:
        print("shared spec cache contract mismatch:", file=sys.stderr)
        for key, (actual, expected_value) in sorted(mismatches.items()):
            print(f"  {key}: found={actual!r} expected={expected_value!r}", file=sys.stderr)
        raise SystemExit(1)
else:
    raise SystemExit(f"unknown shared spec manifest mode: {mode}")
PY
}

ensure_shared_spec() {
  local out_dir="$1"
  if [[ "${REBUILD_SHARED:-0}" != "1" && "${REBUILD_GEOMETRY_SPEC:-0}" != "1" && -f "${out_dir}/spec.pt" && -f "${out_dir}/config.json" ]]; then
    shared_spec_manifest validate "${out_dir}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "Dry-run validated cached aligned shared spec manifest: ${out_dir}"
      return 0
    fi
    echo "Using cached aligned shared spec: ${out_dir}"
    return 0
  fi

  echo "Building aligned shared spec from full training shards: ${out_dir}"
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/inspect_model.py" init "${out_dir}"
    --data "${DATA_PATH}"
    --storage-dtype "${STORAGE_DTYPE}"
    --suppress-config-summary
    --vocab-size 1024
    --core-dim "${GEOMETRY_CORE_DIM}"
    --branch-lags "${GEOMETRY_BRANCH_LAGS}"
    --num-blocks "${GEOMETRY_NUM_BLOCKS}"
    --fixed-dtype bfloat16
    --embedding-init spectral
    --spectral-neighbors 64
    --lag-identity-base 0.15
    --spec-strategy auto
    --spec-workers -1
    --core-layers "${GEOMETRY_CORE_LAYERS}"
    --core-expansion "${GEOMETRY_CORE_EXPANSION}"
    --residual-core 1
    --residual-core-init -3.0
    --branch-temporal-mode current
    --residual-token-gate-mode none
    --branch-router-mode none
    --base-bigram-delta none
    --trigram-memory "${TRIGRAM_MEMORY}"
    --trigram-log-scale-init "${TRIGRAM_LOG_SCALE_INIT}"
    --trigram-top-k "${TRIGRAM_TOP_K}"
    --trigram-smoothing 0.25
    --trigram-residual-clip "${TRIGRAM_RESIDUAL_CLIP}"
    --trigram-confidence-count-cap "${TRIGRAM_CONFIDENCE_COUNT_CAP}"
    --trigram-chunk-size "${TRIGRAM_CHUNK_SIZE}"
    --trigram-count-workers "${TRIGRAM_COUNT_WORKERS}"
    --trigram-table-cache-root "${TRIGRAM_MEMORY_TABLE_CACHE_ROOT}"
    --scan-backend "${SCAN_BACKEND}"
  )
  if [[ "${REBUILD_TRIGRAM_MEMORY_TABLE_CACHE}" == "1" ]]; then
    cmd+=(--rebuild-trigram-table-cache)
  fi
  if [[ -n "${TRIGRAM_MAX_TOKENS:-}" ]]; then
    cmd+=(--trigram-max-tokens "${TRIGRAM_MAX_TOKENS}")
  fi
  if [[ -n "${SPEC_MAX_TOKENS:-}" ]]; then
    cmd+=(--max-tokens "${SPEC_MAX_TOKENS}")
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
  shared_spec_manifest write "${out_dir}"
}

preflight_artifact_budget() {
  local spec_dir="$1"
  if [[ "${ARTIFACT_PREFLIGHT}" != "1" ]]; then
    return 0
  fi
  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/estimate_artifact_bytes.py" "${spec_dir}"
    --assume-trainable-payload-bytes "${PREFLIGHT_TRAINABLE_PAYLOAD_BYTES}"
    --fail-over-limit
  )
  echo "Preflight artifact budget check: ${spec_dir}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

print_header() {
  cat <<EOF
5090 trigram aligned-geometry screen
repo_root=${REPO_ROOT}
python=${PYTHON_BIN}
seeds=${SEEDS}
run_version=${RUN_VERSION}
geometry_core_dim=${GEOMETRY_CORE_DIM} layers=${GEOMETRY_CORE_LAYERS} inner_dim=${GEOMETRY_CORE_INNER_DIM_RESOLVED} expansion=${GEOMETRY_CORE_EXPANSION} blocks=${GEOMETRY_NUM_BLOCKS}
geometry_branch_lags=${GEOMETRY_BRANCH_LAGS}
trigram_memory=${TRIGRAM_MEMORY} top_k=${TRIGRAM_TOP_K} log_scale_init=${TRIGRAM_LOG_SCALE_INIT} count_workers=${TRIGRAM_COUNT_WORKERS}
rebuild_shared=${REBUILD_SHARED} rebuild_trigram_memory_table_cache=${REBUILD_TRIGRAM_MEMORY_TABLE_CACHE}
spec_max_tokens=${SPEC_MAX_TOKENS:-}
artifact_preflight=${ARTIFACT_PREFLIGHT} preflight_trainable_payload_bytes=${PREFLIGHT_TRAINABLE_PAYLOAD_BYTES}
preflight_only=${PREFLIGHT_ONLY}
learning_rate=${LEARNING_RATE}
compile=${COMPILE} gradient_checkpointing=${GRADIENT_CHECKPOINTING} skip_done=${SKIP_DONE}
lr_schedule=${LR_SCHEDULE} weight_decay=${WEIGHT_DECAY} grad_clip=${GRAD_CLIP} hard_loss_gamma=${HARD_LOSS_GAMMA} hard_loss_cap=${HARD_LOSS_CAP}
val_every=${VAL_EVERY} val_steps=${VAL_STEPS} log_every=${LOG_EVERY} log_state_every=${LOG_STATE_EVERY} save_every=${SAVE_EVERY} full_val_final=${FULL_VAL_FINAL}
validation_policy=explicit_val_shard_required mmap=${MMAP} autocast=${AUTOCAST} tokens_on_device=${TOKENS_ON_DEVICE}
num_steps=${GEOMETRY_NUM_STEPS} lr_hold_steps=${GEOMETRY_LR_HOLD_STEPS} carry_chunks=${GEOMETRY_CARRY_CHUNKS} bptt_chunks=${GEOMETRY_BPTT_CHUNKS}
batch_size=${GEOMETRY_BATCH_SIZE} seq_len=${GEOMETRY_SEQ_LEN} target_effective_step_tokens=${TARGET_EFFECTIVE_STEP_TOKENS}
data_max_tokens=${DATA_MAX_TOKENS:-}
coreamp_spec_cache_root=${COREAMP_SPEC_CACHE_ROOT}
trigram_memory_table_cache_root=${TRIGRAM_MEMORY_TABLE_CACHE_ROOT}
scan_backend=${SCAN_BACKEND} wandb_project=${WANDB_PROJECT} cublaslt=${TORCH_BLAS_PREFER_CUBLASLT} py_unbuffered=${PYTHONUNBUFFERED}
smoke_test=${SMOKE_TEST}
EOF
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "dry_run=1"
  fi
}

run_blocks0_seed() {
  local seed="$1"
  local source_spec_dir
  source_spec_dir="$(shared_spec_dir)"
  ensure_shared_spec "${source_spec_dir}"
  REBUILD_SHARED=0
  preflight_artifact_budget "${source_spec_dir}"
  if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "Dry-run preflight-only requested; would stop after shared-spec artifact preflight."
    else
      echo "Preflight-only requested; shared spec is prepared and artifact budget check passed."
    fi
    return 0
  fi

  local model_root="${REPO_ROOT}/experiments/5090_architecture/${GEOMETRY_LABEL}_trigram_seed${seed}_${RUN_VERSION}"
  local run_label="${GEOMETRY_TRAIN_LABEL}"
  if [[ "${run_label}" != *"k${TRIGRAM_TOP_K}"* ]]; then
    run_label="${run_label}_k${TRIGRAM_TOP_K}"
  fi
  local run_name
  run_name="$(slugify "${GEOMETRY_LABEL}_trigram_${run_label}")"
  local run_specs
  read -r -d '' run_specs <<EOF || true
${run_name} ${GEOMETRY_CORE_LAYERS} ${GEOMETRY_CORE_EXPANSION} ${GEOMETRY_CARRY_CHUNKS} ${GEOMETRY_BPTT_CHUNKS} 1 -3.0 ${LEARNING_RATE} ${GEOMETRY_LR_WARMUP_STEPS} ${GEOMETRY_LR_HOLD_STEPS} ${GEOMETRY_MIN_LR} ${GEOMETRY_NUM_STEPS} ${GEOMETRY_BATCH_SIZE} ${GEOMETRY_SEQ_LEN}
EOF

  echo
  echo "[blocks0_aligned] seed=${seed} trigram_top_k=${TRIGRAM_TOP_K} lr=${LEARNING_RATE} inner_dim=${GEOMETRY_CORE_INNER_DIM_RESOLVED} model_root=${model_root}"
  local sweep_cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/run_core_amp_sweep.py" controller
    --preset "${PRESET}"
    --seed "${seed}"
    --data-path "${DATA_PATH}"
    --storage-dtype "${STORAGE_DTYPE}"
    --target-effective-step-tokens "${TARGET_EFFECTIVE_STEP_TOKENS}"
    --shared-spec-dir "${source_spec_dir}"
    --model-root "${model_root}"
    --lr-schedule "${LR_SCHEDULE}"
    --weight-decay "${WEIGHT_DECAY}"
    --grad-clip "${GRAD_CLIP}"
    --hard-loss-gamma "${HARD_LOSS_GAMMA}"
    --hard-loss-cap "${HARD_LOSS_CAP}"
    --val-every "${VAL_EVERY}"
    --val-steps "${VAL_STEPS}"
    --log-every "${LOG_EVERY}"
    --log-state-every "${LOG_STATE_EVERY}"
    --save-every "${SAVE_EVERY}"
    --core-dim "${GEOMETRY_CORE_DIM}"
    --core-layers "${GEOMETRY_CORE_LAYERS}"
    --core-expansion "${GEOMETRY_CORE_EXPANSION}"
    --residual-core 1
    --residual-core-init -3.0
    --num-blocks "${GEOMETRY_NUM_BLOCKS}"
    --branch-lags "${GEOMETRY_BRANCH_LAGS}"
    --base-bigram-delta "${BASE_BIGRAM_DELTA}"
    --trigram-memory "${TRIGRAM_MEMORY}"
    --trigram-log-scale-init "${TRIGRAM_LOG_SCALE_INIT}"
    --trigram-top-k "${TRIGRAM_TOP_K}"
    --trigram-smoothing 0.25
    --trigram-residual-clip "${TRIGRAM_RESIDUAL_CLIP}"
    --trigram-confidence-count-cap "${TRIGRAM_CONFIDENCE_COUNT_CAP}"
    --trigram-chunk-size "${TRIGRAM_CHUNK_SIZE}"
    --trigram-count-workers "${TRIGRAM_COUNT_WORKERS}"
    --trigram-table-cache-root "${TRIGRAM_MEMORY_TABLE_CACHE_ROOT}"
    --residual-readout-delta-rank "${RESIDUAL_READOUT_DELTA_RANK}"
    --residual-readout-delta-init-std "${RESIDUAL_READOUT_DELTA_INIT_STD}"
    --scan-backend "${SCAN_BACKEND}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-watch "${WANDB_WATCH}"
    --wandb-watch-log-freq "${WANDB_WATCH_LOG_FREQ}"
    --wandb-group "${GEOMETRY_LABEL}_trigram${GEOMETRY_TRAIN_LABEL}_${RUN_VERSION}"
    --wandb-tags "core_amp,5090,architecture,trigram_memory,aligned_geometry,screening,${LEARNING_RATE_TAG}"
    --core-amp-phase "5090_trigram_aligned_geometry_screen"
    --run-spec "${run_specs}"
  )
  if [[ -n "${DATA_MAX_TOKENS:-}" ]]; then
    sweep_cmd+=(--data-max-tokens "${DATA_MAX_TOKENS}")
  fi
  if [[ -n "${SPEC_MAX_TOKENS:-}" ]]; then
    sweep_cmd+=(--spec-max-tokens "${SPEC_MAX_TOKENS}")
  fi
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "compile" "${COMPILE}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "gradient-checkpointing" "${GRADIENT_CHECKPOINTING}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "full-val-final" "${FULL_VAL_FINAL}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "mmap" "${MMAP}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "autocast" "${AUTOCAST}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "tokens-on-device" "${TOKENS_ON_DEVICE}"
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "skip-done" "${SKIP_DONE}"
  sweep_cmd+=(--no-rebuild-shared)
  pg_5090_append_bool_flag "$(basename "$0")" sweep_cmd "wandb" "${WANDB}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+ REBUILD_SHARED=0'
    printf ' %q' "${sweep_cmd[@]}"
    printf '\n'
    return 0
  fi
  REBUILD_SHARED=0 "${sweep_cmd[@]}"
}

main() {
  local seed
  print_header
  for seed in ${SEEDS}; do
    if [[ "${RUN_BLOCKS0}" == "1" ]]; then
      run_blocks0_seed "${seed}"
    fi
  done
}

main
