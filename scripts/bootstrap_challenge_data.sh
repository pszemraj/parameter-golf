#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

mode="${1:-h100}"

usage() {
    cat <<'EOF'
Usage: scripts/bootstrap_challenge_data.sh {smoke|h100|full|help}

Purpose:
  Download the published challenge tokenizer + FineWeb shards into the repo's
  canonical local layout so the training helpers work out of the box.

Modes:
  smoke
    Download the smallest useful local subset:
    - variant: sp1024
    - train shards: 1
    - full validation split

  h100
    Download enough shards for the current 1xH100 HGDN fixed2k runs without
    immediately pulling the much larger default cache:
    - variant: sp1024
    - train shards: 12
    - full validation split

  full
    Download the published default cache:
    - variant: sp1024
    - train shards: 80
    - full validation split

Environment overrides:
  PYTHON_BIN        Python executable, defaults to python3.
  VARIANT           Tokenizer family, defaults to sp1024.
  TRAIN_SHARDS      Overrides the mode default shard count.
  DOWNLOAD_JOBS     Passed through as --jobs to the downloader.
  WITH_DOCS         Set to 1 to also fetch docs_selected.jsonl and its sidecar.
  SKIP_MANIFEST     Set to 1 to pass --skip-manifest.

Notes:
  - Files are downloaded via data/cached_challenge_fineweb.py from the published
    Hugging Face dataset repo.
  - The training helpers expect:
      DATA_PATH=./data/datasets/fineweb10B_sp1024
      TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model

Examples:
  scripts/bootstrap_challenge_data.sh smoke
  scripts/bootstrap_challenge_data.sh h100
  TRAIN_SHARDS=24 scripts/bootstrap_challenge_data.sh h100
  DOWNLOAD_JOBS=16 scripts/bootstrap_challenge_data.sh h100
  WITH_DOCS=1 scripts/bootstrap_challenge_data.sh full
EOF
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

dataset_dir_for_variant() {
    local variant="$1"
    if [[ "$variant" == "byte260" ]]; then
        echo "fineweb10B_byte260"
        return 0
    fi
    if [[ "$variant" =~ ^sp[0-9]+$ ]]; then
        echo "fineweb10B_${variant}"
        return 0
    fi
    echo "Unsupported VARIANT: $variant" >&2
    exit 1
}

tokenizer_model_for_variant() {
    local variant="$1"
    if [[ "$variant" == "byte260" ]]; then
        echo "data/tokenizers/fineweb_pure_byte_260.json"
        return 0
    fi
    if [[ "$variant" =~ ^sp([0-9]+)$ ]]; then
        echo "data/tokenizers/fineweb_${BASH_REMATCH[1]}_bpe.model"
        return 0
    fi
    echo "Unsupported VARIANT: $variant" >&2
    exit 1
}

require_cmd bash

python_bin="${PYTHON_BIN:-python3}"
require_cmd "$python_bin"

variant="${VARIANT:-sp1024}"
case "$mode" in
smoke)
    train_shards_default=1
    ;;
h100)
    train_shards_default=12
    ;;
full)
    train_shards_default=80
    ;;
help|-h|--help)
    usage
    exit 0
    ;;
*)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 1
    ;;
esac

train_shards="${TRAIN_SHARDS:-$train_shards_default}"
dataset_dir="data/datasets/$(dataset_dir_for_variant "$variant")"
tokenizer_path="$(tokenizer_model_for_variant "$variant")"

cmd=("$python_bin" "data/cached_challenge_fineweb.py" "--variant" "$variant" "--train-shards" "$train_shards")
if [[ -n "${DOWNLOAD_JOBS:-}" ]]; then
    cmd+=("--jobs" "$DOWNLOAD_JOBS")
fi
if [[ "${SKIP_MANIFEST:-0}" == "1" ]]; then
    cmd+=("--skip-manifest")
fi
if [[ "${WITH_DOCS:-0}" == "1" ]]; then
    cmd+=("--with-docs")
fi

echo "=== Challenge data bootstrap ==="
echo "mode=$mode"
echo "variant=$variant"
echo "train_shards=$train_shards"
echo "python_bin=$python_bin"
printf 'command='
printf '%q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

if [[ ! -d "$dataset_dir" ]]; then
    echo "Expected dataset directory missing after download: $dataset_dir" >&2
    exit 1
fi
if [[ ! -f "$tokenizer_path" ]]; then
    echo "Expected tokenizer file missing after download: $tokenizer_path" >&2
    exit 1
fi

train_count="$(find "$dataset_dir" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"
val_count="$(find "$dataset_dir" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l | tr -d ' ')"

echo
echo "Download complete."
echo "dataset_dir=$dataset_dir"
echo "tokenizer_path=$tokenizer_path"
echo "train_shards_present=$train_count"
echo "val_shards_present=$val_count"
echo
echo "Training helpers can use:"
echo "  DATA_PATH=$repo_root/$dataset_dir"
echo "  TOKENIZER_PATH=$repo_root/$tokenizer_path"
