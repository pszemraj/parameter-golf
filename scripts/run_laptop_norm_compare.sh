#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

mode="${1:-hybrid}"

usage() {
    cat <<'EOF'
Usage: scripts/run_laptop_norm_compare.sh {hybrid|depth|both|help}

Purpose:
  Single-GPU laptop helper for fixed-step norm-placement screens on the HGDN branch.
  It runs matched pre/post/keel jobs so you can compare train/eval loss curves cleanly.

Modes:
  hybrid
    Run the current HGDN operating point with all three norm styles:
    - preset: single
    - GDN_RATIO=1
    - MLP_MULT=3.25

  depth
    Run the pure-attention depth control with all three norm styles:
    - preset: depth
    - MLP_MULT=4.0

  both
    Run hybrid first, then depth.

Defaults:
  - NGPU=1
  - ITERATIONS=1000
  - MAX_WALLCLOCK_SECONDS=0
  - TRAIN_BATCH_TOKENS=65536
  - TRAIN_SEQ_LEN=1024
  - VAL_LOSS_EVERY=100
  - TRAIN_LOG_EVERY=25
  - COMPILE=1
  - COMPILE_STRATEGY=model
  - WANDB_WATCH=none
  - NORM_STYLES=pre,post,keel
  - USE_WANDB=1
  - WANDB_MODE=online

Environment overrides:
  RUN_PREFIX             Base prefix for run ids.
  USE_WANDB              Defaults to 1.
  WANDB_MODE             Defaults to online.
  WANDB_PROJECT          Passed through to sweep.sh if enabled.
  DATA_PATH              Passed through to sweep.sh.
  TOKENIZER_PATH         Passed through to sweep.sh.
  COMPILE                Defaults to 1.
  COMPILE_STRATEGY       Defaults to model.
  WANDB_WATCH            Defaults to none.
  WANDB_WATCH_LOG_FREQ   Defaults to trainer default.
  TRAIN_BATCH_TOKENS     Defaults to 65536.
  TRAIN_SEQ_LEN          Defaults to 1024.
  ITERATIONS             Defaults to 1000.
  VAL_LOSS_EVERY         Defaults to 100.
  TRAIN_LOG_EVERY        Defaults to 25.
  NORM_STYLES            Comma-separated list, defaults to pre,post,keel.
  HYBRID_GDN_RATIO       Defaults to 1.
  HYBRID_MLP_MULT        Defaults to 3.25.
  DEPTH_MLP_MULT         Defaults to 4.0.

Examples:
  scripts/run_laptop_norm_compare.sh hybrid
  RUN_PREFIX=norma scripts/run_laptop_norm_compare.sh depth
  USE_WANDB=0 WANDB_MODE=offline scripts/run_laptop_norm_compare.sh both
  TRAIN_SEQ_LEN=2048 ITERATIONS=750 scripts/run_laptop_norm_compare.sh both
EOF
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

run_sweep() {
    local label="$1"
    local preset="$2"
    shift 2

    echo
    echo ">>> $label"
    (
        export NGPU=1
        export USE_WANDB="${USE_WANDB:-1}"
        export WANDB_MODE="${WANDB_MODE:-online}"
        export WANDB_WATCH="${WANDB_WATCH:-none}"
        export COMPILE="${COMPILE:-1}"
        export COMPILE_STRATEGY="${COMPILE_STRATEGY:-model}"
        for kv in "$@"; do
            export "$kv"
        done
        bash "$repo_root/scripts/sweep.sh" "$preset"
    )
}

run_norm_triplet() {
    local family="$1"
    local preset="$2"
    local prefix="$3"
    shift 3

    local iterations="${ITERATIONS:-1000}"
    local train_batch_tokens="${TRAIN_BATCH_TOKENS:-65536}"
    local train_seq_len="${TRAIN_SEQ_LEN:-1024}"
    local val_loss_every="${VAL_LOSS_EVERY:-100}"
    local train_log_every="${TRAIN_LOG_EVERY:-25}"
    local norm_styles_csv="${NORM_STYLES:-pre,post,keel}"

    local -a extra_env=("$@")
    IFS=',' read -r -a norm_styles <<<"$norm_styles_csv"

    for norm_style in "${norm_styles[@]}"; do
        case "$norm_style" in
        pre | post | keel) ;;
        *)
            echo "Unsupported NORM_STYLE in list: $norm_style" >&2
            exit 1
            ;;
        esac

        local -a env_args=(
            "RUN_ID=${prefix}_${family}_${norm_style}_seq${train_seq_len}_it${iterations}"
            "NORM_STYLE=${norm_style}"
            "ITERATIONS=${iterations}"
            "MAX_WALLCLOCK_SECONDS=0"
            "TRAIN_BATCH_TOKENS=${train_batch_tokens}"
            "TRAIN_SEQ_LEN=${train_seq_len}"
            "VAL_LOSS_EVERY=${val_loss_every}"
            "TRAIN_LOG_EVERY=${train_log_every}"
        )
        env_args+=("${extra_env[@]}")

        run_sweep \
            "laptop norm screen: ${family} norm=${norm_style}" \
            "$preset" \
            "${env_args[@]}"
    done
}

run_hybrid_triplet() {
    local prefix="$1"
    local hybrid_gdn_ratio="${HYBRID_GDN_RATIO:-1}"
    local hybrid_mlp_mult="${HYBRID_MLP_MULT:-3.25}"

    run_norm_triplet \
        "hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}" \
        single \
        "$prefix" \
        "GDN_RATIO=${hybrid_gdn_ratio}" \
        "MLP_MULT=${hybrid_mlp_mult}"
}

run_depth_triplet() {
    local prefix="$1"
    local depth_mlp_mult="${DEPTH_MLP_MULT:-4.0}"

    run_norm_triplet \
        "depth_mlp${depth_mlp_mult}" \
        depth \
        "$prefix" \
        "MLP_MULT=${depth_mlp_mult}"
}

require_cmd bash
require_cmd torchrun

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_prefix="${RUN_PREFIX:-laptop_norm_${run_stamp}}"

case "$mode" in
hybrid)
    run_hybrid_triplet "$run_prefix"
    ;;
depth)
    run_depth_triplet "$run_prefix"
    ;;
both)
    run_hybrid_triplet "$run_prefix"
    run_depth_triplet "$run_prefix"
    ;;
help | -h | --help)
    usage
    ;;
*)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 1
    ;;
esac
