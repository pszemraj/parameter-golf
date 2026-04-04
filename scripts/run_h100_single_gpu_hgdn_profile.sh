#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

mode="${1:-hybrid}"

usage() {
    cat <<'EOF'
Usage: scripts/run_h100_single_gpu_hgdn_profile.sh {hybrid|depth|both|help}

Purpose:
  Run a short 1xH100 profiling capture for the current HGDN finalist pair.
  This is for trace collection, not leaderboard-quality training.

Modes:
  hybrid
    Profile the current hybrid operating point:
    - GDN_RATIO=1
    - MLP_MULT=3.25

  depth
    Profile the pure-attention depth control:
    - GDN_RATIO=0 via the `depth` preset
    - MLP_MULT=4.0

  both
    Run hybrid first, then depth.

Defaults:
  - USE_WANDB=0
  - WANDB_MODE=offline
  - WANDB_WATCH=none
  - COMPILE_STRATEGY=model
  - TRAIN_BATCH_TOKENS=524288
  - TRAIN_SEQ_LEN=2048
  - ITERATIONS=24
  - WARMUP_STEPS=20
  - PROFILE=1
  - PROFILE_RANGES=1
  - PROFILE_WAIT=5
  - PROFILE_WARMUP=3
  - PROFILE_ACTIVE=4
  - PROFILE_REPEAT=1
  - PROFILE_STOP_ON_COMPLETE=1
  - PERF_ISOLATE_COMPILE_CACHE=1
  - VAL_LOSS_EVERY=0
  - PERF_SKIP_FINAL_EVAL=1

Outputs:
  - Chrome/Perfetto traces under profiles/<run_id>/traces/
  - Operator summary under profiles/<run_id>/key_averages.txt

Examples:
  scripts/run_h100_single_gpu_hgdn_profile.sh hybrid
  RUN_PREFIX=h100prof scripts/run_h100_single_gpu_hgdn_profile.sh both
  RUN_PREFIX=h100prof USE_WANDB=1 WANDB_MODE=online scripts/run_h100_single_gpu_hgdn_profile.sh both
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
        export USE_WANDB="${USE_WANDB:-0}"
        export WANDB_MODE="${WANDB_MODE:-offline}"
        export WANDB_WATCH="${WANDB_WATCH:-none}"
        export COMPILE_STRATEGY="${COMPILE_STRATEGY:-model}"
        export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
        export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
        export ITERATIONS="${ITERATIONS:-24}"
        export MAX_WALLCLOCK_SECONDS=0
        export WARMUP_STEPS="${WARMUP_STEPS:-20}"
        export VAL_LOSS_EVERY=0
        export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
        export PERF_SKIP_FINAL_EVAL=1
        export PERF_ISOLATE_COMPILE_CACHE="${PERF_ISOLATE_COMPILE_CACHE:-1}"
        export PROFILE=1
        export PROFILE_DIR="${PROFILE_DIR:-./profiles}"
        export PROFILE_RANGES="${PROFILE_RANGES:-1}"
        export PROFILE_WAIT="${PROFILE_WAIT:-5}"
        export PROFILE_WARMUP="${PROFILE_WARMUP:-3}"
        export PROFILE_ACTIVE="${PROFILE_ACTIVE:-4}"
        export PROFILE_REPEAT="${PROFILE_REPEAT:-1}"
        export PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-1}"
        export PROFILE_MEMORY="${PROFILE_MEMORY:-1}"
        export PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-0}"
        export PROFILE_WITH_FLOPS="${PROFILE_WITH_FLOPS:-0}"
        export PROFILE_WITH_MODULES="${PROFILE_WITH_MODULES:-0}"
        export PROFILE_ROW_LIMIT="${PROFILE_ROW_LIMIT:-60}"
        export PROFILE_SORT_BY="${PROFILE_SORT_BY:-self_cuda_time_total}"
        export PROFILE_STOP_ON_COMPLETE="${PROFILE_STOP_ON_COMPLETE:-1}"
        for kv in "$@"; do
            export "$kv"
        done
        bash "$repo_root/scripts/sweep.sh" "$preset"
    )
}

run_hybrid() {
    local prefix="$1"
    local hybrid_gdn_ratio="${HYBRID_GDN_RATIO:-1}"
    local hybrid_mlp_mult="${HYBRID_MLP_MULT:-3.25}"
    local seq_len="${TRAIN_SEQ_LEN:-2048}"

    run_sweep \
        "1xH100 profile: hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single \
        "RUN_ID=${prefix}_profile_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${seq_len}" \
        "GDN_RATIO=${hybrid_gdn_ratio}" \
        "MLP_MULT=${hybrid_mlp_mult}"
}

run_depth() {
    local prefix="$1"
    local depth_mlp_mult="${DEPTH_MLP_MULT:-4.0}"
    local seq_len="${TRAIN_SEQ_LEN:-2048}"

    run_sweep \
        "1xH100 profile: depth MLP_MULT=${depth_mlp_mult}" \
        depth \
        "RUN_ID=${prefix}_profile_depth_mlp${depth_mlp_mult}_seq${seq_len}" \
        "MLP_MULT=${depth_mlp_mult}"
}

main() {
    require_cmd bash
    local run_prefix="${RUN_PREFIX:-h100_hgdn}"
    case "$mode" in
    hybrid)
        run_hybrid "$run_prefix"
        ;;
    depth)
        run_depth "$run_prefix"
        ;;
    both)
        run_hybrid "$run_prefix"
        run_depth "$run_prefix"
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
}

main "$@"
