#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

mode="${1:-perf}"

usage() {
    cat <<'EOF'
Usage: scripts/run_h100_single_gpu_hgdn.sh {perf|fixed2k|fixed2k-hybrid|all|help}

Purpose:
  Single-GPU H100 helper for HGDN target-hardware calibration.
  This script is intentionally 1xH100-only. It is for measuring H100 behavior as
  an accelerator, not for reproducing the 8xH100 submission protocol.

Modes:
  perf
    Run the current finalist pair with the perf harness:
    - hybrid: GDN_RATIO=1, MLP_MULT=3.25
    - attention-only baseline: MLP_MULT=4.0
    Contract:
    - ITERATIONS=50
    - MAX_WALLCLOCK_SECONDS=0
    - VAL_LOSS_EVERY=0
    - PERF_TIMING=1
    - PERF_IGNORE_STEPS=10
    - PERF_ISOLATE_COMPILE_CACHE=1
    - PERF_SKIP_FINAL_EVAL=1

  fixed2k
    Run the same pair under a fixed-step quality contract:
    - ITERATIONS=2000
    - MAX_WALLCLOCK_SECONDS=0
    - TRAIN_SEQ_LEN=2048
    - TRAIN_BATCH_TOKENS=524288
    - VAL_LOSS_EVERY=500
    - PERF_ISOLATE_COMPILE_CACHE=1

  fixed2k-hybrid
    Run only the hybrid side of the same fixed-step quality contract.
    Use this for architecture retune candidates after the attention-only
    baseline has already been established separately.
    Contract also forces:
    - PERF_ISOLATE_COMPILE_CACHE=1

  all
    Run perf first, then fixed2k.

Environment overrides:
  RUN_PREFIX                 Base prefix for run ids.
  SEED                       Defaults to 1337.
  PYTHONHASHSEED             Defaults to SEED.
  CUDNN_BENCHMARK            Defaults to 0.
  USE_WANDB                  Defaults to 1.
  WANDB_MODE                 Defaults to online.
  WANDB_WATCH                Defaults to none.
  WANDB_WATCH_LOG_FREQ       Defaults to trainer default.
  WANDB_PROJECT              Passed through to sweep.sh if enabled.
  DATA_PATH                  Passed through to sweep.sh.
  TOKENIZER_PATH             Passed through to sweep.sh.
  COMPILE_STRATEGY           Defaults to model.
  TRAIN_BATCH_TOKENS         Defaults to 524288.
  HYBRID_GDN_RATIO           Defaults to 1.
  HYBRID_MLP_MULT            Defaults to 3.25.
  DEPTH_MLP_MULT             Defaults to 4.0.
  PERF_SEQ_LEN               Defaults to 2048.
  PERF_ITERATIONS            Defaults to 50.
  PERF_IGNORE_STEPS          Defaults to 10.
  PERF_TRAIN_LOG_EVERY       Defaults to 10.
  FIXED2K_SEQ_LEN            Defaults to 2048.
  FIXED2K_ITERATIONS         Defaults to 2000.
  FIXED2K_VAL_LOSS_EVERY     Defaults to 500.
  FIXED2K_TRAIN_LOG_EVERY    Defaults to 200.

Examples:
  scripts/run_h100_single_gpu_hgdn.sh perf
  RUN_PREFIX=h100a scripts/run_h100_single_gpu_hgdn.sh fixed2k
  RUN_PREFIX=h100a scripts/run_h100_single_gpu_hgdn.sh fixed2k-hybrid
  USE_WANDB=0 WANDB_MODE=offline scripts/run_h100_single_gpu_hgdn.sh perf
EOF
}

run_sweep() {
    local label="$1"
    local preset="$2"
    shift 2
    hgdn_run_sweep \
        "$label" \
        "$preset" \
        "NGPU=1" \
        "USE_WANDB=${USE_WANDB:-1}" \
        "WANDB_MODE=${WANDB_MODE:-online}" \
        "WANDB_WATCH=${WANDB_WATCH:-none}" \
        "COMPILE_STRATEGY=${COMPILE_STRATEGY:-model}" \
        "$@"
}

run_perf_pair() {
    local prefix="$1"
    local train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
    local perf_seq_len="${PERF_SEQ_LEN:-2048}"
    local perf_iterations="${PERF_ITERATIONS:-50}"
    local perf_ignore_steps="${PERF_IGNORE_STEPS:-10}"
    local perf_train_log_every="${PERF_TRAIN_LOG_EVERY:-10}"
    local hybrid_gdn_ratio="${HYBRID_GDN_RATIO:-1}"
    local hybrid_mlp_mult="${HYBRID_MLP_MULT:-${MLP_MULT:-3.25}}"
    local depth_mlp_mult="${DEPTH_MLP_MULT:-4.0}"

    run_sweep \
        "1xH100 perf: hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single \
        "RUN_ID=${prefix}_perf_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${perf_seq_len}" \
        "GDN_RATIO=${hybrid_gdn_ratio}" \
        "MLP_MULT=${hybrid_mlp_mult}" \
        "ITERATIONS=${perf_iterations}" \
        "MAX_WALLCLOCK_SECONDS=0" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${perf_seq_len}" \
        "VAL_LOSS_EVERY=0" \
        "TRAIN_LOG_EVERY=${perf_train_log_every}" \
        "PERF_TIMING=1" \
        "PERF_IGNORE_STEPS=${perf_ignore_steps}" \
        "PERF_ISOLATE_COMPILE_CACHE=1" \
        "PERF_SKIP_FINAL_EVAL=1"

    run_sweep \
        "1xH100 perf: attention-only baseline MLP_MULT=${depth_mlp_mult}" \
        depth \
        "RUN_ID=${prefix}_perf_depth_mlp${depth_mlp_mult}_seq${perf_seq_len}" \
        "MLP_MULT=${depth_mlp_mult}" \
        "ITERATIONS=${perf_iterations}" \
        "MAX_WALLCLOCK_SECONDS=0" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${perf_seq_len}" \
        "VAL_LOSS_EVERY=0" \
        "TRAIN_LOG_EVERY=${perf_train_log_every}" \
        "PERF_TIMING=1" \
        "PERF_IGNORE_STEPS=${perf_ignore_steps}" \
        "PERF_ISOLATE_COMPILE_CACHE=1" \
        "PERF_SKIP_FINAL_EVAL=1"
}

run_fixed2k_pair() {
    local prefix="$1"
    local train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
    local seq_len="${FIXED2K_SEQ_LEN:-2048}"
    local iterations="${FIXED2K_ITERATIONS:-2000}"
    local val_loss_every="${FIXED2K_VAL_LOSS_EVERY:-500}"
    local train_log_every="${FIXED2K_TRAIN_LOG_EVERY:-200}"
    local hybrid_gdn_ratio="${HYBRID_GDN_RATIO:-1}"
    local hybrid_mlp_mult="${HYBRID_MLP_MULT:-${MLP_MULT:-3.25}}"
    local depth_mlp_mult="${DEPTH_MLP_MULT:-4.0}"

    run_sweep \
        "1xH100 fixed2k: hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single \
        "RUN_ID=${prefix}_fixed2k_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${seq_len}" \
        "GDN_RATIO=${hybrid_gdn_ratio}" \
        "MLP_MULT=${hybrid_mlp_mult}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=0" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "PERF_TIMING=0" \
        "PERF_ISOLATE_COMPILE_CACHE=1" \
        "PERF_SKIP_FINAL_EVAL=0"

    run_sweep \
        "1xH100 fixed2k: attention-only baseline MLP_MULT=${depth_mlp_mult}" \
        depth \
        "RUN_ID=${prefix}_fixed2k_depth_mlp${depth_mlp_mult}_seq${seq_len}" \
        "MLP_MULT=${depth_mlp_mult}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=0" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "PERF_TIMING=0" \
        "PERF_ISOLATE_COMPILE_CACHE=1" \
        "PERF_SKIP_FINAL_EVAL=0"
}

run_fixed2k_hybrid() {
    local prefix="$1"
    local train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
    local seq_len="${FIXED2K_SEQ_LEN:-2048}"
    local iterations="${FIXED2K_ITERATIONS:-2000}"
    local val_loss_every="${FIXED2K_VAL_LOSS_EVERY:-500}"
    local train_log_every="${FIXED2K_TRAIN_LOG_EVERY:-200}"
    local hybrid_gdn_ratio="${HYBRID_GDN_RATIO:-1}"
    local hybrid_mlp_mult="${HYBRID_MLP_MULT:-${MLP_MULT:-3.25}}"

    run_sweep \
        "1xH100 fixed2k: hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single \
        "RUN_ID=${prefix}_fixed2k_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${seq_len}" \
        "GDN_RATIO=${hybrid_gdn_ratio}" \
        "MLP_MULT=${hybrid_mlp_mult}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=0" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "PERF_TIMING=0" \
        "PERF_SKIP_FINAL_EVAL=0"
}

hgdn_require_cmd bash
hgdn_require_cmd torchrun

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_prefix="${RUN_PREFIX:-h1001_${run_stamp}}"

case "$mode" in
perf)
    run_perf_pair "$run_prefix"
    ;;
fixed2k)
    run_fixed2k_pair "$run_prefix"
    ;;
fixed2k-hybrid)
    run_fixed2k_hybrid "$run_prefix"
    ;;
all)
    run_perf_pair "$run_prefix"
    run_fixed2k_pair "$run_prefix"
    ;;
help|-h|--help)
    usage
    ;;
*)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 1
    ;;
esac
