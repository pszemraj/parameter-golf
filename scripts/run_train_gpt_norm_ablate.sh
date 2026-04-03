#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

mode="${1:-all}"

usage() {
    cat <<'EOF'
Usage: scripts/run_train_gpt_norm_ablate.sh {all|shortlist|help}

Purpose:
  Run a small decoder-only norm-shell ablation matrix on train_gpt.py.
  This is aimed at checking whether affine LayerNorm + a cleaner post-norm shell
  behaves differently from the current pre-RMSNorm baseline.

Modes:
  shortlist
    Run the three highest-signal configs:
    - current pre-RMSNorm baseline
    - pre-LayerNorm with the current shell
    - plain post-LayerNorm decoder shell

  all
    Run the fuller five-run matrix:
    - current pre-RMSNorm baseline
    - pre-LayerNorm with the current shell
    - post-LayerNorm with the current shell
    - plain post-LayerNorm decoder shell
    - KEEL-style LayerNorm decoder shell

Defaults:
  - NGPU=1
  - ITERATIONS=1000
  - MAX_WALLCLOCK_SECONDS=0
  - TRAIN_BATCH_TOKENS=524288
  - TRAIN_SEQ_LEN=1024
  - VAL_LOSS_EVERY=100
  - TRAIN_LOG_EVERY=25
  - COMPILE follows train_gpt.py defaults

Notes:
  - "current shell" keeps the baseline's input norm, final norm, residual mix, and skip weights.
  - "decoder shell" disables input norm, final norm, residual mix, and skip weights so the
    residual path is closer to a plain decoder-only post-norm stack.

Environment overrides:
  RUN_PREFIX             Base prefix for run ids.
  NGPU                   torchrun worker count, defaults to 1.
  ITERATIONS             Defaults to 1000.
  TRAIN_BATCH_TOKENS     Defaults to 524288.
  TRAIN_SEQ_LEN          Defaults to 1024.
  VAL_LOSS_EVERY         Defaults to 100.
  TRAIN_LOG_EVERY        Defaults to 25.
  DATA_PATH              Passed through to train_gpt.py.
  TOKENIZER_PATH         Passed through to train_gpt.py.
  TORCH_LOGS             Optional torch compile diagnostics.

Examples:
  scripts/run_train_gpt_norm_ablate.sh shortlist
  RUN_PREFIX=postln_h100 TRAIN_SEQ_LEN=2048 ITERATIONS=750 scripts/run_train_gpt_norm_ablate.sh all
  NGPU=8 RUN_PREFIX=postln_8gpu scripts/run_train_gpt_norm_ablate.sh shortlist
EOF
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

run_case() {
    local label="$1"
    shift

    local ngpu="${NGPU:-1}"
    local iterations="${ITERATIONS:-1000}"
    local train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
    local train_seq_len="${TRAIN_SEQ_LEN:-1024}"
    local val_loss_every="${VAL_LOSS_EVERY:-100}"
    local train_log_every="${TRAIN_LOG_EVERY:-25}"

    echo
    echo ">>> $label"
    (
        export RUN_ID="$label"
        export ITERATIONS="${iterations}"
        export MAX_WALLCLOCK_SECONDS=0
        export TRAIN_BATCH_TOKENS="${train_batch_tokens}"
        export TRAIN_SEQ_LEN="${train_seq_len}"
        export VAL_LOSS_EVERY="${val_loss_every}"
        export TRAIN_LOG_EVERY="${train_log_every}"
        for kv in "$@"; do
            export "$kv"
        done
        torchrun --standalone --nproc_per_node="${ngpu}" train_gpt.py
    )
}

run_shortlist() {
    local prefix="$1"
    local seq="${TRAIN_SEQ_LEN:-1024}"
    local iters="${ITERATIONS:-1000}"

    run_case \
        "${prefix}_current_pre_rms_seq${seq}_it${iters}" \
        "NORM_KIND=rms" \
        "NORM_STYLE=pre" \
        "INPUT_NORM=1" \
        "FINAL_NORM=1" \
        "USE_RESIDUAL_MIX=1" \
        "USE_SKIP_WEIGHTS=1"

    run_case \
        "${prefix}_pre_layer_current_seq${seq}_it${iters}" \
        "NORM_KIND=layer" \
        "NORM_STYLE=pre" \
        "INPUT_NORM=1" \
        "FINAL_NORM=1" \
        "USE_RESIDUAL_MIX=1" \
        "USE_SKIP_WEIGHTS=1"

    run_case \
        "${prefix}_post_layer_decoder_seq${seq}_it${iters}" \
        "NORM_KIND=layer" \
        "NORM_STYLE=post" \
        "INPUT_NORM=0" \
        "FINAL_NORM=0" \
        "USE_RESIDUAL_MIX=0" \
        "USE_SKIP_WEIGHTS=0"
}

run_all() {
    local prefix="$1"
    local seq="${TRAIN_SEQ_LEN:-1024}"
    local iters="${ITERATIONS:-1000}"

    run_shortlist "$prefix"

    run_case \
        "${prefix}_post_layer_current_seq${seq}_it${iters}" \
        "NORM_KIND=layer" \
        "NORM_STYLE=post" \
        "INPUT_NORM=1" \
        "FINAL_NORM=1" \
        "USE_RESIDUAL_MIX=1" \
        "USE_SKIP_WEIGHTS=1"

    run_case \
        "${prefix}_keel_layer_decoder_seq${seq}_it${iters}" \
        "NORM_KIND=layer" \
        "NORM_STYLE=keel" \
        "INPUT_NORM=0" \
        "FINAL_NORM=0" \
        "USE_RESIDUAL_MIX=0" \
        "USE_SKIP_WEIGHTS=0"
}

require_cmd torchrun

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_prefix="${RUN_PREFIX:-gpt_norm_${run_stamp}}"

case "$mode" in
shortlist)
    run_shortlist "$run_prefix"
    ;;
all)
    run_all "$run_prefix"
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
