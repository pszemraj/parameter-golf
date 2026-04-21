#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

mode="${1:-perf}"
python_bin="${PYTHON_BIN:-python}"
attn_use_flash_attn3="${ATTN_USE_FLASH_ATTN3:-1}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_prefix="${RUN_PREFIX:-h1001_${run_stamp}}"
bundle_name="${run_prefix}_${mode}"
output_dir="${HP_OUTPUT_DIR:-artifacts/hgdn_single_gpu/${bundle_name}}"
archive_output="${HP_ARCHIVE_OUTPUT:-${output_dir}.7z}"
command_log="${COMMAND_LOG:-${output_dir}/commands.sh}"
metadata_file="${output_dir}/metadata.txt"
git_commit="$(git rev-parse HEAD)"
git_branch="$(git rev-parse --abbrev-ref HEAD)"
host_name="$(hostname)"
timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
_hp_bundle_done=0
_hp_exit_status=0
bundle_run_ids=()
bundle_run_labels=()
bundle_run_presets=()
bundle_run_compile_strategies=()
bundle_configs=()

usage() {
    cat <<'EOF'
Usage: scripts/run_h100_single_gpu_hgdn.sh {perf|fixed2k|fixed2k-hybrid|fixed2k-hybrid-compile-matrix|all|help}

Purpose:
  Single-GPU H100 helper for HGDN target-hardware calibration.
  This script is intentionally 1xH100-only. It is for measuring H100 behavior as
  an accelerator, not for reproducing the 8xH100 submission protocol.

Modes:
  perf
    Run the current finalist pair with the perf harness:
    - hybrid: live packed HGDN `14L x 384d x mlp3.25`
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
    The hybrid leg is pinned to the live packed `14L x 384d` shell.
    Contract also forces:
    - PERF_ISOLATE_COMPILE_CACHE=1

  fixed2k-hybrid-compile-matrix
    Run the same live packed `14L x 384d` hybrid leg under multiple compile
    strategies on one 1xH100 box.
    Default strategy list:
    - model
    - selective
    - hybrid
    Use this when the packed HGDN path needs a fair compile-placement check
    without manually reassembling three long commands.

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
  COMPILE_STRATEGY           Defaults to hybrid.
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
  PACKED_COMPILE_MATRIX_STRATEGIES
                             Space-delimited compile strategy list for
                             fixed2k-hybrid-compile-matrix, defaults to
                             "hybrid selective model".
  PYTHON_BIN                 Python binary used for py7zr bundle creation,
                             defaults to python.
  HP_OUTPUT_DIR              Packed helper bundle stage directory. Defaults to
                             artifacts/hgdn_single_gpu/<run_prefix>_<mode>.
  HP_ARCHIVE_OUTPUT          Packed helper `.7z` archive path. Defaults to
                             <HP_OUTPUT_DIR>.7z.
  COMMAND_LOG                Command log path. Defaults to
                             <HP_OUTPUT_DIR>/commands.sh.
  DRY_RUN                    If set to 1, print commands and bundle paths
                             without executing sweeps.

Examples:
  scripts/run_h100_single_gpu_hgdn.sh perf
  RUN_PREFIX=h100a scripts/run_h100_single_gpu_hgdn.sh fixed2k
  RUN_PREFIX=h100a scripts/run_h100_single_gpu_hgdn.sh fixed2k-hybrid
  RUN_PREFIX=h100a scripts/run_h100_single_gpu_hgdn.sh fixed2k-hybrid-compile-matrix
  USE_WANDB=0 WANDB_MODE=offline scripts/run_h100_single_gpu_hgdn.sh perf
EOF
}

case "$mode" in
perf|fixed2k|fixed2k-hybrid|fixed2k-hybrid-compile-matrix|all) ;;
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

record_bundle_run() {
    local label="${1:?label required}"
    local preset="${2:?preset required}"
    local run_id="${3:?run_id required}"
    local compile_strategy="${4:-}"
    bundle_run_labels+=("${label}")
    bundle_run_presets+=("${preset}")
    bundle_run_ids+=("${run_id}")
    bundle_run_compile_strategies+=("${compile_strategy}")
}

record_bundle_config() {
    local config_path="${1:?config path required}"
    local existing
    for existing in "${bundle_configs[@]:-}"; do
        if [[ "${existing}" == "${config_path}" ]]; then
            return 0
        fi
    done
    bundle_configs+=("${config_path}")
}

append_metadata_line() {
    printf '%s\n' "$1" >> "${metadata_file}"
}

run_sweep() {
    local label="$1"
    local preset="$2"
    shift 2
    hgdn_append_command "${command_log}" "$@" bash "$HGDN_REPO_ROOT/scripts/sweep.sh" "${preset}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo
        printf '>>> '
        printf '%q ' "$@" bash "$HGDN_REPO_ROOT/scripts/sweep.sh" "${preset}"
        printf '\n'
        return 0
    fi
    hgdn_run_sweep \
        "$label" \
        "$preset" \
        "NGPU=1" \
        "USE_WANDB=${USE_WANDB:-1}" \
        "WANDB_MODE=${WANDB_MODE:-online}" \
        "WANDB_WATCH=${WANDB_WATCH:-none}" \
        "ATTN_USE_FLASH_ATTN3=${attn_use_flash_attn3}" \
        "COMPILE_STRATEGY=${COMPILE_STRATEGY:-hybrid}" \
        "$@"
}

build_bundle() {
    local matched_logs=0
    local idx
    local log_path
    local config_path
    local manifest_iterations
    local manifest_seq_len
    local manifest_val_loss_every
    local manifest_train_log_every
    local manifest_train_batch_tokens

    case "${mode}" in
    perf)
        manifest_iterations="${PERF_ITERATIONS:-50}"
        manifest_seq_len="${PERF_SEQ_LEN:-2048}"
        manifest_val_loss_every=0
        manifest_train_log_every="${PERF_TRAIN_LOG_EVERY:-10}"
        manifest_train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
        ;;
    fixed2k|fixed2k-hybrid|fixed2k-hybrid-compile-matrix|all)
        manifest_iterations="${FIXED2K_ITERATIONS:-2000}"
        manifest_seq_len="${FIXED2K_SEQ_LEN:-2048}"
        manifest_val_loss_every="${FIXED2K_VAL_LOSS_EVERY:-500}"
        manifest_train_log_every="${FIXED2K_TRAIN_LOG_EVERY:-200}"
        manifest_train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
        ;;
    esac

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo
        echo "bundle_dir=${output_dir}"
        echo "bundle_archive=${archive_output}"
        return 0
    fi

    mkdir -p "${output_dir}"
    rm -rf "${output_dir}/logs" "${output_dir}/configs"
    mkdir -p "${output_dir}/logs" "${output_dir}/configs"

    for idx in "${!bundle_run_ids[@]}"; do
        log_path="logs/${bundle_run_ids[$idx]}.txt"
        if [[ -f "${log_path}" ]]; then
            cp "${log_path}" "${output_dir}/logs/"
            matched_logs=1
        fi
    done

    for config_path in "${bundle_configs[@]:-}"; do
        if [[ -f "${config_path}" ]]; then
            cp "${config_path}" "${output_dir}/configs/"
        fi
    done

    {
        echo "{"
        echo "  \"mode\": \"${mode}\","
        echo "  \"run_prefix\": \"${run_prefix}\","
        echo "  \"archive_output\": \"${archive_output}\","
        echo "  \"command_log\": \"${command_log}\","
        echo "  \"matched_logs\": ${matched_logs},"
        echo "  \"contract\": {"
        echo "    \"compile_strategy_default\": \"${COMPILE_STRATEGY:-hybrid}\","
        echo "    \"train_batch_tokens\": ${manifest_train_batch_tokens},"
        echo "    \"train_seq_len\": ${manifest_seq_len},"
        echo "    \"iterations\": ${manifest_iterations},"
        echo "    \"val_loss_every\": ${manifest_val_loss_every},"
        echo "    \"train_log_every\": ${manifest_train_log_every}"
        echo "  },"
        echo "  \"runs\": ["
        for idx in "${!bundle_run_ids[@]}"; do
            [[ "${idx}" -gt 0 ]] && echo ","
            printf '    {"label":"%s","preset":"%s","run_id":"%s"' \
                "${bundle_run_labels[$idx]}" \
                "${bundle_run_presets[$idx]}" \
                "${bundle_run_ids[$idx]}"
            if [[ -n "${bundle_run_compile_strategies[$idx]}" ]]; then
                printf ',"compile_strategy":"%s"' "${bundle_run_compile_strategies[$idx]}"
            fi
            printf '}'
        done
        echo
        echo "  ]"
        echo "}"
    } > "${output_dir}/bundle_manifest.json"

    hgdn_create_7z_archive "${python_bin}" "${archive_output}" "${output_dir}"
    echo
    echo "bundle_dir=${output_dir}"
    echo "bundle_archive=${archive_output}"
}

build_bundle_once() {
    if [[ "${_hp_bundle_done}" == "1" ]]; then
        return 0
    fi
    _hp_bundle_done=1
    build_bundle || true
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
    local hybrid_run_id="${prefix}_perf_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${perf_seq_len}"
    local depth_run_id="${prefix}_perf_depth_mlp${depth_mlp_mult}_seq${perf_seq_len}"

    record_bundle_run \
        "hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        "single-live14" \
        "${hybrid_run_id}" \
        "${COMPILE_STRATEGY:-hybrid}"
    record_bundle_run \
        "attention-only baseline MLP_MULT=${depth_mlp_mult}" \
        "depth" \
        "${depth_run_id}" \
        "${COMPILE_STRATEGY:-hybrid}"
    record_bundle_config "${HGDN_REPO_ROOT}/configs/hgdn/winner_20260405_19_live14.toml"

    run_sweep \
        "1xH100 perf: hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single-live14 \
        "RUN_ID=${hybrid_run_id}" \
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
        "RUN_ID=${depth_run_id}" \
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
    local hybrid_run_id="${prefix}_fixed2k_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${seq_len}"
    local depth_run_id="${prefix}_fixed2k_depth_mlp${depth_mlp_mult}_seq${seq_len}"

    record_bundle_run \
        "hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        "single-live14" \
        "${hybrid_run_id}" \
        "${COMPILE_STRATEGY:-hybrid}"
    record_bundle_run \
        "attention-only baseline MLP_MULT=${depth_mlp_mult}" \
        "depth" \
        "${depth_run_id}" \
        "${COMPILE_STRATEGY:-hybrid}"
    record_bundle_config "${HGDN_REPO_ROOT}/configs/hgdn/winner_20260405_19_live14.toml"

    run_sweep \
        "1xH100 fixed2k: hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single-live14 \
        "RUN_ID=${hybrid_run_id}" \
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
        "RUN_ID=${depth_run_id}" \
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
    local hybrid_run_id="${prefix}_fixed2k_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${seq_len}"

    record_bundle_run \
        "hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        "single-live14" \
        "${hybrid_run_id}" \
        "${COMPILE_STRATEGY:-hybrid}"
    record_bundle_config "${HGDN_REPO_ROOT}/configs/hgdn/winner_20260405_19_live14.toml"

    run_sweep \
        "1xH100 fixed2k: hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single-live14 \
        "RUN_ID=${hybrid_run_id}" \
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

run_fixed2k_hybrid_compile_matrix() {
    local prefix="$1"
    local strategies="${PACKED_COMPILE_MATRIX_STRATEGIES:-hybrid selective model}"
    local strategy

    for strategy in ${strategies}; do
        local run_id="${prefix}_fixed2k_hybrid_compile_${strategy}"
        record_bundle_run \
            "hybrid packed COMPILE_STRATEGY=${strategy}" \
            "single-live14" \
            "${run_id}" \
            "${strategy}"
        record_bundle_config "${HGDN_REPO_ROOT}/configs/hgdn/winner_20260405_19_live14.toml"
        run_sweep \
            "1xH100 fixed2k compile-matrix: hybrid packed COMPILE_STRATEGY=${strategy}" \
            single-live14 \
            "RUN_ID=${run_id}" \
            "COMPILE_STRATEGY=${strategy}" \
            "GDN_RATIO=${HYBRID_GDN_RATIO:-1}" \
            "MLP_MULT=${HYBRID_MLP_MULT:-${MLP_MULT:-3.25}}" \
            "ITERATIONS=${FIXED2K_ITERATIONS:-2000}" \
            "MAX_WALLCLOCK_SECONDS=0" \
            "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288}" \
            "TRAIN_SEQ_LEN=${FIXED2K_SEQ_LEN:-2048}" \
            "VAL_LOSS_EVERY=${FIXED2K_VAL_LOSS_EVERY:-500}" \
            "TRAIN_LOG_EVERY=${FIXED2K_TRAIN_LOG_EVERY:-200}" \
            "PERF_TIMING=0" \
            "PERF_ISOLATE_COMPILE_CACHE=1" \
            "PERF_SKIP_FINAL_EVAL=0"
    done
}

hgdn_require_cmd bash
hgdn_require_cmd torchrun
hgdn_require_cmd "${python_bin}"

mkdir -p "${output_dir}"
: > "${command_log}"
chmod +x "${command_log}"
cat > "${metadata_file}" <<EOF
mode=${mode}
run_prefix=${run_prefix}
python_bin=${python_bin}
hp_output_dir=${output_dir}
hp_archive_output=${archive_output}
compile_strategy_default=${COMPILE_STRATEGY:-hybrid}
use_wandb=${USE_WANDB:-1}
wandb_mode=${WANDB_MODE:-online}
git_commit=${git_commit}
git_branch=${git_branch}
host_name=${host_name}
timestamp_utc=${timestamp_utc}
EOF

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    hgdn_ensure_python_module "${python_bin}" py7zr py7zr
fi

trap '_hp_exit_status=$?; append_metadata_line "bundle_exit_status=${_hp_exit_status}"; build_bundle_once; exit ${_hp_exit_status}' EXIT

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
fixed2k-hybrid-compile-matrix)
    run_fixed2k_hybrid_compile_matrix "$run_prefix"
    ;;
all)
    run_perf_pair "$run_prefix"
    run_fixed2k_pair "$run_prefix"
    ;;
esac
