#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

mode="${1:-hybrid}"
python_bin="${PYTHON_BIN:-python}"
run_prefix="${RUN_PREFIX:-h100_hgdn}"
bundle_name="${run_prefix}_${mode}"
output_dir="${HPROF_OUTPUT_DIR:-artifacts/hgdn_profile/${bundle_name}}"
archive_output="${HPROF_ARCHIVE_OUTPUT:-${output_dir}.7z}"
command_log="${COMMAND_LOG:-${output_dir}/commands.sh}"
metadata_file="${output_dir}/metadata.txt"
_hprof_bundle_done=0
_hprof_exit_status=0
bundle_run_ids=()
bundle_run_labels=()
bundle_run_presets=()

usage() {
    cat <<'EOF'
Usage: scripts/run_h100_single_gpu_hgdn_profile.sh {hybrid|depth|both|hybrid-eager|depth-eager|both-eager|help}

Purpose:
  Run a short 1xH100 profiling capture for the current HGDN finalist pair.
  This is for trace collection, not leaderboard-quality training.

Modes:
  hybrid
    Profile the current hybrid operating point:
    - GDN_RATIO=1
    - MLP_MULT=3.25

  depth
    Profile the attention-only baseline:
    - GDN_RATIO=0 via the `depth` preset
    - MLP_MULT=4.0

  both
    Run hybrid first, then depth.

  hybrid-eager
    Profile the current hybrid operating point with `COMPILE=0`.
    Use this for attribution when compiled traces swallow `record_function` labels.

  depth-eager
    Profile the attention-only baseline with `COMPILE=0`.

  both-eager
    Run eager hybrid first, then eager depth.

Defaults:
  - USE_WANDB=0
  - WANDB_MODE=offline
  - WANDB_WATCH=none
  - SEED=1337
  - PYTHONHASHSEED=1337
  - CUDNN_BENCHMARK=0
  - COMPILE_STRATEGY=hybrid
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
  - GDN_LOG_LAYOUTS=0
  - GDN_LOG_LAYOUTS_LIMIT=1

Outputs:
  - Chrome/Perfetto traces under profiles/<run_id>/traces/
  - Operator summary under profiles/<run_id>/key_averages.json and key_averages.csv
  - Auto-archived helper bundle at `HPROF_ARCHIVE_OUTPUT` (defaults to
    `artifacts/hgdn_profile/<RUN_PREFIX>_<mode>.7z`)

Examples:
  scripts/run_h100_single_gpu_hgdn_profile.sh hybrid
  RUN_PREFIX=h100prof scripts/run_h100_single_gpu_hgdn_profile.sh both
  RUN_PREFIX=h100prof scripts/run_h100_single_gpu_hgdn_profile.sh both-eager
  RUN_PREFIX=h100prof USE_WANDB=1 WANDB_MODE=online scripts/run_h100_single_gpu_hgdn_profile.sh both
EOF
}

record_bundle_run() {
    local run_id="${1:?run id required}"
    local label="${2:?label required}"
    local preset="${3:?preset required}"
    bundle_run_ids+=("${run_id}")
    bundle_run_labels+=("${label}")
    bundle_run_presets+=("${preset}")
}

run_sweep() {
    local label="$1"
    local preset="$2"
    shift 2
    hgdn_append_command \
        "${command_log}" \
        "NGPU=1" \
        "USE_WANDB=${USE_WANDB:-0}" \
        "WANDB_MODE=${WANDB_MODE:-offline}" \
        "WANDB_WATCH=${WANDB_WATCH:-none}" \
        "COMPILE_STRATEGY=${COMPILE_STRATEGY:-hybrid}" \
        "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288}" \
        "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-2048}" \
        "ITERATIONS=${ITERATIONS:-24}" \
        "MAX_WALLCLOCK_SECONDS=0" \
        "WARMUP_STEPS=${WARMUP_STEPS:-20}" \
        "VAL_LOSS_EVERY=0" \
        "TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-10}" \
        "PERF_SKIP_FINAL_EVAL=1" \
        "PERF_ISOLATE_COMPILE_CACHE=${PERF_ISOLATE_COMPILE_CACHE:-1}" \
        "PROFILE=1" \
        "PROFILE_DIR=${PROFILE_DIR:-./profiles}" \
        "PROFILE_RANGES=${PROFILE_RANGES:-1}" \
        "PROFILE_WAIT=${PROFILE_WAIT:-5}" \
        "PROFILE_WARMUP=${PROFILE_WARMUP:-3}" \
        "PROFILE_ACTIVE=${PROFILE_ACTIVE:-4}" \
        "PROFILE_REPEAT=${PROFILE_REPEAT:-1}" \
        "PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-1}" \
        "PROFILE_MEMORY=${PROFILE_MEMORY:-1}" \
        "PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-0}" \
        "PROFILE_WITH_FLOPS=${PROFILE_WITH_FLOPS:-0}" \
        "PROFILE_WITH_MODULES=${PROFILE_WITH_MODULES:-0}" \
        "PROFILE_ROW_LIMIT=${PROFILE_ROW_LIMIT:-60}" \
        "PROFILE_SORT_BY=${PROFILE_SORT_BY:-self_cuda_time_total}" \
        "PROFILE_STOP_ON_COMPLETE=${PROFILE_STOP_ON_COMPLETE:-1}" \
        "GDN_LOG_LAYOUTS=${GDN_LOG_LAYOUTS:-0}" \
        "GDN_LOG_LAYOUTS_LIMIT=${GDN_LOG_LAYOUTS_LIMIT:-1}" \
        "$@" \
        bash "$HGDN_REPO_ROOT/scripts/sweep.sh" "$preset"
    hgdn_run_sweep \
        "$label" \
        "$preset" \
        "NGPU=1" \
        "USE_WANDB=${USE_WANDB:-0}" \
        "WANDB_MODE=${WANDB_MODE:-offline}" \
        "WANDB_WATCH=${WANDB_WATCH:-none}" \
        "COMPILE_STRATEGY=${COMPILE_STRATEGY:-hybrid}" \
        "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288}" \
        "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-2048}" \
        "ITERATIONS=${ITERATIONS:-24}" \
        "MAX_WALLCLOCK_SECONDS=0" \
        "WARMUP_STEPS=${WARMUP_STEPS:-20}" \
        "VAL_LOSS_EVERY=0" \
        "TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-10}" \
        "PERF_SKIP_FINAL_EVAL=1" \
        "PERF_ISOLATE_COMPILE_CACHE=${PERF_ISOLATE_COMPILE_CACHE:-1}" \
        "PROFILE=1" \
        "PROFILE_DIR=${PROFILE_DIR:-./profiles}" \
        "PROFILE_RANGES=${PROFILE_RANGES:-1}" \
        "PROFILE_WAIT=${PROFILE_WAIT:-5}" \
        "PROFILE_WARMUP=${PROFILE_WARMUP:-3}" \
        "PROFILE_ACTIVE=${PROFILE_ACTIVE:-4}" \
        "PROFILE_REPEAT=${PROFILE_REPEAT:-1}" \
        "PROFILE_RECORD_SHAPES=${PROFILE_RECORD_SHAPES:-1}" \
        "PROFILE_MEMORY=${PROFILE_MEMORY:-1}" \
        "PROFILE_WITH_STACK=${PROFILE_WITH_STACK:-0}" \
        "PROFILE_WITH_FLOPS=${PROFILE_WITH_FLOPS:-0}" \
        "PROFILE_WITH_MODULES=${PROFILE_WITH_MODULES:-0}" \
        "PROFILE_ROW_LIMIT=${PROFILE_ROW_LIMIT:-60}" \
        "PROFILE_SORT_BY=${PROFILE_SORT_BY:-self_cuda_time_total}" \
        "PROFILE_STOP_ON_COMPLETE=${PROFILE_STOP_ON_COMPLETE:-1}" \
        "GDN_LOG_LAYOUTS=${GDN_LOG_LAYOUTS:-0}" \
        "GDN_LOG_LAYOUTS_LIMIT=${GDN_LOG_LAYOUTS_LIMIT:-1}" \
        "$@"
}

run_hybrid() {
    local prefix="$1"
    local compile_enabled="${2:-1}"
    local hybrid_gdn_ratio="${HYBRID_GDN_RATIO:-1}"
    local hybrid_mlp_mult="${HYBRID_MLP_MULT:-${MLP_MULT:-3.25}}"
    local seq_len="${TRAIN_SEQ_LEN:-2048}"
    local compile_label="compiled"
    if [[ "$compile_enabled" == "0" ]]; then
        compile_label="eager"
    fi
    local run_id="${prefix}_profile_${compile_label}_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${seq_len}"

    record_bundle_run \
        "${run_id}" \
        "hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult} (${compile_label})" \
        "single"
    run_sweep \
        "1xH100 profile (${compile_label}): hybrid GDN_RATIO=${hybrid_gdn_ratio} MLP_MULT=${hybrid_mlp_mult}" \
        single \
        "RUN_ID=${run_id}" \
        "COMPILE=${compile_enabled}" \
        "GDN_RATIO=${hybrid_gdn_ratio}" \
        "MLP_MULT=${hybrid_mlp_mult}"
}

run_depth() {
    local prefix="$1"
    local compile_enabled="${2:-1}"
    local depth_mlp_mult="${DEPTH_MLP_MULT:-4.0}"
    local seq_len="${TRAIN_SEQ_LEN:-2048}"
    local compile_label="compiled"
    if [[ "$compile_enabled" == "0" ]]; then
        compile_label="eager"
    fi
    local run_id="${prefix}_profile_${compile_label}_depth_mlp${depth_mlp_mult}_seq${seq_len}"

    record_bundle_run \
        "${run_id}" \
        "attention-only baseline MLP_MULT=${depth_mlp_mult} (${compile_label})" \
        "depth"
    run_sweep \
        "1xH100 profile (${compile_label}): attention-only baseline MLP_MULT=${depth_mlp_mult}" \
        depth \
        "RUN_ID=${run_id}" \
        "COMPILE=${compile_enabled}" \
        "MLP_MULT=${depth_mlp_mult}"
}

build_bundle() {
    local matched_logs=0
    local idx
    local log_path
    local profile_path

    mkdir -p "${output_dir}"
    rm -rf "${output_dir}/logs" "${output_dir}/profiles"
    mkdir -p "${output_dir}/logs" "${output_dir}/profiles"

    for idx in "${!bundle_run_ids[@]}"; do
        log_path="logs/${bundle_run_ids[$idx]}.txt"
        if [[ -f "${log_path}" ]]; then
            cp "${log_path}" "${output_dir}/logs/"
            matched_logs=1
        fi
        profile_path="${PROFILE_DIR:-./profiles}/${bundle_run_ids[$idx]}"
        if [[ -d "${profile_path}" ]]; then
            cp -R "${profile_path}" "${output_dir}/profiles/"
        fi
    done
    cp "${command_log}" "${output_dir}/commands.sh"
    {
        echo "{"
        echo "  \"mode\": \"${mode}\","
        echo "  \"run_prefix\": \"${run_prefix}\","
        echo "  \"archive_output\": \"${archive_output}\","
        echo "  \"command_log\": \"${command_log}\","
        echo "  \"matched_logs\": ${matched_logs},"
        echo "  \"contract\": {"
        echo "    \"compile_strategy_default\": \"${COMPILE_STRATEGY:-hybrid}\","
        echo "    \"train_batch_tokens\": ${TRAIN_BATCH_TOKENS:-524288},"
        echo "    \"train_seq_len\": ${TRAIN_SEQ_LEN:-2048},"
        echo "    \"iterations\": ${ITERATIONS:-24},"
        echo "    \"profile_dir\": \"${PROFILE_DIR:-./profiles}\""
        echo "  },"
        echo "  \"runs\": ["
        for idx in "${!bundle_run_ids[@]}"; do
            [[ "${idx}" -gt 0 ]] && echo ","
            printf '    {"label":"%s","preset":"%s","run_id":"%s"}' \
                "${bundle_run_labels[$idx]}" \
                "${bundle_run_presets[$idx]}" \
                "${bundle_run_ids[$idx]}"
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
    if [[ "${_hprof_bundle_done}" == "1" ]]; then
        return 0
    fi
    _hprof_bundle_done=1
    build_bundle || true
}

main() {
    hgdn_require_cmd bash
    hgdn_require_cmd "${python_bin}"
    case "$mode" in
    help | -h | --help)
        usage
        return 0
        ;;
    hybrid|depth|both|hybrid-eager|depth-eager|both-eager) ;;
    *)
        echo "Unknown mode: $mode" >&2
        usage >&2
        return 1
        ;;
    esac
    mkdir -p "${output_dir}"
    : > "${command_log}"
    chmod +x "${command_log}"
    cat > "${metadata_file}" <<EOF
mode=${mode}
run_prefix=${run_prefix}
python_bin=${python_bin}
hprof_output_dir=${output_dir}
hprof_archive_output=${archive_output}
compile_strategy_default=${COMPILE_STRATEGY:-hybrid}
use_wandb=${USE_WANDB:-0}
wandb_mode=${WANDB_MODE:-offline}
EOF
    hgdn_ensure_python_module "${python_bin}" py7zr py7zr
    trap '_hprof_exit_status=$?; printf "bundle_exit_status=%s\n" "${_hprof_exit_status}" >> "${metadata_file}"; build_bundle_once; exit ${_hprof_exit_status}' EXIT
    case "$mode" in
    hybrid)
        run_hybrid "$run_prefix" 1
        ;;
    depth)
        run_depth "$run_prefix" 1
        ;;
    both)
        run_hybrid "$run_prefix" 1
        run_depth "$run_prefix" 1
        ;;
    hybrid-eager)
        run_hybrid "$run_prefix" 0
        ;;
    depth-eager)
        run_depth "$run_prefix" 0
        ;;
    both-eager)
        run_hybrid "$run_prefix" 0
        run_depth "$run_prefix" 0
        ;;
    esac
}

main "$@"
