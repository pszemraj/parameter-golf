#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It always executes the full local HGDN naive-contract search batch." >&2
    exit 1
fi

hgdn_require_cmd bash
hgdn_require_cmd torchrun
hgdn_require_cmd python

python_bin="${PYTHON_BIN:-python}"
use_wandb="${USE_WANDB:-1}"
wandb_mode="${WANDB_MODE:-online}"
wandb_project="${WANDB_PROJECT:-pg-hgdn-ablations}"
wandb_watch="${WANDB_WATCH:-none}"
wandb_watch_log_freq="${WANDB_WATCH_LOG_FREQ:-25}"
run_prefix_base="${RUN_PREFIX_BASE:-localnaivehgdn1}"
bundle_stage_dir="${BUNDLE_STAGE_DIR:-local-scratch/${run_prefix_base}_bundle}"
archive_output="${ARCHIVE_OUTPUT:-local-scratch/${run_prefix_base}_bundle.7z}"
command_log="${COMMAND_LOG:-local-scratch/${run_prefix_base}_commands.sh}"
size_screen_output="${SIZE_SCREEN_OUTPUT:-local-scratch/${run_prefix_base}_size_screen}"
size_screen_config="${SIZE_SCREEN_CONFIG:-configs/hgdn/naive_contract_search.toml}"
torchinductor_max_autotune="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
torchinductor_max_autotune_gemm="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-0}"
torch_logs="${TORCH_LOGS:-}"
torch_trace="${TORCH_TRACE:-}"

ngpu="${NGPU:-1}"
iterations="${ITERATIONS:-500}"
train_batch_tokens="${TRAIN_BATCH_TOKENS:-65536}"
train_seq_len="${TRAIN_SEQ_LEN:-1024}"
val_loss_every="${VAL_LOSS_EVERY:-100}"
train_log_every="${TRAIN_LOG_EVERY:-25}"
max_wallclock_seconds="${MAX_WALLCLOCK_SECONDS:-0}"
compile="${COMPILE:-1}"
compile_strategy="${COMPILE_STRATEGY:-hybrid}"
weight_decay="${WEIGHT_DECAY:-0}"
perf_skip_final_eval="${PERF_SKIP_FINAL_EVAL:-1}"
grad_accum_steps_override="${GRAD_ACCUM_STEPS:-}"

hgdn_ensure_python_module "${python_bin}" py7zr py7zr

resolve_grad_accum_steps() {
    if [[ -n "${grad_accum_steps_override}" ]]; then
        if (( grad_accum_steps_override < 1 )); then
            echo "GRAD_ACCUM_STEPS must be >= 1, got ${grad_accum_steps_override}" >&2
            exit 1
        fi
        echo "${grad_accum_steps_override}"
        return
    fi
    if (( 8 % ngpu != 0 )); then
        echo "NGPU must evenly divide 8 when GRAD_ACCUM_STEPS is not set: NGPU=${ngpu}" >&2
        exit 1
    fi
    echo $((8 / ngpu))
}

grad_accum_steps="$(resolve_grad_accum_steps)"
min_val_batch_size=$((ngpu * grad_accum_steps * train_seq_len))
val_batch_size="${VAL_BATCH_SIZE:-${min_val_batch_size}}"
if (( val_batch_size < min_val_batch_size )); then
    echo "VAL_BATCH_SIZE must be at least ${min_val_batch_size} for NGPU=${ngpu}, GRAD_ACCUM_STEPS=${grad_accum_steps}, TRAIN_SEQ_LEN=${train_seq_len}" >&2
    exit 1
fi

case "${wandb_mode}" in
online | offline) ;;
*)
    echo "Unsupported WANDB_MODE: ${wandb_mode} (expected online or offline)" >&2
    exit 1
    ;;
esac

default_prefixes=(
    "${run_prefix_base}_a"
    "${run_prefix_base}_b"
    "${run_prefix_base}_c"
    "${run_prefix_base}_d"
    "${run_prefix_base}_e"
    "${run_prefix_base}_f"
)

if [[ -n "${RUN_PREFIXES:-}" ]]; then
    IFS=',' read -r -a run_prefixes <<<"${RUN_PREFIXES}"
else
    run_prefixes=("${default_prefixes[@]}")
fi

configs=(
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml"
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m1p75.toml"
    "configs/hgdn/naive_contract_l9_d512_mid2_dk48_m1p75.toml"
    "configs/hgdn/naive_contract_l9_d512_mid3_dk48_m1p75.toml"
    "configs/hgdn/naive_contract_l8_d512_r0_m2.toml"
    "configs/hgdn/naive_contract_l9_d512_r0_m1p75.toml"
)

labels=(
    "HGDN 8Lx512d mid2 dk48 mlp2.0"
    "HGDN 8Lx512d mid2 dk48 mlp1.75"
    "HGDN 9Lx512d mid2 dk48 mlp1.75"
    "HGDN 9Lx512d mid3 dk48 mlp1.75"
    "Attention-only 8Lx512d mlp2.0"
    "Attention-only 9Lx512d mlp1.75"
)

if [[ -n "${CANDIDATE_INDEXES:-}" ]]; then
    IFS=',' read -r -a selected_indexes <<<"${CANDIDATE_INDEXES}"
    selected_run_prefixes=()
    selected_configs=()
    selected_labels=()
    for raw_index in "${selected_indexes[@]}"; do
        if [[ ! "${raw_index}" =~ ^[0-9]+$ ]]; then
            echo "CANDIDATE_INDEXES entries must be zero-based integers: ${raw_index}" >&2
            exit 1
        fi
        if (( raw_index < 0 || raw_index >= ${#configs[@]} )); then
            echo "CANDIDATE_INDEXES entry out of range: ${raw_index}" >&2
            exit 1
        fi
        selected_run_prefixes+=("${run_prefixes[$raw_index]}")
        selected_configs+=("${configs[$raw_index]}")
        selected_labels+=("${labels[$raw_index]}")
    done
    run_prefixes=("${selected_run_prefixes[@]}")
    configs=("${selected_configs[@]}")
    labels=("${selected_labels[@]}")
fi

if [[ "${#run_prefixes[@]}" -ne "${#configs[@]}" ]]; then
    echo "RUN_PREFIXES count (${#run_prefixes[@]}) must match config count (${#configs[@]})." >&2
    exit 1
fi

resolved_run_ids=()

load_config_env() {
    local config_path="$1"
    mapfile -t config_env < <(
        "${python_bin}" scripts/hgdn_helper_cli.py load-env --path "${config_path}"
    )
}

print_plan() {
    echo
    echo ">>> Local HGDN naive-contract search (sparse exact-contract candidate screen)"
    echo "python_bin=${python_bin}"
    echo "use_wandb=${use_wandb}"
    echo "wandb_mode=${wandb_mode}"
    echo "wandb_project=${wandb_project}"
    echo "wandb_watch=${wandb_watch}"
    echo "wandb_watch_log_freq=${wandb_watch_log_freq}"
    echo "TORCH_LOGS=${torch_logs:-<unset>}"
    echo "TORCH_TRACE=${torch_trace:-<unset>}"
    echo "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}"
    echo "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}"
    echo "ngpu=${ngpu}"
    echo "grad_accum_steps=${grad_accum_steps}"
    echo "iterations=${iterations}"
    echo "train_batch_tokens=${train_batch_tokens}"
    echo "train_seq_len=${train_seq_len}"
    echo "val_loss_every=${val_loss_every}"
    echo "train_log_every=${train_log_every}"
    echo "val_batch_size=${val_batch_size}"
    echo "weight_decay=${weight_decay}"
    echo "perf_skip_final_eval=${perf_skip_final_eval}"
    echo "compile=${compile}"
    echo "compile_strategy=${compile_strategy}"
    echo "max_wallclock_seconds=${max_wallclock_seconds}"
    echo "size_screen_config=${size_screen_config}"
    echo "size_screen_output=${size_screen_output}"
    echo "archive_output=${archive_output}"
    echo "batch:"
    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        echo "  - ${run_prefixes[$i]} :: ${labels[$i]} :: ${configs[$i]}"
    done
}

run_size_screen() {
    mkdir -p "$(dirname "${command_log}")"
    {
        echo "#!/bin/bash"
        echo "set -euo pipefail"
    } >"${command_log}"

    hgdn_append_plain_command \
        "${command_log}" \
        "${python_bin}" scripts/screen_hgdn_arch_sizes.py \
        --config "${size_screen_config}" \
        --output-dir "${size_screen_output}"

    echo
    echo ">>> artifact-size screen"
    "${python_bin}" scripts/screen_hgdn_arch_sizes.py \
        --config "${size_screen_config}" \
        --output-dir "${size_screen_output}"
}

run_batch() {
    local diagnostic_env=()
    if [[ -n "${torch_logs}" ]]; then
        diagnostic_env+=("TORCH_LOGS=${torch_logs}")
    fi
    if [[ -n "${torch_trace}" ]]; then
        diagnostic_env+=("TORCH_TRACE=${torch_trace}")
    fi

    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        local config_path="${configs[$i]}"
        local config_stem
        config_stem="$(basename "${config_path}" .toml)"
        local run_id="${run_prefixes[$i]}_${config_stem}_seq${train_seq_len}_it${iterations}"

        load_config_env "${config_path}"
        resolved_run_ids+=("${run_id}")

        echo
        echo ">>> local naive-contract search: ${labels[$i]}"
        echo "run_id=${run_id}"

        hgdn_append_command \
            "${command_log}" \
            "NGPU=${ngpu}" \
            "USE_WANDB=${use_wandb}" \
            "WANDB_MODE=${wandb_mode}" \
            "WANDB_PROJECT=${wandb_project}" \
            "WANDB_WATCH=${wandb_watch}" \
            "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
            "${diagnostic_env[@]}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}" \
            "COMPILE=${compile}" \
            "COMPILE_STRATEGY=${compile_strategy}" \
            "RUN_ID=${run_id}" \
            "GRAD_ACCUM_STEPS=${grad_accum_steps}" \
            "ITERATIONS=${iterations}" \
            "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
            "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
            "TRAIN_SEQ_LEN=${train_seq_len}" \
            "VAL_LOSS_EVERY=${val_loss_every}" \
            "TRAIN_LOG_EVERY=${train_log_every}" \
            "VAL_BATCH_SIZE=${val_batch_size}" \
            "WEIGHT_DECAY=${weight_decay}" \
            "PERF_SKIP_FINAL_EVAL=${perf_skip_final_eval}" \
            "${config_env[@]}" \
            bash scripts/sweep.sh single

        hgdn_run_sweep \
            "local naive-contract search: ${labels[$i]}" \
            single \
            "NGPU=${ngpu}" \
            "USE_WANDB=${use_wandb}" \
            "WANDB_MODE=${wandb_mode}" \
            "WANDB_PROJECT=${wandb_project}" \
            "WANDB_WATCH=${wandb_watch}" \
            "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
            "${diagnostic_env[@]}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}" \
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}" \
            "COMPILE=${compile}" \
            "COMPILE_STRATEGY=${compile_strategy}" \
            "RUN_ID=${run_id}" \
            "GRAD_ACCUM_STEPS=${grad_accum_steps}" \
            "ITERATIONS=${iterations}" \
            "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
            "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
            "TRAIN_SEQ_LEN=${train_seq_len}" \
            "VAL_LOSS_EVERY=${val_loss_every}" \
            "TRAIN_LOG_EVERY=${train_log_every}" \
            "VAL_BATCH_SIZE=${val_batch_size}" \
            "WEIGHT_DECAY=${weight_decay}" \
            "PERF_SKIP_FINAL_EVAL=${perf_skip_final_eval}" \
            "${config_env[@]}"
    done
}

build_bundle() {
    echo
    echo ">>> bundle outputs"

    rm -rf "${bundle_stage_dir}"
    mkdir -p \
        "${bundle_stage_dir}/logs" \
        "${bundle_stage_dir}/configs" \
        "${bundle_stage_dir}/size_screen"

    local config_path
    for config_path in "${configs[@]}" "${size_screen_config}"; do
        cp "${config_path}" "${bundle_stage_dir}/configs/"
    done

    cp "${command_log}" "${bundle_stage_dir}/commands.sh"
    cp -R "${size_screen_output}/." "${bundle_stage_dir}/size_screen/"

    local matched_logs=0
    local run_id
    for run_id in "${resolved_run_ids[@]}"; do
        local log_path="logs/${run_id}.txt"
        if [[ -f "${log_path}" ]]; then
            cp "${log_path}" "${bundle_stage_dir}/logs/"
            matched_logs=1
        fi
    done

    local -a manifest_cmd
    manifest_cmd=(
        "${python_bin}" scripts/hgdn_helper_cli.py write-local-naive-contract-search-manifest
        --output "${bundle_stage_dir}/bundle_manifest.json"
        --run-prefix-base "${run_prefix_base}"
        --wandb-project "${wandb_project}"
        --wandb-mode "${wandb_mode}"
        --archive-output "${archive_output}"
        --matched-logs "${matched_logs}"
        --size-screen-config "${size_screen_config}"
        --size-screen-output "${size_screen_output}"
        --torch-logs "${torch_logs}"
        --torch-trace "${torch_trace}"
        --torchinductor-max-autotune "${torchinductor_max_autotune}"
        --torchinductor-max-autotune-gemm "${torchinductor_max_autotune_gemm}"
        --iterations "${iterations}"
        --train-batch-tokens "${train_batch_tokens}"
        --train-seq-len "${train_seq_len}"
        --val-loss-every "${val_loss_every}"
        --train-log-every "${train_log_every}"
        --val-batch-size "${val_batch_size}"
        --max-wallclock-seconds "${max_wallclock_seconds}"
        --weight-decay "${weight_decay}"
        --perf-skip-final-eval "${perf_skip_final_eval}"
        --compile-enabled "${compile}"
        --compile-strategy "${compile_strategy}"
        --muon-distributed-mode "packed_allreduce"
        --gdn-w-g-optimizer "per_config"
    )
    local i
    for ((i = 0; i < ${#configs[@]}; i++)); do
        manifest_cmd+=(--config "${configs[$i]}")
        manifest_cmd+=(--label "${labels[$i]}")
    done
    local run_id
    for run_id in "${resolved_run_ids[@]}"; do
        manifest_cmd+=(--run-id "${run_id}")
    done
    "${manifest_cmd[@]}"

    hgdn_create_7z_archive "${python_bin}" "${archive_output}" "${bundle_stage_dir}"
    echo "bundle_archive=${archive_output}"
}

print_plan
run_size_screen
run_batch
build_bundle
