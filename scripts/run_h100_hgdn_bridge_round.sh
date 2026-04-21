#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It always executes the full current exact-8x HGDN bridge batch." >&2
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
wandb_watch_log_freq="${WANDB_WATCH_LOG_FREQ:-100}"
run_prefix_base="${RUN_PREFIX_BASE:-h100bridge1}"
bundle_stage_dir="${BUNDLE_STAGE_DIR:-local-scratch/${run_prefix_base}_bundle}"
archive_output="${ARCHIVE_OUTPUT:-local-scratch/${run_prefix_base}_bundle.7z}"
command_log="${COMMAND_LOG:-local-scratch/${run_prefix_base}_commands.sh}"
torch_logs="${TORCH_LOGS:-}"
torch_trace="${TORCH_TRACE:-}"
build_hgdn_cuda="${BUILD_HGDN_CUDA:-1}"
run_hgdn_cuda_parity="${RUN_HGDN_CUDA_PARITY:-1}"
omp_num_threads="${OMP_NUM_THREADS:-1}"
mkl_num_threads="${MKL_NUM_THREADS:-1}"
openblas_num_threads="${OPENBLAS_NUM_THREADS:-1}"
numexpr_num_threads="${NUMEXPR_NUM_THREADS:-1}"

ngpu="${NGPU:-8}"
iterations="${ITERATIONS:-20000}"
train_batch_tokens="${TRAIN_BATCH_TOKENS:-2097152}"
train_seq_len="${TRAIN_SEQ_LEN:-2048}"
val_loss_every="${VAL_LOSS_EVERY:-1000}"
train_log_every="${TRAIN_LOG_EVERY:-200}"
val_batch_size="${VAL_BATCH_SIZE:-524288}"
max_wallclock_seconds="${MAX_WALLCLOCK_SECONDS:-600}"
compile="${COMPILE:-1}"
compile_strategy="${COMPILE_STRATEGY:-hybrid}"
depth_mlp_mult="${DEPTH_MLP_MULT:-4.0}"
attn_use_flash_attn3="${ATTN_USE_FLASH_ATTN3:-1}"
distributed_mode="${DISTRIBUTED_MODE:-parallel_muon}"
git_commit="$(git rev-parse HEAD)"
git_branch="$(git rev-parse --abbrev-ref HEAD)"
host_name="$(hostname)"
timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

hgdn_config="${HGDN_CONFIG:-configs/hgdn/retune_trim_layers_14.toml}"
hgdn_kernel_config="${HGDN_KERNEL_CONFIG:-configs/hgdn/winner_20260405_19_live14.toml}"
hgdn_run_id="${HGDN_RUN_ID:-${run_prefix_base}_exact_hybrid_r1_mlp3.25_seq${train_seq_len}}"
attn_run_id="${ATTN_RUN_ID:-${run_prefix_base}_exact_depth_mlp${depth_mlp_mult}_seq${train_seq_len}}"

hgdn_ensure_python_module "${python_bin}" py7zr py7zr

case "${wandb_mode}" in
online | offline) ;;
*)
    echo "Unsupported WANDB_MODE: ${wandb_mode} (expected online or offline)" >&2
    exit 1
    ;;
esac

load_config_env() {
    mapfile -t config_env < <(
        "${python_bin}" scripts/hgdn_helper_cli.py load-env --alias-aware --path "$@"
    )
}

print_plan() {
    echo
    echo ">>> H100 HGDN exact 8x matched-control bridge"
    echo "python_bin=${python_bin}"
    echo "use_wandb=${use_wandb}"
    echo "wandb_mode=${wandb_mode}"
    echo "wandb_project=${wandb_project}"
    echo "wandb_watch=${wandb_watch}"
    echo "wandb_watch_log_freq=${wandb_watch_log_freq}"
    echo "TORCH_LOGS=${torch_logs:-<unset>}"
    echo "TORCH_TRACE=${torch_trace:-<unset>}"
    echo "OMP_NUM_THREADS=${omp_num_threads}"
    echo "MKL_NUM_THREADS=${mkl_num_threads}"
    echo "OPENBLAS_NUM_THREADS=${openblas_num_threads}"
    echo "NUMEXPR_NUM_THREADS=${numexpr_num_threads}"
    echo "build_hgdn_cuda=${build_hgdn_cuda}"
    echo "run_hgdn_cuda_parity=${run_hgdn_cuda_parity}"
    echo "ngpu=${ngpu}"
    echo "iterations=${iterations}"
    echo "train_batch_tokens=${train_batch_tokens}"
    echo "train_seq_len=${train_seq_len}"
    echo "val_loss_every=${val_loss_every}"
    echo "train_log_every=${train_log_every}"
    echo "val_batch_size=${val_batch_size}"
    echo "compile=${compile}"
    echo "compile_strategy=${compile_strategy}"
    echo "attn_use_flash_attn3=${attn_use_flash_attn3}"
    echo "distributed_mode=${distributed_mode}"
    echo "max_wallclock_seconds=${max_wallclock_seconds}"
    echo "hgdn_config=${hgdn_config}"
    echo "hgdn_kernel_config=${hgdn_kernel_config}"
    echo "hgdn_run_id=${hgdn_run_id}"
    echo "attention_run_id=${attn_run_id}"
    echo "attention_mlp_mult=${depth_mlp_mult}"
    echo "archive_output=${archive_output}"
    echo "command_log=${command_log}"
    echo "bridge_goal=exact 8x go/no-go for HGDN finalist vs attention-only baseline under the same submission-style contract"
}

prepare_cuda() {
    mkdir -p "$(dirname "${command_log}")"
    {
        echo "#!/bin/bash"
        echo "set -euo pipefail"
    } >"${command_log}"

    if [[ "${build_hgdn_cuda}" == "1" ]]; then
        echo
        echo ">>> build HGDN CUDA extension"
        hgdn_append_plain_command \
            "${command_log}" \
            "${python_bin}" setup_hgdn_cuda.py build_ext --inplace
        "${python_bin}" setup_hgdn_cuda.py build_ext --inplace
    fi

    if [[ "${run_hgdn_cuda_parity}" == "1" ]]; then
        echo
        echo ">>> HGDN CUDA parity"
        hgdn_append_plain_command \
            "${command_log}" \
            "${python_bin}" scripts/hgdn_cuda_parity.py
        "${python_bin}" scripts/hgdn_cuda_parity.py
    fi
}

run_bridge() {
    local diagnostic_env=()
    if [[ -n "${torch_logs}" ]]; then
        diagnostic_env+=("TORCH_LOGS=${torch_logs}")
    fi
    if [[ -n "${torch_trace}" ]]; then
        diagnostic_env+=("TORCH_TRACE=${torch_trace}")
    fi

    load_config_env "${hgdn_kernel_config}" "${hgdn_config}"

    echo
    echo ">>> exact 8x bridge: HGDN finalist"
    hgdn_append_command \
        "${command_log}" \
        "OMP_NUM_THREADS=${omp_num_threads}" \
        "MKL_NUM_THREADS=${mkl_num_threads}" \
        "OPENBLAS_NUM_THREADS=${openblas_num_threads}" \
        "NUMEXPR_NUM_THREADS=${numexpr_num_threads}" \
        "NGPU=${ngpu}" \
        "USE_WANDB=${use_wandb}" \
        "WANDB_MODE=${wandb_mode}" \
        "WANDB_PROJECT=${wandb_project}" \
        "WANDB_WATCH=${wandb_watch}" \
        "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
        "${diagnostic_env[@]}" \
        "COMPILE=${compile}" \
        "COMPILE_STRATEGY=${compile_strategy}" \
        "ATTN_USE_FLASH_ATTN3=${attn_use_flash_attn3}" \
        "DISTRIBUTED_MODE=${distributed_mode}" \
        "RUN_ID=${hgdn_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "${config_env[@]}" \
        bash scripts/sweep.sh single-live14

    hgdn_run_sweep \
        "exact 8x bridge: HGDN finalist" \
        single-live14 \
        "OMP_NUM_THREADS=${omp_num_threads}" \
        "MKL_NUM_THREADS=${mkl_num_threads}" \
        "OPENBLAS_NUM_THREADS=${openblas_num_threads}" \
        "NUMEXPR_NUM_THREADS=${numexpr_num_threads}" \
        "NGPU=${ngpu}" \
        "USE_WANDB=${use_wandb}" \
        "WANDB_MODE=${wandb_mode}" \
        "WANDB_PROJECT=${wandb_project}" \
        "WANDB_WATCH=${wandb_watch}" \
        "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
        "${diagnostic_env[@]}" \
        "COMPILE=${compile}" \
        "COMPILE_STRATEGY=${compile_strategy}" \
        "ATTN_USE_FLASH_ATTN3=${attn_use_flash_attn3}" \
        "DISTRIBUTED_MODE=${distributed_mode}" \
        "RUN_ID=${hgdn_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "${config_env[@]}"

    echo
    echo ">>> exact 8x bridge: attention-only baseline"
    hgdn_append_command \
        "${command_log}" \
        "OMP_NUM_THREADS=${omp_num_threads}" \
        "MKL_NUM_THREADS=${mkl_num_threads}" \
        "OPENBLAS_NUM_THREADS=${openblas_num_threads}" \
        "NUMEXPR_NUM_THREADS=${numexpr_num_threads}" \
        "NGPU=${ngpu}" \
        "USE_WANDB=${use_wandb}" \
        "WANDB_MODE=${wandb_mode}" \
        "WANDB_PROJECT=${wandb_project}" \
        "WANDB_WATCH=${wandb_watch}" \
        "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
        "${diagnostic_env[@]}" \
        "COMPILE=${compile}" \
        "COMPILE_STRATEGY=${compile_strategy}" \
        "ATTN_USE_FLASH_ATTN3=${attn_use_flash_attn3}" \
        "DISTRIBUTED_MODE=${distributed_mode}" \
        "RUN_ID=${attn_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "MLP_MULT=${depth_mlp_mult}" \
        bash scripts/sweep.sh depth

    hgdn_run_sweep \
        "exact 8x bridge: attention-only baseline" \
        depth \
        "OMP_NUM_THREADS=${omp_num_threads}" \
        "MKL_NUM_THREADS=${mkl_num_threads}" \
        "OPENBLAS_NUM_THREADS=${openblas_num_threads}" \
        "NUMEXPR_NUM_THREADS=${numexpr_num_threads}" \
        "NGPU=${ngpu}" \
        "USE_WANDB=${use_wandb}" \
        "WANDB_MODE=${wandb_mode}" \
        "WANDB_PROJECT=${wandb_project}" \
        "WANDB_WATCH=${wandb_watch}" \
        "WANDB_WATCH_LOG_FREQ=${wandb_watch_log_freq}" \
        "${diagnostic_env[@]}" \
        "COMPILE=${compile}" \
        "COMPILE_STRATEGY=${compile_strategy}" \
        "ATTN_USE_FLASH_ATTN3=${attn_use_flash_attn3}" \
        "DISTRIBUTED_MODE=${distributed_mode}" \
        "RUN_ID=${attn_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "MLP_MULT=${depth_mlp_mult}"
}

build_bundle() {
    echo
    echo ">>> bundle outputs"

    rm -rf "${bundle_stage_dir}"
    mkdir -p "${bundle_stage_dir}/logs" "${bundle_stage_dir}/configs"

    cp "${hgdn_config}" "${bundle_stage_dir}/configs/"
    cp "${hgdn_kernel_config}" "${bundle_stage_dir}/configs/"
    cp "${command_log}" "${bundle_stage_dir}/commands.sh"

    local matched_logs=0
    local run_id
    for run_id in "${hgdn_run_id}" "${attn_run_id}"; do
        local log_path="logs/${run_id}.txt"
        if [[ -f "${log_path}" ]]; then
            cp "${log_path}" "${bundle_stage_dir}/logs/"
            matched_logs=1
        fi
    done

    "${python_bin}" scripts/hgdn_helper_cli.py write-h100-bridge-manifest \
        --output "${bundle_stage_dir}/bundle_manifest.json" \
        --run-prefix-base "${run_prefix_base}" \
        --wandb-project "${wandb_project}" \
        --wandb-mode "${wandb_mode}" \
        --archive-output "${archive_output}" \
        --matched-logs "${matched_logs}" \
        --command-log "${command_log}" \
        --torch-logs "${torch_logs}" \
        --torch-trace "${torch_trace}" \
        --omp-num-threads "${omp_num_threads}" \
        --mkl-num-threads "${mkl_num_threads}" \
        --openblas-num-threads "${openblas_num_threads}" \
        --numexpr-num-threads "${numexpr_num_threads}" \
        --attn-use-flash-attn3 "${attn_use_flash_attn3}" \
        --distributed-mode "${distributed_mode}" \
        --ngpu "${ngpu}" \
        --iterations "${iterations}" \
        --train-batch-tokens "${train_batch_tokens}" \
        --train-seq-len "${train_seq_len}" \
        --val-loss-every "${val_loss_every}" \
        --train-log-every "${train_log_every}" \
        --val-batch-size "${val_batch_size}" \
        --max-wallclock-seconds "${max_wallclock_seconds}" \
        --compile-enabled "${compile}" \
        --compile-strategy "${compile_strategy}" \
        --depth-mlp-mult "${depth_mlp_mult}" \
        --hgdn-config "${hgdn_config}" \
        --hgdn-kernel-config "${hgdn_kernel_config}" \
        --hgdn-run-id "${hgdn_run_id}" \
        --attn-run-id "${attn_run_id}" \
        --git-commit "${git_commit}" \
        --git-branch "${git_branch}" \
        --host-name "${host_name}" \
        --timestamp-utc "${timestamp_utc}"

    hgdn_create_7z_archive "${python_bin}" "${archive_output}" "${bundle_stage_dir}"
    echo "bundle_archive=${archive_output}"
}

print_plan
prepare_cuda
run_bridge
build_bundle
