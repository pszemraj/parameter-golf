#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It always executes the bounded naive-baseline-contract three-way sanity batch." >&2
    exit 1
fi

hgdn_require_cmd bash
hgdn_require_cmd torchrun
hgdn_require_cmd python

python_bin="${PYTHON_BIN:-python}"
use_wandb="${USE_WANDB:-1}"
wandb_mode="${WANDB_MODE:-online}"
wandb_project="${WANDB_PROJECT:-pg-hgdn-ablations}"
wandb_watch="${WANDB_WATCH:-gradients}"
wandb_watch_log_freq="${WANDB_WATCH_LOG_FREQ:-100}"
run_prefix_base="${RUN_PREFIX_BASE:-h100naive1}"
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
train_batch_tokens="${TRAIN_BATCH_TOKENS:-524288}"
train_seq_len="${TRAIN_SEQ_LEN:-1024}"
val_loss_every="${VAL_LOSS_EVERY:-200}"
train_log_every="${TRAIN_LOG_EVERY:-50}"
val_batch_size="${VAL_BATCH_SIZE:-524288}"
max_wallclock_seconds="${MAX_WALLCLOCK_SECONDS:-600}"
compile="${COMPILE:-1}"
compile_strategy="${COMPILE_STRATEGY:-hybrid}"
weight_decay="${WEIGHT_DECAY:-0}"
data_path="${DATA_PATH:-$HGDN_REPO_ROOT/data/datasets/fineweb10B_sp1024}"
tokenizer_path="${TOKENIZER_PATH:-$HGDN_REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
vocab_size="${VOCAB_SIZE:-1024}"

hgdn_config="${HGDN_CONFIG:-configs/hgdn/retune_trim_layers_14.toml}"
hgdn_kernel_config="${HGDN_KERNEL_CONFIG:-configs/hgdn/winner_20260405_19_live14.toml}"
gpt_naive_run_id="${GPT_NAIVE_RUN_ID:-${run_prefix_base}_gpt_naive_baseline_seq${train_seq_len}}"
hgdn_run_id="${HGDN_RUN_ID:-${run_prefix_base}_hybrid_naive_contract_seq${train_seq_len}}"
attn_run_id="${ATTN_RUN_ID:-${run_prefix_base}_depth_naive_contract_seq${train_seq_len}}"

naive_reference_name="${NAIVE_REFERENCE_NAME:-2026-03-17_NaiveBaseline}"
naive_reference_roundtrip_bpb="${NAIVE_REFERENCE_ROUNDTRIP_BPB:-1.22436570}"
naive_reference_stop_bpb="${NAIVE_REFERENCE_STOP_BPB:-1.2172}"

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
    echo ">>> H100 HGDN naive-baseline-contract sanity round"
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
    echo "weight_decay=${weight_decay}"
    echo "max_wallclock_seconds=${max_wallclock_seconds}"
    echo "data_path=${data_path}"
    echo "tokenizer_path=${tokenizer_path}"
    echo "vocab_size=${vocab_size}"
    echo "hgdn_config=${hgdn_config}"
    echo "hgdn_kernel_config=${hgdn_kernel_config}"
    echo "gpt_naive_run_id=${gpt_naive_run_id}"
    echo "hgdn_run_id=${hgdn_run_id}"
    echo "attention_run_id=${attn_run_id}"
    echo "naive_reference_name=${naive_reference_name}"
    echo "naive_reference_roundtrip_bpb=${naive_reference_roundtrip_bpb}"
    echo "naive_reference_stop_bpb=${naive_reference_stop_bpb}"
    echo "goal=measure whether the exact repo naive baseline, the hybrid-trainer attention-only control, and the live HGDN finalist can all be compared on the official naive-baseline contract"
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

run_round() {
    local diagnostic_env=()
    if [[ -n "${torch_logs}" ]]; then
        diagnostic_env+=("TORCH_LOGS=${torch_logs}")
    fi
    if [[ -n "${torch_trace}" ]]; then
        diagnostic_env+=("TORCH_TRACE=${torch_trace}")
    fi

    load_config_env "${hgdn_kernel_config}" "${hgdn_config}"

    echo
    echo ">>> exact repo naive baseline"
    hgdn_append_command \
        "${command_log}" \
        "OMP_NUM_THREADS=${omp_num_threads}" \
        "MKL_NUM_THREADS=${mkl_num_threads}" \
        "OPENBLAS_NUM_THREADS=${openblas_num_threads}" \
        "NUMEXPR_NUM_THREADS=${numexpr_num_threads}" \
        "${diagnostic_env[@]}" \
        "RUN_ID=${gpt_naive_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "DATA_PATH=${data_path}" \
        "TOKENIZER_PATH=${tokenizer_path}" \
        "VOCAB_SIZE=${vocab_size}" \
        "NUM_LAYERS=9" \
        "MODEL_DIM=512" \
        "NUM_HEADS=8" \
        "NUM_KV_HEADS=4" \
        "MLP_MULT=2" \
        torchrun --standalone --nproc_per_node="${ngpu}" train_gpt.py

    hgdn_run_with_env \
        "OMP_NUM_THREADS=${omp_num_threads}" \
        "MKL_NUM_THREADS=${mkl_num_threads}" \
        "OPENBLAS_NUM_THREADS=${openblas_num_threads}" \
        "NUMEXPR_NUM_THREADS=${numexpr_num_threads}" \
        "${diagnostic_env[@]}" \
        "RUN_ID=${gpt_naive_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "DATA_PATH=${data_path}" \
        "TOKENIZER_PATH=${tokenizer_path}" \
        "VOCAB_SIZE=${vocab_size}" \
        "NUM_LAYERS=9" \
        "MODEL_DIM=512" \
        "NUM_HEADS=8" \
        "NUM_KV_HEADS=4" \
        "MLP_MULT=2" \
        torchrun --standalone --nproc_per_node="${ngpu}" train_gpt.py

    echo
    echo ">>> naive-contract HGDN finalist"
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
        "WEIGHT_DECAY=${weight_decay}" \
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
        "naive-contract HGDN finalist" \
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
        "WEIGHT_DECAY=${weight_decay}" \
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
    echo ">>> naive-contract attention-only control"
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
        "WEIGHT_DECAY=${weight_decay}" \
        "RUN_ID=${attn_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "NUM_LAYERS=9" \
        "MODEL_DIM=512" \
        "NUM_HEADS=8" \
        "NUM_KV_HEADS=4" \
        "GDN_RATIO=0" \
        "MLP_MULT=2" \
        bash scripts/sweep.sh depth

    hgdn_run_sweep \
        "naive-contract attention-only control" \
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
        "WEIGHT_DECAY=${weight_decay}" \
        "RUN_ID=${attn_run_id}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "NUM_LAYERS=9" \
        "MODEL_DIM=512" \
        "NUM_HEADS=8" \
        "NUM_KV_HEADS=4" \
        "GDN_RATIO=0" \
        "MLP_MULT=2"
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
    for run_id in "${gpt_naive_run_id}" "${hgdn_run_id}" "${attn_run_id}"; do
        local log_path="logs/${run_id}.txt"
        if [[ -f "${log_path}" ]]; then
            cp "${log_path}" "${bundle_stage_dir}/logs/"
            matched_logs=1
        fi
    done

    "${python_bin}" scripts/hgdn_helper_cli.py write-h100-naive-contract-manifest \
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
        --weight-decay "${weight_decay}" \
        --baseline-data-path "${data_path}" \
        --baseline-tokenizer-path "${tokenizer_path}" \
        --baseline-vocab-size "${vocab_size}" \
        --gpt-naive-run-id "${gpt_naive_run_id}" \
        --hgdn-config "${hgdn_config}" \
        --hgdn-kernel-config "${hgdn_kernel_config}" \
        --hgdn-run-id "${hgdn_run_id}" \
        --attn-run-id "${attn_run_id}" \
        --naive-reference-name "${naive_reference_name}" \
        --naive-reference-roundtrip-bpb "${naive_reference_roundtrip_bpb}" \
        --naive-reference-stop-bpb "${naive_reference_stop_bpb}"

    hgdn_create_7z_archive "${python_bin}" "${archive_output}" "${bundle_stage_dir}"
    echo "bundle_archive=${archive_output}"
}

print_plan
prepare_cuda
run_round
build_bundle
