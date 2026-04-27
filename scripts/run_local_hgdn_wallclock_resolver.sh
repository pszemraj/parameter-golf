#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It executes the local true-wallclock HGDN resolver batch." >&2
    exit 1
fi

hgdn_require_cmd bash
hgdn_require_cmd torchrun
hgdn_require_cmd python

python_bin="${PYTHON_BIN:-python}"
use_wandb="${USE_WANDB:-0}"
wandb_mode="${WANDB_MODE:-offline}"
wandb_project="${WANDB_PROJECT:-pg-hgdn-ablations}"
wandb_watch="${WANDB_WATCH:-none}"
wandb_watch_log_freq="${WANDB_WATCH_LOG_FREQ:-25}"
run_prefix_base="${RUN_PREFIX_BASE:-localhgdn_wallclock1}"
bundle_stage_dir="${BUNDLE_STAGE_DIR:-local-scratch/${run_prefix_base}_bundle}"
archive_output="${ARCHIVE_OUTPUT:-local-scratch/${run_prefix_base}_bundle.7z}"
command_log="${COMMAND_LOG:-local-scratch/${run_prefix_base}_commands.sh}"
size_screen_output="${SIZE_SCREEN_OUTPUT:-local-scratch/${run_prefix_base}_size_screen}"
size_screen_config="${SIZE_SCREEN_CONFIG:-configs/hgdn/naive_contract_search.toml}"
torchinductor_max_autotune="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
torchinductor_max_autotune_gemm="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-0}"
torch_logs="${TORCH_LOGS:-}"
torch_trace="${TORCH_TRACE:-}"
allow_existing_logs="${ALLOW_EXISTING_LOGS:-0}"
check_cuda_idle="${CHECK_CUDA_IDLE:-1}"
allow_active_cuda_jobs="${ALLOW_ACTIVE_CUDA_JOBS:-0}"
artifact_limit_bytes="${ARTIFACT_LIMIT_BYTES:-16000000}"

ngpu="${NGPU:-1}"
iterations="${ITERATIONS:-20000}"
train_batch_tokens="${TRAIN_BATCH_TOKENS:-65536}"
train_seq_len="${TRAIN_SEQ_LEN:-1024}"
val_loss_every="${VAL_LOSS_EVERY:-100}"
train_log_every="${TRAIN_LOG_EVERY:-25}"
min_val_seqs="${MIN_VAL_SEQS:-512}"
val_max_seqs="${VAL_MAX_SEQS:-512}"
max_wallclock_seconds="${MAX_WALLCLOCK_SECONDS:-600}"
compile="${COMPILE:-1}"
compile_strategy="${COMPILE_STRATEGY:-hybrid}"
distributed_mode="${DISTRIBUTED_MODE:-parallel_muon}"
gdn_fla_recurrence_mode="${GDN_FLA_RECURRENCE_MODE:-}"
weight_decay="${WEIGHT_DECAY:-0}"
grad_accum_steps_override="${GRAD_ACCUM_STEPS:-}"
data_path="${DATA_PATH:-$HGDN_REPO_ROOT/data/datasets/fineweb10B_sp1024}"
tokenizer_path="${TOKENIZER_PATH:-$HGDN_REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
vocab_size="${VOCAB_SIZE:-1024}"

primary_hgdn_config="${PRIMARY_HGDN_CONFIG:-configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml}"
primary_control_config="${PRIMARY_CONTROL_CONFIG:-configs/hgdn/naive_contract_l8_d512_r0_m2.toml}"
secondary_hgdn_config="${SECONDARY_HGDN_CONFIG:-configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml}"
secondary_control_config="${SECONDARY_CONTROL_CONFIG:-configs/hgdn/naive_contract_l8_d512_r0_m1p25.toml}"
primary_hgdn_recurrence_mode="${PRIMARY_GDN_FLA_RECURRENCE_MODE:-${gdn_fla_recurrence_mode:-direct}}"
primary_control_recurrence_mode="${PRIMARY_CONTROL_GDN_FLA_RECURRENCE_MODE:-${primary_hgdn_recurrence_mode}}"
secondary_hgdn_recurrence_mode="${SECONDARY_GDN_FLA_RECURRENCE_MODE:-${gdn_fla_recurrence_mode:-direct_fused}}"
secondary_control_recurrence_mode="${SECONDARY_CONTROL_GDN_FLA_RECURRENCE_MODE:-${secondary_hgdn_recurrence_mode}}"
primary_control_margin="${PRIMARY_CONTROL_MARGIN:-0.003}"
secondary_primary_margin="${SECONDARY_PRIMARY_MARGIN:-0.005}"

omp_num_threads="${OMP_NUM_THREADS:-1}"
mkl_num_threads="${MKL_NUM_THREADS:-1}"
openblas_num_threads="${OPENBLAS_NUM_THREADS:-1}"
numexpr_num_threads="${NUMEXPR_NUM_THREADS:-1}"
nccl_ib_disable="${NCCL_IB_DISABLE:-1}"

bundle_written=0
analysis_written=0
git_commit="$(git rev-parse HEAD)"
git_branch="$(git rev-parse --abbrev-ref HEAD)"
host_name="$(hostname)"
timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

hgdn_validate_fla_recurrence_mode "${primary_hgdn_recurrence_mode}"
hgdn_validate_fla_recurrence_mode "${primary_control_recurrence_mode}"
hgdn_validate_fla_recurrence_mode "${secondary_hgdn_recurrence_mode}"
hgdn_validate_fla_recurrence_mode "${secondary_control_recurrence_mode}"
hgdn_ensure_python_module "${python_bin}" py7zr py7zr

if ! awk "BEGIN { exit !(${max_wallclock_seconds} > 0) }"; then
    echo "MAX_WALLCLOCK_SECONDS must be > 0 for the true-wallclock resolver." >&2
    exit 1
fi

case "${wandb_mode}" in
online | offline) ;;
*)
    echo "Unsupported WANDB_MODE: ${wandb_mode} (expected online or offline)" >&2
    exit 1
    ;;
esac

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
val_batch_size="$(hgdn_resolve_val_batch_size "${ngpu}" "${grad_accum_steps}" "${train_seq_len}")"

wallclock_tag="${max_wallclock_seconds//./p}"
gpt_naive_run_id="${GPT_NAIVE_RUN_ID:-${run_prefix_base}_gpt_naive_baseline_seq${train_seq_len}_wall${wallclock_tag}}"
primary_hgdn_run_id="${PRIMARY_HGDN_RUN_ID:-${run_prefix_base}_primary_hgdn_$(basename "${primary_hgdn_config}" .toml)_seq${train_seq_len}_wall${wallclock_tag}}"
primary_control_run_id="${PRIMARY_CONTROL_RUN_ID:-${run_prefix_base}_primary_control_$(basename "${primary_control_config}" .toml)_seq${train_seq_len}_wall${wallclock_tag}}"
secondary_hgdn_run_id="${SECONDARY_HGDN_RUN_ID:-${run_prefix_base}_secondary_hgdn_$(basename "${secondary_hgdn_config}" .toml)_seq${train_seq_len}_wall${wallclock_tag}}"
secondary_control_run_id="${SECONDARY_CONTROL_RUN_ID:-${run_prefix_base}_secondary_control_$(basename "${secondary_control_config}" .toml)_seq${train_seq_len}_wall${wallclock_tag}}"
expected_run_ids=(
    "${gpt_naive_run_id}"
    "${primary_hgdn_run_id}"
    "${primary_control_run_id}"
    "${secondary_hgdn_run_id}"
    "${secondary_control_run_id}"
)

diagnostic_env=()
if [[ -n "${torch_logs}" ]]; then
    diagnostic_env+=("TORCH_LOGS=${torch_logs}")
fi
if [[ -n "${torch_trace}" ]]; then
    diagnostic_env+=("TORCH_TRACE=${torch_trace}")
fi

common_thread_env=(
    "OMP_NUM_THREADS=${omp_num_threads}"
    "MKL_NUM_THREADS=${mkl_num_threads}"
    "OPENBLAS_NUM_THREADS=${openblas_num_threads}"
    "NUMEXPR_NUM_THREADS=${numexpr_num_threads}"
    "NCCL_IB_DISABLE=${nccl_ib_disable}"
)

check_cuda_jobs() {
    if [[ "${check_cuda_idle}" != "1" || "${allow_active_cuda_jobs}" == "1" ]]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi
    local active_jobs
    active_jobs="$(
        nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
    )"
    if [[ -n "${active_jobs}" ]]; then
        echo "Refusing to start wallclock resolver while CUDA compute jobs are active:" >&2
        echo "${active_jobs}" >&2
        echo "Set ALLOW_ACTIVE_CUDA_JOBS=1 only if this overlap is intentional." >&2
        exit 1
    fi
}

load_config_env() {
    local config_path="$1"
    local recurrence_mode="$2"
    local raw_config_env=()
    mapfile -t raw_config_env < <(
        "${python_bin}" scripts/hgdn_helper_cli.py load-env --alias-aware --path "${config_path}"
    )
    mapfile -t config_env < <(
        hgdn_filter_recurrence_env "${raw_config_env[@]}"
    )
    config_env+=("GDN_USE_DIRECT_FLA_LAYER_SEMANTICS=0")
    config_env+=("GDN_FLA_RECURRENCE_MODE=${recurrence_mode}")
}

prepare_command_log() {
    mkdir -p "$(dirname "${command_log}")"
    {
        echo "#!/bin/bash"
        echo "set -euo pipefail"
    } >"${command_log}"
}

append_size_screen_command() {
    hgdn_append_plain_command \
        "${command_log}" \
        "${python_bin}" scripts/screen_hgdn_arch_sizes.py \
        --config "${size_screen_config}" \
        --gdn-fla-recurrence-mode "${primary_hgdn_recurrence_mode}" \
        --output-dir "${size_screen_output}"
}

append_exact_baseline_command() {
    hgdn_append_command \
        "${command_log}" \
        "${common_thread_env[@]}" \
        "${diagnostic_env[@]}" \
        "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}" \
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}" \
        "RUN_ID=${gpt_naive_run_id}" \
        "DATA_PATH=${data_path}" \
        "TOKENIZER_PATH=${tokenizer_path}" \
        "VOCAB_SIZE=${vocab_size}" \
        "GRAD_ACCUM_STEPS=${grad_accum_steps}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "MIN_VAL_SEQS=${min_val_seqs}" \
        "VAL_MAX_SEQS=${val_max_seqs}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "ARTIFACT_LIMIT_BYTES=${artifact_limit_bytes}" \
        "NUM_LAYERS=9" \
        "MODEL_DIM=512" \
        "NUM_HEADS=8" \
        "NUM_KV_HEADS=4" \
        "MLP_MULT=2" \
        torchrun --standalone --nproc_per_node="${ngpu}" train_gpt.py
}

append_hybrid_command() {
    local config_path="$1"
    local run_id="$2"
    local recurrence_mode="$3"
    load_config_env "${config_path}" "${recurrence_mode}"
    hgdn_append_command \
        "${command_log}" \
        "${common_thread_env[@]}" \
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
        "DISTRIBUTED_MODE=${distributed_mode}" \
        "RUN_ID=${run_id}" \
        "DATA_PATH=${data_path}" \
        "TOKENIZER_PATH=${tokenizer_path}" \
        "VOCAB_SIZE=${vocab_size}" \
        "GRAD_ACCUM_STEPS=${grad_accum_steps}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "MIN_VAL_SEQS=${min_val_seqs}" \
        "VAL_MAX_SEQS=${val_max_seqs}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "WEIGHT_DECAY=${weight_decay}" \
        "PERF_SKIP_FINAL_EVAL=0" \
        "ARTIFACT_LIMIT_BYTES=${artifact_limit_bytes}" \
        "${config_env[@]}" \
        torchrun --standalone --nproc_per_node="${ngpu}" train_gpt_hybrid.py
}

write_command_log() {
    prepare_command_log
    append_size_screen_command
    append_exact_baseline_command
    append_hybrid_command "${primary_hgdn_config}" "${primary_hgdn_run_id}" "${primary_hgdn_recurrence_mode}"
    append_hybrid_command "${primary_control_config}" "${primary_control_run_id}" "${primary_control_recurrence_mode}"
    append_hybrid_command "${secondary_hgdn_config}" "${secondary_hgdn_run_id}" "${secondary_hgdn_recurrence_mode}"
    append_hybrid_command "${secondary_control_config}" "${secondary_control_run_id}" "${secondary_control_recurrence_mode}"
}

check_planned_logs_are_fresh() {
    if [[ "${allow_existing_logs}" == "1" ]]; then
        return
    fi
    local conflict=0
    local run_id
    for run_id in "${expected_run_ids[@]}"; do
        local log_path="logs/${run_id}.txt"
        if [[ -e "${log_path}" ]]; then
            echo "Refusing to append to existing run log: ${log_path}" >&2
            conflict=1
        fi
    done
    if (( conflict )); then
        echo "Use a fresh RUN_PREFIX_BASE, or set ALLOW_EXISTING_LOGS=1 only for an intentional append." >&2
        exit 1
    fi
}

print_plan() {
    echo
    echo ">>> Local HGDN true-wallclock resolver"
    echo "run_prefix_base=${run_prefix_base}"
    echo "bundle_stage_dir=${bundle_stage_dir}"
    echo "archive_output=${archive_output}"
    echo "ngpu=${ngpu}"
    echo "grad_accum_steps=${grad_accum_steps}"
    echo "iterations=${iterations}"
    echo "max_wallclock_seconds=${max_wallclock_seconds}"
    echo "train_batch_tokens=${train_batch_tokens}"
    echo "train_seq_len=${train_seq_len}"
    echo "val_loss_every=${val_loss_every}"
    echo "train_log_every=${train_log_every}"
    echo "min_val_seqs=${min_val_seqs}"
    echo "val_max_seqs=${val_max_seqs}"
    echo "val_batch_size=${val_batch_size}"
    echo "compile=${compile}"
    echo "compile_strategy=${compile_strategy}"
    echo "primary_gdn_fla_recurrence_mode=${primary_hgdn_recurrence_mode}"
    echo "primary_control_gdn_fla_recurrence_mode=${primary_control_recurrence_mode}"
    echo "secondary_gdn_fla_recurrence_mode=${secondary_hgdn_recurrence_mode}"
    echo "secondary_control_gdn_fla_recurrence_mode=${secondary_control_recurrence_mode}"
    echo "data_path=${data_path}"
    echo "tokenizer_path=${tokenizer_path}"
    echo "vocab_size=${vocab_size}"
    echo "primary_hgdn_config=${primary_hgdn_config}"
    echo "primary_control_config=${primary_control_config}"
    echo "secondary_hgdn_config=${secondary_hgdn_config}"
    echo "secondary_control_config=${secondary_control_config}"
    echo "decision_margins: primary_control=${primary_control_margin} secondary_primary=${secondary_primary_margin}"
}

run_size_screen() {
    echo
    echo ">>> artifact-size screen"
    "${python_bin}" scripts/screen_hgdn_arch_sizes.py \
        --config "${size_screen_config}" \
        --gdn-fla-recurrence-mode "${primary_hgdn_recurrence_mode}" \
        --output-dir "${size_screen_output}"
}

run_exact_baseline() {
    echo
    echo ">>> exact repo naive baseline local wallclock resolver"
    hgdn_run_with_env \
        "${common_thread_env[@]}" \
        "${diagnostic_env[@]}" \
        "TORCHINDUCTOR_MAX_AUTOTUNE=${torchinductor_max_autotune}" \
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=${torchinductor_max_autotune_gemm}" \
        "RUN_ID=${gpt_naive_run_id}" \
        "DATA_PATH=${data_path}" \
        "TOKENIZER_PATH=${tokenizer_path}" \
        "VOCAB_SIZE=${vocab_size}" \
        "GRAD_ACCUM_STEPS=${grad_accum_steps}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "MIN_VAL_SEQS=${min_val_seqs}" \
        "VAL_MAX_SEQS=${val_max_seqs}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "ARTIFACT_LIMIT_BYTES=${artifact_limit_bytes}" \
        "NUM_LAYERS=9" \
        "MODEL_DIM=512" \
        "NUM_HEADS=8" \
        "NUM_KV_HEADS=4" \
        "MLP_MULT=2" \
        torchrun --standalone --nproc_per_node="${ngpu}" train_gpt.py
}

run_hybrid_config() {
    local label="$1"
    local config_path="$2"
    local run_id="$3"
    local recurrence_mode="$4"
    load_config_env "${config_path}" "${recurrence_mode}"
    echo
    echo ">>> ${label}"
    echo "run_id=${run_id}"
    echo "gdn_fla_recurrence_mode=${recurrence_mode}"
    hgdn_run_hybrid_train \
        "${label}" \
        "${common_thread_env[@]}" \
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
        "DISTRIBUTED_MODE=${distributed_mode}" \
        "RUN_ID=${run_id}" \
        "DATA_PATH=${data_path}" \
        "TOKENIZER_PATH=${tokenizer_path}" \
        "VOCAB_SIZE=${vocab_size}" \
        "GRAD_ACCUM_STEPS=${grad_accum_steps}" \
        "ITERATIONS=${iterations}" \
        "MAX_WALLCLOCK_SECONDS=${max_wallclock_seconds}" \
        "TRAIN_BATCH_TOKENS=${train_batch_tokens}" \
        "TRAIN_SEQ_LEN=${train_seq_len}" \
        "VAL_LOSS_EVERY=${val_loss_every}" \
        "TRAIN_LOG_EVERY=${train_log_every}" \
        "MIN_VAL_SEQS=${min_val_seqs}" \
        "VAL_MAX_SEQS=${val_max_seqs}" \
        "VAL_BATCH_SIZE=${val_batch_size}" \
        "WEIGHT_DECAY=${weight_decay}" \
        "PERF_SKIP_FINAL_EVAL=0" \
        "ARTIFACT_LIMIT_BYTES=${artifact_limit_bytes}" \
        "${config_env[@]}"
}

run_batch() {
    run_exact_baseline
    run_hybrid_config "primary HGDN local wallclock resolver" "${primary_hgdn_config}" "${primary_hgdn_run_id}" "${primary_hgdn_recurrence_mode}"
    run_hybrid_config "primary attention-only baseline diagnostic control local wallclock resolver" "${primary_control_config}" "${primary_control_run_id}" "${primary_control_recurrence_mode}"
    run_hybrid_config "secondary HGDN local wallclock resolver" "${secondary_hgdn_config}" "${secondary_hgdn_run_id}" "${secondary_hgdn_recurrence_mode}"
    run_hybrid_config "secondary attention-only baseline diagnostic control local wallclock resolver" "${secondary_control_config}" "${secondary_control_run_id}" "${secondary_control_recurrence_mode}"
}

build_bundle() {
    local exit_status="${1:-0}"
    if [[ "${bundle_written}" == "1" ]]; then
        return
    fi
    bundle_written=1

    echo
    echo ">>> stage bundle outputs"

    rm -rf "${bundle_stage_dir}"
    mkdir -p \
        "${bundle_stage_dir}/logs" \
        "${bundle_stage_dir}/configs" \
        "${bundle_stage_dir}/size_screen"

    local config_path
    for config_path in \
        "${primary_hgdn_config}" \
        "${primary_control_config}" \
        "${secondary_hgdn_config}" \
        "${secondary_control_config}" \
        "${size_screen_config}"; do
        if [[ -f "${config_path}" ]]; then
            cp "${config_path}" "${bundle_stage_dir}/configs/"
        fi
    done

    if [[ -f "${command_log}" ]]; then
        cp "${command_log}" "${bundle_stage_dir}/commands.sh"
    fi
    if [[ -d "${size_screen_output}" ]]; then
        cp -R "${size_screen_output}/." "${bundle_stage_dir}/size_screen/"
    fi

    local matched_logs=1
    local completed_log_count=0
    local missing_run_ids=()
    local run_id
    for run_id in "${expected_run_ids[@]}"; do
        local log_path="logs/${run_id}.txt"
        if [[ -f "${log_path}" ]]; then
            cp "${log_path}" "${bundle_stage_dir}/logs/"
            completed_log_count=$((completed_log_count + 1))
        else
            matched_logs=0
            missing_run_ids+=("${run_id}")
        fi
    done

    local -a manifest_cmd
    local manifest_recurrence_mode="mixed"
    if [[ \
        "${primary_hgdn_recurrence_mode}" == "${primary_control_recurrence_mode}" && \
        "${primary_hgdn_recurrence_mode}" == "${secondary_hgdn_recurrence_mode}" && \
        "${primary_hgdn_recurrence_mode}" == "${secondary_control_recurrence_mode}" \
    ]]; then
        manifest_recurrence_mode="${primary_hgdn_recurrence_mode}"
    fi
    manifest_cmd=(
        "${python_bin}" scripts/hgdn_helper_cli.py write-local-wallclock-resolver-manifest
        --output "${bundle_stage_dir}/bundle_manifest.json"
        --run-prefix-base "${run_prefix_base}"
        --wandb-project "${wandb_project}"
        --wandb-mode "${wandb_mode}"
        --archive-output "${archive_output}"
        --exit-status "${exit_status}"
        --matched-logs "${matched_logs}"
        --completed-log-count "${completed_log_count}"
        --command-log "${command_log}"
        --size-screen-config "${size_screen_config}"
        --size-screen-output "${size_screen_output}"
        --torch-logs "${torch_logs}"
        --torch-trace "${torch_trace}"
        --torchinductor-max-autotune "${torchinductor_max_autotune}"
        --torchinductor-max-autotune-gemm "${torchinductor_max_autotune_gemm}"
        --ngpu "${ngpu}"
        --grad-accum-steps "${grad_accum_steps}"
        --iterations "${iterations}"
        --train-batch-tokens "${train_batch_tokens}"
        --train-seq-len "${train_seq_len}"
        --val-loss-every "${val_loss_every}"
        --train-log-every "${train_log_every}"
        --min-val-seqs "${min_val_seqs}"
        --val-max-seqs "${val_max_seqs}"
        --val-batch-size "${val_batch_size}"
        --max-wallclock-seconds "${max_wallclock_seconds}"
        --compile-enabled "${compile}"
        --compile-strategy "${compile_strategy}"
        --distributed-mode "${distributed_mode}"
        --gdn-fla-recurrence-mode "${manifest_recurrence_mode}"
        --primary-hgdn-recurrence-mode "${primary_hgdn_recurrence_mode}"
        --primary-control-recurrence-mode "${primary_control_recurrence_mode}"
        --secondary-hgdn-recurrence-mode "${secondary_hgdn_recurrence_mode}"
        --secondary-control-recurrence-mode "${secondary_control_recurrence_mode}"
        --weight-decay "${weight_decay}"
        --data-path "${data_path}"
        --tokenizer-path "${tokenizer_path}"
        --vocab-size "${vocab_size}"
        --gpt-naive-run-id "${gpt_naive_run_id}"
        --primary-hgdn-config "${primary_hgdn_config}"
        --primary-control-config "${primary_control_config}"
        --secondary-hgdn-config "${secondary_hgdn_config}"
        --secondary-control-config "${secondary_control_config}"
        --primary-hgdn-run-id "${primary_hgdn_run_id}"
        --primary-control-run-id "${primary_control_run_id}"
        --secondary-hgdn-run-id "${secondary_hgdn_run_id}"
        --secondary-control-run-id "${secondary_control_run_id}"
        --git-commit "${git_commit}"
        --git-branch "${git_branch}"
        --host-name "${host_name}"
        --timestamp-utc "${timestamp_utc}"
    )
    for run_id in "${missing_run_ids[@]}"; do
        manifest_cmd+=(--missing-run-id "${run_id}")
    done
    "${manifest_cmd[@]}"
    echo "bundle_dir=${bundle_stage_dir}"
}

analyze_bundle() {
    if [[ "${analysis_written}" == "1" ]]; then
        return
    fi
    analysis_written=1
    echo
    echo ">>> analyze local wallclock resolver"
    "${python_bin}" scripts/analyze_hgdn_experiment_bundle.py \
        --bundle-dir "${bundle_stage_dir}" \
        --output-dir "${bundle_stage_dir}/analysis" \
        --decision-json "${bundle_stage_dir}/selected_hgdn.json" \
        --select config \
        --metric final_roundtrip_bpb \
        --confirm-top-n 2 \
        --top 20
    "${python_bin}" scripts/resolve_hgdn_wallclock_decision.py \
        --rows-json "${bundle_stage_dir}/analysis/rows.json" \
        --output-json "${bundle_stage_dir}/wallclock_decision.json" \
        --primary-config "${primary_hgdn_config}" \
        --primary-control-config "${primary_control_config}" \
        --secondary-config "${secondary_hgdn_config}" \
        --secondary-control-config "${secondary_control_config}" \
        --primary-control-margin "${primary_control_margin}" \
        --secondary-primary-margin "${secondary_primary_margin}"
}

archive_bundle() {
    if [[ -d "${bundle_stage_dir}" ]]; then
        hgdn_create_7z_archive "${python_bin}" "${archive_output}" "${bundle_stage_dir}"
        echo "bundle_archive=${archive_output}"
    fi
}

on_exit() {
    local status="$1"
    trap - EXIT
    if [[ -f "${command_log}" || -d "${size_screen_output}" ]]; then
        build_bundle "${status}" || status=$?
        if [[ "${status}" == "0" ]]; then
            analyze_bundle || status=$?
        fi
        archive_bundle || status=$?
    fi
    exit "${status}"
}

main() {
    print_plan
    check_cuda_jobs
    write_command_log
    check_planned_logs_are_fresh
    trap 'on_exit $?' EXIT
    run_size_screen
    run_batch
}

main
