#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

if [[ "$#" -ne 0 ]]; then
    echo "Run this script with no arguments." >&2
    echo "It executes the staged local HGDN overnight pipeline." >&2
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
run_prefix_base="${RUN_PREFIX_BASE:-localhgdnpipeline1}"
pipeline_dir="${PIPELINE_DIR:-local-scratch/${run_prefix_base}_pipeline}"
archive_output="${ARCHIVE_OUTPUT:-local-scratch/${run_prefix_base}_pipeline.7z}"
torch_logs="${TORCH_LOGS:-}"
torch_trace="${TORCH_TRACE:-}"
allow_existing_logs="${ALLOW_EXISTING_LOGS:-0}"
check_cuda_idle="${CHECK_CUDA_IDLE:-1}"
allow_active_cuda_jobs="${ALLOW_ACTIVE_CUDA_JOBS:-0}"

ngpu="${NGPU:-1}"
train_batch_tokens="${TRAIN_BATCH_TOKENS:-65536}"
train_seq_len="${TRAIN_SEQ_LEN:-1024}"
val_loss_every="${VAL_LOSS_EVERY:-100}"
train_log_every="${TRAIN_LOG_EVERY:-25}"
max_wallclock_seconds="${MAX_WALLCLOCK_SECONDS:-0}"
compile="${COMPILE:-1}"
compile_strategy="${COMPILE_STRATEGY:-hybrid}"
distributed_mode="${DISTRIBUTED_MODE:-parallel_muon}"
weight_decay="${WEIGHT_DECAY:-0}"
torchinductor_max_autotune="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
torchinductor_max_autotune_gemm="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-0}"
data_path="${DATA_PATH:-$HGDN_REPO_ROOT/data/datasets/fineweb10B_sp1024}"
tokenizer_path="${TOKENIZER_PATH:-$HGDN_REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
vocab_size="${VOCAB_SIZE:-1024}"

run_stage0="${RUN_STAGE0:-1}"
run_stage1="${RUN_STAGE1:-1}"
run_stage2="${RUN_STAGE2:-1}"
recurrence_iterations="${RECURRENCE_ITERATIONS:-500}"
screen_iterations="${SCREEN_ITERATIONS:-300}"
confirm_iterations="${CONFIRM_ITERATIONS:-500}"
screen_perf_skip_final_eval="${SCREEN_PERF_SKIP_FINAL_EVAL:-1}"
confirm_perf_skip_final_eval="${CONFIRM_PERF_SKIP_FINAL_EVAL:-0}"
confirm_top_hgdn="${CONFIRM_TOP_HGDN:-2}"
recurrence_selection_metric="${RECURRENCE_SELECTION_METRIC:-final_step_ms}"
search_selection_metric="${SEARCH_SELECTION_METRIC:-auto}"

default_screen_configs=(
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml"
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml"
    "configs/hgdn/naive_contract_l8_d512_mid3_dk48_m1p75.toml"
    "configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml"
    "configs/hgdn/naive_contract_l9_d512_mid3_dk48_v1p5_m1p75.toml"
    "configs/hgdn/naive_contract_l8_d512_r0_m1p25.toml"
    "configs/hgdn/naive_contract_l8_d512_r0_m1p5.toml"
    "configs/hgdn/naive_contract_l8_d512_r0_m1p75.toml"
    "configs/hgdn/naive_contract_l8_d512_r0_m2.toml"
    "configs/hgdn/naive_contract_l9_d512_r0_m1p75.toml"
)

join_by_comma() {
    local IFS=,
    echo "$*"
}

screen_candidate_configs="${SCREEN_CANDIDATE_CONFIGS:-$(join_by_comma "${default_screen_configs[@]}")}"

hgdn_ensure_python_module "${python_bin}" py7zr py7zr

case "${wandb_mode}" in
online | offline) ;;
*)
    echo "Unsupported WANDB_MODE: ${wandb_mode} (expected online or offline)" >&2
    exit 1
    ;;
esac

check_bool_flag() {
    local name="$1"
    local value="$2"
    case "${value}" in
    0 | 1) ;;
    *)
        echo "${name} must be 0 or 1, got ${value}" >&2
        exit 1
        ;;
    esac
}

check_bool_flag RUN_STAGE0 "${run_stage0}"
check_bool_flag RUN_STAGE1 "${run_stage1}"
check_bool_flag RUN_STAGE2 "${run_stage2}"
check_bool_flag SCREEN_PERF_SKIP_FINAL_EVAL "${screen_perf_skip_final_eval}"
check_bool_flag CONFIRM_PERF_SKIP_FINAL_EVAL "${confirm_perf_skip_final_eval}"

if (( screen_perf_skip_final_eval == 1 && screen_iterations % val_loss_every != 0 )); then
    echo "SCREEN_ITERATIONS must be divisible by VAL_LOSS_EVERY when SCREEN_PERF_SKIP_FINAL_EVAL=1." >&2
    exit 1
fi

if (( recurrence_iterations % val_loss_every != 0 )); then
    echo "RECURRENCE_ITERATIONS should be divisible by VAL_LOSS_EVERY for clean promotion." >&2
    exit 1
fi

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
        echo "Refusing to start overnight pipeline stage while CUDA compute jobs are active:" >&2
        echo "${active_jobs}" >&2
        echo "Set ALLOW_ACTIVE_CUDA_JOBS=1 only if this overlap is intentional." >&2
        exit 1
    fi
}

stage_bundle_dir() {
    local stage_prefix="$1"
    echo "local-scratch/${stage_prefix}_bundle"
}

stage_archive() {
    local stage_prefix="$1"
    echo "local-scratch/${stage_prefix}_bundle.7z"
}

stage_command_log() {
    local stage_prefix="$1"
    echo "local-scratch/${stage_prefix}_commands.sh"
}

write_plan() {
    mkdir -p "${pipeline_dir}"
    {
        echo "run_prefix_base=${run_prefix_base}"
        echo "use_wandb=${use_wandb}"
        echo "wandb_mode=${wandb_mode}"
        echo "wandb_project=${wandb_project}"
        echo "ngpu=${ngpu}"
        echo "train_batch_tokens=${train_batch_tokens}"
        echo "train_seq_len=${train_seq_len}"
        echo "val_loss_every=${val_loss_every}"
        echo "train_log_every=${train_log_every}"
        echo "compile=${compile}"
        echo "compile_strategy=${compile_strategy}"
        echo "distributed_mode=${distributed_mode}"
        echo "data_path=${data_path}"
        echo "tokenizer_path=${tokenizer_path}"
        echo "vocab_size=${vocab_size}"
        echo "recurrence_iterations=${recurrence_iterations}"
        echo "screen_iterations=${screen_iterations}"
        echo "confirm_iterations=${confirm_iterations}"
        echo "confirm_top_hgdn=${confirm_top_hgdn}"
        echo "recurrence_selection_metric=${recurrence_selection_metric}"
        echo "search_selection_metric=${search_selection_metric}"
        echo "screen_candidate_configs=${screen_candidate_configs}"
    } >"${pipeline_dir}/pipeline_plan.env"
}

print_plan() {
    echo
    echo ">>> Local HGDN overnight hierarchy"
    echo "pipeline_dir=${pipeline_dir}"
    echo "archive_output=${archive_output}"
    echo "run_stage0=${run_stage0} recurrence_iterations=${recurrence_iterations}"
    echo "run_stage1=${run_stage1} screen_iterations=${screen_iterations}"
    echo "run_stage2=${run_stage2} confirm_iterations=${confirm_iterations}"
    echo "screen_candidate_configs=${screen_candidate_configs}"
    echo "confirm_top_hgdn=${confirm_top_hgdn}"
    echo "recurrence_selection_metric=${recurrence_selection_metric}"
    echo "search_selection_metric=${search_selection_metric}"
    echo "data_path=${data_path}"
    echo "tokenizer_path=${tokenizer_path}"
    echo "vocab_size=${vocab_size}"
    echo "gates:"
    echo "  stage0: recurrence mode matrix on v2_m1p5"
    echo "  stage1: bounded candidate/control screen using the selected recurrence mode"
    echo "  stage2: longer confirmation for top HGDN configs plus matched controls"
    echo "No H100 job is launched; the final H100 command is written for review."
}

run_recurrence_stage() {
    local stage_prefix="${run_prefix_base}_s0_recur"
    local diagnostic_env=()
    if [[ -n "${torch_logs}" ]]; then
        diagnostic_env+=("TORCH_LOGS=${torch_logs}")
    fi
    if [[ -n "${torch_trace}" ]]; then
        diagnostic_env+=("TORCH_TRACE=${torch_trace}")
    fi
    echo
    echo ">>> stage0 recurrence implementation matrix"
    check_cuda_jobs
    env \
        PYTHON_BIN="${python_bin}" \
        USE_WANDB="${use_wandb}" \
        WANDB_MODE="${wandb_mode}" \
        WANDB_PROJECT="${wandb_project}" \
        WANDB_WATCH="${wandb_watch}" \
        WANDB_WATCH_LOG_FREQ="${wandb_watch_log_freq}" \
        RUN_PREFIX_BASE="${stage_prefix}" \
        BUNDLE_STAGE_DIR="$(stage_bundle_dir "${stage_prefix}")" \
        ARCHIVE_OUTPUT="$(stage_archive "${stage_prefix}")" \
        COMMAND_LOG="$(stage_command_log "${stage_prefix}")" \
        "${diagnostic_env[@]}" \
        TORCHINDUCTOR_MAX_AUTOTUNE="${torchinductor_max_autotune}" \
        TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${torchinductor_max_autotune_gemm}" \
        DATA_PATH="${data_path}" \
        TOKENIZER_PATH="${tokenizer_path}" \
        VOCAB_SIZE="${vocab_size}" \
        ALLOW_EXISTING_LOGS="${allow_existing_logs}" \
        CHECK_CUDA_IDLE=0 \
        NGPU="${ngpu}" \
        ITERATIONS="${recurrence_iterations}" \
        TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
        TRAIN_SEQ_LEN="${train_seq_len}" \
        VAL_LOSS_EVERY="${val_loss_every}" \
        TRAIN_LOG_EVERY="${train_log_every}" \
        MAX_WALLCLOCK_SECONDS="${max_wallclock_seconds}" \
        COMPILE="${compile}" \
        COMPILE_STRATEGY="${compile_strategy}" \
        DISTRIBUTED_MODE="${distributed_mode}" \
        WEIGHT_DECAY="${weight_decay}" \
        PERF_SKIP_FINAL_EVAL=0 \
        bash scripts/run_local_hgdn_recurrence_matrix.sh

    analyze_stage \
        "stage0_recurrence" \
        "$(stage_bundle_dir "${stage_prefix}")" \
        "mode" \
        "${recurrence_selection_metric}" \
        "${pipeline_dir}/stage0_decision.env"
}

run_search_stage() {
    local stage_name="$1"
    local stage_prefix="$2"
    local iterations="$3"
    local perf_skip_final_eval="$4"
    local selected_mode="$5"
    local candidate_configs="$6"
    local decision_kind="$7"
    local metric="$8"
    local decision_env="$9"
    local diagnostic_env=()
    if [[ -n "${torch_logs}" ]]; then
        diagnostic_env+=("TORCH_LOGS=${torch_logs}")
    fi
    if [[ -n "${torch_trace}" ]]; then
        diagnostic_env+=("TORCH_TRACE=${torch_trace}")
    fi

    echo
    echo ">>> ${stage_name}"
    echo "selected_mode=${selected_mode}"
    echo "candidate_configs=${candidate_configs}"
    hgdn_validate_fla_recurrence_mode "${selected_mode}"
    check_cuda_jobs
    env \
        PYTHON_BIN="${python_bin}" \
        USE_WANDB="${use_wandb}" \
        WANDB_MODE="${wandb_mode}" \
        WANDB_PROJECT="${wandb_project}" \
        WANDB_WATCH="${wandb_watch}" \
        WANDB_WATCH_LOG_FREQ="${wandb_watch_log_freq}" \
        RUN_PREFIX_BASE="${stage_prefix}" \
        BUNDLE_STAGE_DIR="$(stage_bundle_dir "${stage_prefix}")" \
        ARCHIVE_OUTPUT="$(stage_archive "${stage_prefix}")" \
        COMMAND_LOG="$(stage_command_log "${stage_prefix}")" \
        SIZE_SCREEN_OUTPUT="local-scratch/${stage_prefix}_size_screen" \
        "${diagnostic_env[@]}" \
        TORCHINDUCTOR_MAX_AUTOTUNE="${torchinductor_max_autotune}" \
        TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${torchinductor_max_autotune_gemm}" \
        DATA_PATH="${data_path}" \
        TOKENIZER_PATH="${tokenizer_path}" \
        VOCAB_SIZE="${vocab_size}" \
        ALLOW_EXISTING_LOGS="${allow_existing_logs}" \
        CANDIDATE_CONFIGS="${candidate_configs}" \
        GDN_FLA_RECURRENCE_MODE="${selected_mode}" \
        NGPU="${ngpu}" \
        ITERATIONS="${iterations}" \
        TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
        TRAIN_SEQ_LEN="${train_seq_len}" \
        VAL_LOSS_EVERY="${val_loss_every}" \
        TRAIN_LOG_EVERY="${train_log_every}" \
        MAX_WALLCLOCK_SECONDS="${max_wallclock_seconds}" \
        COMPILE="${compile}" \
        COMPILE_STRATEGY="${compile_strategy}" \
        DISTRIBUTED_MODE="${distributed_mode}" \
        WEIGHT_DECAY="${weight_decay}" \
        PERF_SKIP_FINAL_EVAL="${perf_skip_final_eval}" \
        bash scripts/run_local_hgdn_naive_contract_search.sh

    analyze_stage \
        "${stage_name}" \
        "$(stage_bundle_dir "${stage_prefix}")" \
        "${decision_kind}" \
        "${metric}" \
        "${decision_env}"
}

analyze_stage() {
    local stage_name="$1"
    local bundle_dir="$2"
    local select_kind="$3"
    local metric="$4"
    local decision_env="$5"
    local output_dir="${pipeline_dir}/${stage_name}_analysis"

    echo
    echo ">>> analyze ${stage_name}"
    "${python_bin}" scripts/analyze_hgdn_experiment_bundle.py \
        --bundle-dir "${bundle_dir}" \
        --output-dir "${output_dir}" \
        --decision-env "${decision_env}" \
        --select "${select_kind}" \
        --metric "${metric}" \
        --confirm-top-n "${confirm_top_hgdn}" \
        --top 20
}

write_h100_next_command() {
    local decision_env="$1"
    # shellcheck disable=SC1090
    source "${decision_env}"
    local h100_command="${pipeline_dir}/next_h100_command.sh"
    {
        echo "#!/bin/bash"
        echo "set -euo pipefail"
        printf '%q ' \
            "USE_WANDB=0" \
            "WANDB_MODE=offline" \
            "ATTN_USE_FLASH_ATTN3=1" \
            "DISTRIBUTED_MODE=parallel_muon" \
            "MUON_DISTRIBUTED_MODE=packed_allreduce" \
            "GDN_W_G_OPTIMIZER=matrix" \
            "GDN_FLA_RECURRENCE_MODE=${SELECTED_GDN_FLA_RECURRENCE_MODE}" \
            "HGDN_CONFIG=${SELECTED_CONFIG}" \
            "ATTN_CONFIG=${SELECTED_CONTROL_CONFIG:-}" \
            "WANDB_WATCH=none" \
            "RUN_PREFIX_BASE=h100naive_${run_prefix_base}_finalist" \
            bash scripts/run_h100_hgdn_naive_contract_round.sh
        printf '\n'
    } >"${h100_command}"
    chmod +x "${h100_command}"
    echo "wrote ${h100_command}"
}

build_pipeline_bundle() {
    if [[ ! -d "${pipeline_dir}" ]]; then
        return
    fi
    echo
    echo ">>> pipeline bundle"
    hgdn_create_7z_archive "${python_bin}" "${archive_output}" "${pipeline_dir}"
    echo "pipeline_archive=${archive_output}"
}

on_exit() {
    local status="$1"
    trap - EXIT
    build_pipeline_bundle || status=$?
    exit "${status}"
}

main() {
    print_plan
    write_plan
    trap 'on_exit $?' EXIT

    selected_mode="${GDN_FLA_RECURRENCE_MODE:-direct}"
    if [[ "${run_stage0}" == "1" ]]; then
        run_recurrence_stage
        # shellcheck disable=SC1090
        source "${pipeline_dir}/stage0_decision.env"
        selected_mode="${SELECTED_GDN_FLA_RECURRENCE_MODE}"
    fi

    stage1_decision="${pipeline_dir}/stage1_decision.env"
    if [[ "${run_stage1}" == "1" ]]; then
        run_search_stage \
            "stage1_screen" \
            "${run_prefix_base}_s1_screen" \
            "${screen_iterations}" \
            "${screen_perf_skip_final_eval}" \
            "${selected_mode}" \
            "${screen_candidate_configs}" \
            "config" \
            "${search_selection_metric}" \
            "${stage1_decision}"
    fi

    stage2_configs="${CONFIRM_CANDIDATE_CONFIGS:-}"
    if [[ -z "${stage2_configs}" && -f "${stage1_decision}" ]]; then
        # shellcheck disable=SC1090
        source "${stage1_decision}"
        stage2_configs="${SELECTED_CONFIRM_CONFIGS_CSV}"
    fi
    if [[ "${run_stage2}" == "1" ]]; then
        if [[ -z "${stage2_configs}" ]]; then
            echo "No confirmation configs available. Run stage1 or set CONFIRM_CANDIDATE_CONFIGS." >&2
            exit 1
        fi
        run_search_stage \
            "stage2_confirm" \
            "${run_prefix_base}_s2_confirm" \
            "${confirm_iterations}" \
            "${confirm_perf_skip_final_eval}" \
            "${selected_mode}" \
            "${stage2_configs}" \
            "config" \
            "${search_selection_metric}" \
            "${pipeline_dir}/stage2_decision.env"
        write_h100_next_command "${pipeline_dir}/stage2_decision.env"
    fi
}

main
