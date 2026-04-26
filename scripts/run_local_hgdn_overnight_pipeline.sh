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
grad_accum_steps_override="${GRAD_ACCUM_STEPS:-}"
val_loss_every="${VAL_LOSS_EVERY:-100}"
train_log_every="${TRAIN_LOG_EVERY:-25}"
min_val_seqs="${MIN_VAL_SEQS:-512}"
val_max_seqs="${VAL_MAX_SEQS:-512}"
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
run_stage3="${RUN_STAGE3:-1}"
recurrence_iterations="${RECURRENCE_ITERATIONS:-500}"
screen_iterations="${SCREEN_ITERATIONS:-300}"
confirm_iterations="${CONFIRM_ITERATIONS:-500}"
secondary_iterations="${SECONDARY_ITERATIONS:-${confirm_iterations}}"
screen_perf_skip_final_eval="${SCREEN_PERF_SKIP_FINAL_EVAL:-1}"
confirm_perf_skip_final_eval="${CONFIRM_PERF_SKIP_FINAL_EVAL:-0}"
secondary_perf_skip_final_eval="${SECONDARY_PERF_SKIP_FINAL_EVAL:-${confirm_perf_skip_final_eval}}"
confirm_top_hgdn="${CONFIRM_TOP_HGDN:-2}"
recurrence_selection_metric="${RECURRENCE_SELECTION_METRIC:-equal_wallclock_bpb}"
screen_selection_metric="${SCREEN_SELECTION_METRIC:-${SEARCH_SELECTION_METRIC:-auto}}"
confirm_selection_metric="${CONFIRM_SELECTION_METRIC:-final_roundtrip_bpb}"
secondary_selection_metric="${SECONDARY_SELECTION_METRIC:-${confirm_selection_metric}}"
secondary_force="${SECONDARY_FORCE:-0}"

default_screen_configs=(
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_m2.toml"
    "configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml"
    "configs/hgdn/naive_contract_l8_d512_r0_m1p5.toml"
    "configs/hgdn/naive_contract_l8_d512_r0_m2.toml"
)

join_by_comma() {
    local IFS=,
    echo "$*"
}

screen_candidate_configs="${SCREEN_CANDIDATE_CONFIGS:-$(join_by_comma "${default_screen_configs[@]}")}"
secondary_candidate_configs="${SECONDARY_CANDIDATE_CONFIGS:-configs/hgdn/naive_contract_l8_d512_olmoish_6g2a_v2_m1p25.toml,configs/hgdn/naive_contract_l8_d512_r0_m1p25.toml}"

hgdn_ensure_python_module "${python_bin}" py7zr py7zr

resolve_grad_accum_steps() {
    if [[ -n "${grad_accum_steps_override}" ]]; then
        if ((grad_accum_steps_override < 1)); then
            echo "GRAD_ACCUM_STEPS must be >= 1, got ${grad_accum_steps_override}" >&2
            exit 1
        fi
        echo "${grad_accum_steps_override}"
        return
    fi
    if ((8 % ngpu != 0)); then
        echo "NGPU must evenly divide 8 when GRAD_ACCUM_STEPS is not set: NGPU=${ngpu}" >&2
        exit 1
    fi
    echo $((8 / ngpu))
}

grad_accum_steps="$(resolve_grad_accum_steps)"
val_batch_size="$(hgdn_resolve_val_batch_size "${ngpu}" "${grad_accum_steps}" "${train_seq_len}")"
val_batch_seqs=$((val_batch_size / train_seq_len))

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
check_bool_flag RUN_STAGE3 "${run_stage3}"
check_bool_flag SCREEN_PERF_SKIP_FINAL_EVAL "${screen_perf_skip_final_eval}"
check_bool_flag CONFIRM_PERF_SKIP_FINAL_EVAL "${confirm_perf_skip_final_eval}"
check_bool_flag SECONDARY_PERF_SKIP_FINAL_EVAL "${secondary_perf_skip_final_eval}"
check_bool_flag SECONDARY_FORCE "${secondary_force}"

if (( screen_perf_skip_final_eval == 1 && screen_iterations % val_loss_every != 0 )); then
    echo "SCREEN_ITERATIONS must be divisible by VAL_LOSS_EVERY when SCREEN_PERF_SKIP_FINAL_EVAL=1." >&2
    exit 1
fi

if (( recurrence_iterations % val_loss_every != 0 )); then
    echo "RECURRENCE_ITERATIONS should be divisible by VAL_LOSS_EVERY for clean promotion." >&2
    exit 1
fi

if (( secondary_perf_skip_final_eval == 1 && secondary_iterations % val_loss_every != 0 )); then
    echo "SECONDARY_ITERATIONS must be divisible by VAL_LOSS_EVERY when SECONDARY_PERF_SKIP_FINAL_EVAL=1." >&2
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

json_get_scalar() {
    local json_path="$1"
    local key="$2"
    "${python_bin}" - "${json_path}" "${key}" <<'PY'
import json
import sys

path, key = sys.argv[1:3]
value = json.loads(open(path, encoding="utf-8").read()).get(key)
if isinstance(value, bool):
    print("1" if value else "0")
elif value is None:
    print("")
elif isinstance(value, list):
    print(",".join(str(item) for item in value))
else:
    print(value)
PY
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
    env \
        RUN_PREFIX_BASE_JSON="${run_prefix_base}" \
        USE_WANDB_JSON="${use_wandb}" \
        WANDB_MODE_JSON="${wandb_mode}" \
        WANDB_PROJECT_JSON="${wandb_project}" \
        NGPU_JSON="${ngpu}" \
        TRAIN_BATCH_TOKENS_JSON="${train_batch_tokens}" \
        TRAIN_SEQ_LEN_JSON="${train_seq_len}" \
        GRAD_ACCUM_STEPS_JSON="${grad_accum_steps}" \
        VAL_LOSS_EVERY_JSON="${val_loss_every}" \
        TRAIN_LOG_EVERY_JSON="${train_log_every}" \
        MIN_VAL_SEQS_JSON="${min_val_seqs}" \
        VAL_MAX_SEQS_JSON="${val_max_seqs}" \
        VAL_BATCH_SIZE_JSON="${val_batch_size}" \
        VAL_BATCH_SEQS_JSON="${val_batch_seqs}" \
        COMPILE_JSON="${compile}" \
        COMPILE_STRATEGY_JSON="${compile_strategy}" \
        DISTRIBUTED_MODE_JSON="${distributed_mode}" \
        DATA_PATH_JSON="${data_path}" \
        TOKENIZER_PATH_JSON="${tokenizer_path}" \
        VOCAB_SIZE_JSON="${vocab_size}" \
        RECURRENCE_ITERATIONS_JSON="${recurrence_iterations}" \
        SCREEN_ITERATIONS_JSON="${screen_iterations}" \
        CONFIRM_ITERATIONS_JSON="${confirm_iterations}" \
        SECONDARY_ITERATIONS_JSON="${secondary_iterations}" \
        CONFIRM_TOP_HGDN_JSON="${confirm_top_hgdn}" \
        RECURRENCE_SELECTION_METRIC_JSON="${recurrence_selection_metric}" \
        SCREEN_SELECTION_METRIC_JSON="${screen_selection_metric}" \
        CONFIRM_SELECTION_METRIC_JSON="${confirm_selection_metric}" \
        SECONDARY_SELECTION_METRIC_JSON="${secondary_selection_metric}" \
        SCREEN_CANDIDATE_CONFIGS_JSON="${screen_candidate_configs}" \
        SECONDARY_CANDIDATE_CONFIGS_JSON="${secondary_candidate_configs}" \
        SECONDARY_FORCE_JSON="${secondary_force}" \
        "${python_bin}" - "${pipeline_dir}/pipeline_plan.json" <<'PY'
import json
import os
import sys


def split_csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


plan = {
    "run_prefix_base": os.environ["RUN_PREFIX_BASE_JSON"],
    "use_wandb": os.environ["USE_WANDB_JSON"] == "1",
    "wandb_mode": os.environ["WANDB_MODE_JSON"],
    "wandb_project": os.environ["WANDB_PROJECT_JSON"],
    "ngpu": int(os.environ["NGPU_JSON"]),
    "train_batch_tokens": int(os.environ["TRAIN_BATCH_TOKENS_JSON"]),
    "train_seq_len": int(os.environ["TRAIN_SEQ_LEN_JSON"]),
    "grad_accum_steps": int(os.environ["GRAD_ACCUM_STEPS_JSON"]),
    "val_loss_every": int(os.environ["VAL_LOSS_EVERY_JSON"]),
    "train_log_every": int(os.environ["TRAIN_LOG_EVERY_JSON"]),
    "min_val_seqs": int(os.environ["MIN_VAL_SEQS_JSON"]),
    "val_max_seqs": int(os.environ["VAL_MAX_SEQS_JSON"]),
    "val_batch_size": int(os.environ["VAL_BATCH_SIZE_JSON"]),
    "val_batch_seqs": int(os.environ["VAL_BATCH_SEQS_JSON"]),
    "compile": os.environ["COMPILE_JSON"] == "1",
    "compile_strategy": os.environ["COMPILE_STRATEGY_JSON"],
    "distributed_mode": os.environ["DISTRIBUTED_MODE_JSON"],
    "data_path": os.environ["DATA_PATH_JSON"],
    "tokenizer_path": os.environ["TOKENIZER_PATH_JSON"],
    "vocab_size": int(os.environ["VOCAB_SIZE_JSON"]),
    "recurrence_iterations": int(os.environ["RECURRENCE_ITERATIONS_JSON"]),
    "screen_iterations": int(os.environ["SCREEN_ITERATIONS_JSON"]),
    "confirm_iterations": int(os.environ["CONFIRM_ITERATIONS_JSON"]),
    "secondary_iterations": int(os.environ["SECONDARY_ITERATIONS_JSON"]),
    "confirm_top_hgdn": int(os.environ["CONFIRM_TOP_HGDN_JSON"]),
    "recurrence_selection_metric": os.environ["RECURRENCE_SELECTION_METRIC_JSON"],
    "screen_selection_metric": os.environ["SCREEN_SELECTION_METRIC_JSON"],
    "confirm_selection_metric": os.environ["CONFIRM_SELECTION_METRIC_JSON"],
    "secondary_selection_metric": os.environ["SECONDARY_SELECTION_METRIC_JSON"],
    "screen_candidate_configs": split_csv(os.environ["SCREEN_CANDIDATE_CONFIGS_JSON"]),
    "secondary_candidate_configs": split_csv(os.environ["SECONDARY_CANDIDATE_CONFIGS_JSON"]),
    "secondary_force": os.environ["SECONDARY_FORCE_JSON"] == "1",
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(plan, fh, indent=2)
    fh.write("\n")
PY
}

print_plan() {
    echo
    echo ">>> Local HGDN overnight hierarchy"
    echo "pipeline_dir=${pipeline_dir}"
    echo "archive_output=${archive_output}"
    echo "run_stage0=${run_stage0} recurrence_iterations=${recurrence_iterations}"
    echo "run_stage1=${run_stage1} screen_iterations=${screen_iterations}"
    echo "run_stage2=${run_stage2} confirm_iterations=${confirm_iterations}"
    echo "run_stage3=${run_stage3} secondary_iterations=${secondary_iterations}"
    echo "screen_candidate_configs=${screen_candidate_configs}"
    echo "secondary_candidate_configs=${secondary_candidate_configs}"
    echo "confirm_top_hgdn=${confirm_top_hgdn}"
    echo "recurrence_selection_metric=${recurrence_selection_metric}"
    echo "screen_selection_metric=${screen_selection_metric}"
    echo "confirm_selection_metric=${confirm_selection_metric}"
    echo "secondary_selection_metric=${secondary_selection_metric}"
    echo "secondary_force=${secondary_force}"
    echo "data_path=${data_path}"
    echo "tokenizer_path=${tokenizer_path}"
    echo "vocab_size=${vocab_size}"
    echo "grad_accum_steps=${grad_accum_steps}"
    echo "min_val_seqs=${min_val_seqs}"
    echo "val_max_seqs=${val_max_seqs}"
    echo "val_batch_size=${val_batch_size} tokens (${val_batch_seqs} sequences)"
    echo "gates:"
    echo "  stage0: recurrence mode matrix on v2_m1p5"
    echo "  stage1: bounded candidate/control screen using the selected recurrence mode"
    echo "  stage2: longer confirmation for top HGDN configs plus matched controls"
    echo "  stage3: conditional OLMo-ish 6G/2A sanity check when the primary beats control"
    echo "No paid-H100 job or paid-H100 handoff is launched or generated."
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
        GRAD_ACCUM_STEPS="${grad_accum_steps}" \
        ITERATIONS="${recurrence_iterations}" \
        TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
        TRAIN_SEQ_LEN="${train_seq_len}" \
        VAL_LOSS_EVERY="${val_loss_every}" \
        TRAIN_LOG_EVERY="${train_log_every}" \
        MIN_VAL_SEQS="${min_val_seqs}" \
        VAL_MAX_SEQS="${val_max_seqs}" \
        VAL_BATCH_SIZE="${val_batch_size}" \
        VAL_BATCH_SEQS= \
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
        "${pipeline_dir}/stage0_decision.json"
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
    local decision_json="$9"
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
        GRAD_ACCUM_STEPS="${grad_accum_steps}" \
        ITERATIONS="${iterations}" \
        TRAIN_BATCH_TOKENS="${train_batch_tokens}" \
        TRAIN_SEQ_LEN="${train_seq_len}" \
        VAL_LOSS_EVERY="${val_loss_every}" \
        TRAIN_LOG_EVERY="${train_log_every}" \
        MIN_VAL_SEQS="${min_val_seqs}" \
        VAL_MAX_SEQS="${val_max_seqs}" \
        VAL_BATCH_SIZE="${val_batch_size}" \
        VAL_BATCH_SEQS= \
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
        "${decision_json}"
}

analyze_stage() {
    local stage_name="$1"
    local bundle_dir="$2"
    local select_kind="$3"
    local metric="$4"
    local decision_json="$5"
    local output_dir="${pipeline_dir}/${stage_name}_analysis"

    echo
    echo ">>> analyze ${stage_name}"
    "${python_bin}" scripts/analyze_hgdn_experiment_bundle.py \
        --bundle-dir "${bundle_dir}" \
        --output-dir "${output_dir}" \
        --decision-json "${decision_json}" \
        --select "${select_kind}" \
        --metric "${metric}" \
        --confirm-top-n "${confirm_top_hgdn}" \
        --top 20
}

write_stage3_skip() {
    local reason="$1"
    local skip_json="${pipeline_dir}/stage3_skipped.json"
    env \
        STAGE3_SKIP_REASON_JSON="${reason}" \
        SECONDARY_CANDIDATE_CONFIGS_JSON="${secondary_candidate_configs}" \
        "${python_bin}" - "${skip_json}" <<'PY'
import json
import os
import sys

payload = {
    "stage3_skipped": True,
    "reason": os.environ["STAGE3_SKIP_REASON_JSON"],
    "secondary_candidate_configs": [
        item
        for item in os.environ["SECONDARY_CANDIDATE_CONFIGS_JSON"].split(",")
        if item
    ],
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
    fh.write("\n")
PY
    echo "stage3_skipped=${reason}"
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
        selected_mode="$(
            json_get_scalar \
                "${pipeline_dir}/stage0_decision.json" \
                selected_gdn_fla_recurrence_mode
        )"
    fi

    stage1_decision="${pipeline_dir}/stage1_decision.json"
    if [[ "${run_stage1}" == "1" ]]; then
        run_search_stage \
            "stage1_screen" \
            "${run_prefix_base}_s1_screen" \
            "${screen_iterations}" \
            "${screen_perf_skip_final_eval}" \
            "${selected_mode}" \
            "${screen_candidate_configs}" \
            "config" \
            "${screen_selection_metric}" \
            "${stage1_decision}"
    fi

    stage2_configs="${CONFIRM_CANDIDATE_CONFIGS:-}"
    if [[ -z "${stage2_configs}" && -f "${stage1_decision}" ]]; then
        stage2_configs="$(json_get_scalar "${stage1_decision}" selected_confirm_configs)"
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
            "${confirm_selection_metric}" \
            "${pipeline_dir}/stage2_decision.json"
    fi

    if [[ "${run_stage3}" == "1" ]]; then
        local stage2_decision="${pipeline_dir}/stage2_decision.json"
        if [[ ! -f "${stage2_decision}" ]]; then
            write_stage3_skip "missing_stage2_decision"
        else
            selected_beats_control="$(
                json_get_scalar "${stage2_decision}" selected_beats_control
            )"
            if [[ "${secondary_force}" != "1" && "${selected_beats_control:-0}" != "1" ]]; then
                write_stage3_skip "primary_did_not_beat_matched_control"
            else
                run_search_stage \
                    "stage3_secondary_sanity" \
                    "${run_prefix_base}_s3_secondary" \
                    "${secondary_iterations}" \
                    "${secondary_perf_skip_final_eval}" \
                    "${selected_mode}" \
                    "${secondary_candidate_configs}" \
                    "config" \
                    "${secondary_selection_metric}" \
                    "${pipeline_dir}/stage3_decision.json"
            fi
        fi
    fi
}

main
