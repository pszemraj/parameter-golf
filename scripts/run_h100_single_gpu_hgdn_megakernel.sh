#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

mode="${1:-all}"

usage() {
    cat <<'EOF'
Usage: scripts/run_h100_single_gpu_hgdn_megakernel.sh {parity|trainer-smoke|all|matrix|compare100|help}

Purpose:
  Bounded 1xH100 validation helper for the repo-backed HGDN megakernel path.
  This helper is intentionally narrow:
  - build the extension for the requested CUDA arch list
  - run the static contract audit
  - run the isolated parity/launch-count/timing harness
  - optionally run one short trainer smoke in megakernel mode

  It is not a sweep and it is not a final H100 timing harness.

Modes:
  parity
    Build the extension, run the static audit, then run the isolated
    megakernel parity/launch-count/timing harness with:
    - B=1,T=8/32/128/512
    - optional B=2,T=512
    - optional B=1,T=2048
    - timing repeats defaulting to 3

  trainer-smoke
    Run one bounded trainer smoke in megakernel mode with the current
    winner-like hybrid contract:
    - NUM_LAYERS=14
    - MODEL_DIM=384
    - MLP_MULT=3.25
    - GDN_RATIO=1
    - TRAIN_SEQ_LEN=2048
    - TRAIN_BATCH_TOKENS=524288
    - ITERATIONS=5
    - COMPILE=1
    - COMPILE_STRATEGY=hybrid
    - COMPILE_WARMUP_STEPS=2
    - VAL_LOSS_EVERY=0
    - PERF_SKIP_FINAL_EVAL=1
    - GDN_USE_CUDA_MEGAKERNEL=1
    - GDN_CONTROL_PROJ_FP32=0
    - GDN_USE_PACKED_QKV_CONV=1
    - GDN_USE_PACKED_QKV_PROJ=1
    - GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=0
    - GDN_CONV_OUTPUT_CONTIGUOUS=1

  all
    Run parity first, then the bounded trainer smoke.

  matrix
    Run a small candidate matrix sequentially from one entrypoint. Each
    candidate reuses the same helper contract and gets its own bundle under a
    shared matrix output directory, plus a top-level matrix summary/archive.
    The default matrix now covers the only runtime cadence points still worth
    spending H100 time on after the first broad pass:
    - base_rc8_v8:  GDN_MEGAKERNEL_REC_CHUNK_T=8
    - rc6_v8:       GDN_MEGAKERNEL_REC_CHUNK_T=6
    - rc4_v8:       GDN_MEGAKERNEL_REC_CHUNK_T=4

  compare100
    Run a narrowed candidate matrix under a more reliable 100-step trainer
    contract instead of a 5-step smoke. By default it revalidates parity and
    then runs the trainer for 100 steps on:
    - base_rc8_v8:  GDN_MEGAKERNEL_REC_CHUNK_T=8
    - rc6_v8:       GDN_MEGAKERNEL_REC_CHUNK_T=6
    - rc4_v8:       GDN_MEGAKERNEL_REC_CHUNK_T=4
    Default compare100 overrides:
    - MK_MATRIX_CHILD_MODE=all
    - ITERATIONS=100
    - COMPILE_WARMUP_STEPS=20
    - TRAIN_LOG_EVERY=25
    - MAX_WALLCLOCK_SECONDS=0

Important notes:
  - Default target arch is H100: TORCH_CUDA_ARCH_LIST=9.0
  - For local command-path validation only, override TORCH_CUDA_ARCH_LIST=8.9
    or 12.0 and do not interpret the result as H100 timing evidence
  - Trainer smoke assumes challenge data/tokenizer paths are already available
    in the environment or via repo defaults

Environment overrides:
  PYTHON_BIN                Explicit Python executable to use.
  TORCH_CUDA_ARCH_LIST      Defaults to 9.0.
  GDN_MEGAKERNEL_ALLOW_JIT_BUILD Must remain 0 for this helper.
  GDN_MEGAKERNEL_REC_CHUNK_T Defaults to 8.
  MK_CANDIDATE_SPECS        Candidate matrix specs for mode=matrix.
                            Format: label:KEY=VALUE[,KEY=VALUE][;...]
                            Default:
                            base_rc8_v8:GDN_MEGAKERNEL_REC_CHUNK_T=8;
                            rc6_v8:GDN_MEGAKERNEL_REC_CHUNK_T=6;
                            rc4_v8:GDN_MEGAKERNEL_REC_CHUNK_T=4
  MK_MATRIX_CHILD_MODE      Child mode for mode=matrix, defaults to all.
  MK_MATRIX_CONTINUE_ON_ERROR Continue through later candidates when one fails,
                            defaults to 1.
  MK_TIMING_REPEATS         Defaults to 3.
  MK_CASES_DIR              Defaults to hgdn_megakernel/cases.
  MK_OUTPUT_DIR             Defaults to artifacts/hgdn_megakernel/<run>.
  MK_ARCHIVE_OUTPUT         Defaults to <MK_OUTPUT_DIR>.7z.
  RUN_PREFIX                Base prefix for trainer smoke run ids.
  RUN_ID                    Explicit trainer smoke run id.
  TORCH_LOGS                Optional torch.compile diagnostics. Unset by default;
                            set explicitly for graph-break / recompile debugging.
  TORCHINDUCTOR_CACHE_DIR   Defaults to /tmp/<run-id>_inductor.
  ITERATIONS                Defaults to 5 for trainer-smoke.
  COMPILE_WARMUP_STEPS      Defaults to 2 for trainer-smoke.
  USE_WANDB                 Defaults to 0.
  WANDB_MODE                Defaults to offline.
  WANDB_WATCH               Defaults to none.
  DRY_RUN                   If 1, print commands without executing them.

Examples:
  scripts/run_h100_single_gpu_hgdn_megakernel.sh all
  scripts/run_h100_single_gpu_hgdn_megakernel.sh matrix
  scripts/run_h100_single_gpu_hgdn_megakernel.sh compare100
  TORCH_CUDA_ARCH_LIST=8.9 scripts/run_h100_single_gpu_hgdn_megakernel.sh parity
  PYTHON_BIN=python3 scripts/run_h100_single_gpu_hgdn_megakernel.sh all
  GDN_MEGAKERNEL_REC_CHUNK_T=4 scripts/run_h100_single_gpu_hgdn_megakernel.sh trainer-smoke
  MK_CANDIDATE_SPECS='base_rc8_v8:GDN_MEGAKERNEL_REC_CHUNK_T=8;rc4_v8:GDN_MEGAKERNEL_REC_CHUNK_T=4' scripts/run_h100_single_gpu_hgdn_megakernel.sh matrix
  MK_MATRIX_CHILD_MODE=trainer-smoke ITERATIONS=100 MAX_WALLCLOCK_SECONDS=0 scripts/run_h100_single_gpu_hgdn_megakernel.sh compare100
  MK_OUTPUT_DIR=/tmp/h100mk_case MK_ARCHIVE_OUTPUT=/tmp/h100mk_case.7z scripts/run_h100_single_gpu_hgdn_megakernel.sh all
  DRY_RUN=1 scripts/run_h100_single_gpu_hgdn_megakernel.sh all
EOF
}

resolve_python_cmd() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "${PYTHON_BIN}"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        echo "python"
        return 0
    fi
    echo "Missing required Python runtime. Set PYTHON_BIN or install python3/python." >&2
    exit 1
}

read -r -a python_cmd <<<"$(resolve_python_cmd)"
python_cmd_rendered="$(printf '%q ' "${python_cmd[@]}")"
torch_arch_list="${TORCH_CUDA_ARCH_LIST:-9.0}"
torch_logs="${TORCH_LOGS-}"
rec_chunk_t="${GDN_MEGAKERNEL_REC_CHUNK_T:-8}"
timing_repeats="${MK_TIMING_REPEATS:-3}"
cases_dir="${MK_CASES_DIR:-hgdn_megakernel/cases}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_prefix="${RUN_PREFIX:-h100mk_${run_stamp}}"
trainer_run_id="${RUN_ID:-${run_prefix}_compile_parity_smoke_rc${rec_chunk_t}}"
trainer_cache_dir="${TORCHINDUCTOR_CACHE_DIR:-/tmp/${trainer_run_id}_inductor}"
bundle_name="${run_prefix}_${mode}_rc${rec_chunk_t}"
output_dir="${MK_OUTPUT_DIR:-artifacts/hgdn_megakernel/${bundle_name}}"
archive_output="${MK_ARCHIVE_OUTPUT:-${output_dir}.7z}"
commands_file="${output_dir}/commands.sh"
metadata_file="${output_dir}/metadata.txt"
git_commit="$(git rev-parse HEAD)"
git_branch="$(git rev-parse --abbrev-ref HEAD)"
host_name="$(hostname)"
timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
allow_jit_build="${GDN_MEGAKERNEL_ALLOW_JIT_BUILD:-0}"
default_matrix_candidate_specs="base_rc8_v8:GDN_MEGAKERNEL_REC_CHUNK_T=8;rc6_v8:GDN_MEGAKERNEL_REC_CHUNK_T=6;rc4_v8:GDN_MEGAKERNEL_REC_CHUNK_T=4"
default_compare100_candidate_specs="${default_matrix_candidate_specs}"
_mk_bundle_done=0
_mk_exit_status=0

if [[ "${allow_jit_build}" != "0" ]]; then
    echo "Refusing H100 helper run with GDN_MEGAKERNEL_ALLOW_JIT_BUILD=${allow_jit_build}" >&2
    exit 1
fi
export GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0

if [[ "${mode}" != "matrix" && "${mode}" != "compare100" ]]; then
    mkdir -p "${output_dir}"
    : > "${commands_file}"
    cat > "${metadata_file}" <<EOF
mode=${mode}
run_prefix=${run_prefix}
trainer_run_id=${trainer_run_id}
torch_cuda_arch_list=${torch_arch_list}
gdn_megakernel_rec_chunk_t=${rec_chunk_t}
mk_timing_repeats=${timing_repeats}
mk_cases_dir=${cases_dir}
torchinductor_cache_dir=${trainer_cache_dir}
mk_archive_output=${archive_output}
python_cmd=${python_cmd_rendered% }
gdn_megakernel_allow_jit_build=${allow_jit_build}
git_commit=${git_commit}
git_branch=${git_branch}
host_name=${host_name}
timestamp_utc=${timestamp_utc}
EOF
    chmod +x "${commands_file}"
fi

python_has_module() {
    local module_name="${1:?module name required}"
    "${python_cmd[@]}" scripts/hgdn_helper_cli.py module-exists --module "${module_name}" >/dev/null 2>&1
}

ensure_python_module() {
    local module_name="${1:?module name required}"
    local package_name="${2:-$module_name}"
    if python_has_module "${module_name}"; then
        return 0
    fi
    echo
    echo ">>> install python package: ${package_name}"
    "${python_cmd[@]}" -m pip install "${package_name}"
}

create_7z_archive() {
    local archive_path="${1:?archive path required}"
    local source_path="${2:?source path required}"
    rm -f "${archive_path}"
    mkdir -p "$(dirname "${archive_path}")"
    "${python_cmd[@]}" scripts/hgdn_helper_cli.py create-7z \
        --archive-output "${archive_path}" \
        --source-path "${source_path}"
}

json_escape() {
    local value="${1-}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$'\n'/\\n}"
    value="${value//$'\r'/\\r}"
    value="${value//$'\t'/\\t}"
    printf '%s' "${value}"
}

run_matrix() {
    local matrix_mode_label="${1:-matrix}"
    local matrix_child_mode="${MK_MATRIX_CHILD_MODE:-all}"
    local matrix_continue_on_error="${MK_MATRIX_CONTINUE_ON_ERROR:-1}"
    local matrix_candidate_specs="${MK_CANDIDATE_SPECS:-${default_matrix_candidate_specs}}"
    local matrix_output_dir_default="artifacts/hgdn_megakernel/${run_prefix}_matrix"
    local -a matrix_shared_env
    matrix_shared_env=()
    if [[ "${matrix_mode_label}" == "compare100" ]]; then
        if [[ -z "${MK_MATRIX_CHILD_MODE:-}" ]]; then
            matrix_child_mode="all"
        fi
        if [[ -z "${MK_CANDIDATE_SPECS:-}" ]]; then
            matrix_candidate_specs="${default_compare100_candidate_specs}"
        fi
        matrix_shared_env+=("ITERATIONS=${ITERATIONS:-100}")
        matrix_shared_env+=("COMPILE_WARMUP_STEPS=${COMPILE_WARMUP_STEPS:-20}")
        matrix_shared_env+=("TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-25}")
        matrix_shared_env+=("MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-0}")
        matrix_output_dir_default="artifacts/hgdn_megakernel/${run_prefix}_compare100"
    fi
    local matrix_output_dir="${MK_OUTPUT_DIR:-${matrix_output_dir_default}}"
    local matrix_archive_output="${MK_ARCHIVE_OUTPUT:-${matrix_output_dir}.7z}"
    local matrix_commands_file="${matrix_output_dir}/commands.sh"
    local matrix_metadata_file="${matrix_output_dir}/metadata.txt"
    local matrix_summary_file="${matrix_output_dir}/matrix_summary.tsv"
    local matrix_manifest_file="${matrix_output_dir}/matrix_manifest.json"

    case "${matrix_child_mode}" in
        parity|trainer-smoke|all)
            ;;
        *)
            echo "Unsupported MK_MATRIX_CHILD_MODE=${matrix_child_mode}; expected parity, trainer-smoke, or all" >&2
            exit 1
            ;;
    esac
    if [[ "${matrix_continue_on_error}" != "0" && "${matrix_continue_on_error}" != "1" ]]; then
        echo "Unsupported MK_MATRIX_CONTINUE_ON_ERROR=${matrix_continue_on_error}; expected 0 or 1" >&2
        exit 1
    fi
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        ensure_python_module py7zr py7zr
    fi

    mkdir -p "${matrix_output_dir}"
    : > "${matrix_commands_file}"
    chmod +x "${matrix_commands_file}"
    cat > "${matrix_metadata_file}" <<EOF
mode=${matrix_mode_label}
run_prefix=${run_prefix}
matrix_child_mode=${matrix_child_mode}
matrix_continue_on_error=${matrix_continue_on_error}
matrix_candidate_specs=${matrix_candidate_specs}
torch_cuda_arch_list=${torch_arch_list}
mk_timing_repeats=${timing_repeats}
mk_cases_dir=${cases_dir}
python_cmd=${python_cmd_rendered% }
gdn_megakernel_allow_jit_build=${allow_jit_build}
git_commit=${git_commit}
git_branch=${git_branch}
host_name=${host_name}
timestamp_utc=${timestamp_utc}
EOF
    printf 'label\tstatus\texit_code\tmode\toutput_dir\tarchive_output\toverrides\n' > "${matrix_summary_file}"

    IFS=';' read -r -a candidate_specs <<< "${matrix_candidate_specs}"
    if [[ "${#candidate_specs[@]}" -eq 0 ]]; then
        echo "MK_CANDIDATE_SPECS resolved to zero candidates" >&2
        exit 1
    fi

    local overall_status=0
    local manifest_candidates=""
    local manifest_separator=""

    for raw_spec in "${candidate_specs[@]}"; do
        local spec="${raw_spec}"
        if [[ -z "${spec// }" ]]; then
            continue
        fi
        if [[ "${spec}" != *:* ]]; then
            echo "Invalid candidate spec: ${spec}. Expected label:KEY=VALUE[,KEY=VALUE]" >&2
            overall_status=1
            if [[ "${matrix_continue_on_error}" == "1" ]]; then
                continue
            fi
            break
        fi

        local label="${spec%%:*}"
        local overrides_csv="${spec#*:}"
        local child_output_dir="${matrix_output_dir}/${label}"
        local child_archive_output="${matrix_output_dir}/${label}.7z"
        local child_run_prefix="${run_prefix}_${label}"
        local child_exit_code=0
        local child_status="ok"
        local -a child_env
        child_env=(
            "RUN_PREFIX=${child_run_prefix}"
            "MK_OUTPUT_DIR=${child_output_dir}"
            "MK_ARCHIVE_OUTPUT=${child_archive_output}"
        )
        child_env+=("${matrix_shared_env[@]}")

        IFS=',' read -r -a override_pairs <<< "${overrides_csv}"
        for override in "${override_pairs[@]}"; do
            if [[ -z "${override// }" ]]; then
                continue
            fi
            if [[ "${override}" != *=* ]]; then
                echo "Invalid override in candidate ${label}: ${override}" >&2
                overall_status=1
                child_exit_code=2
                child_status="invalid_override"
                break
            fi
            child_env+=("${override}")
        done
        if [[ "${child_exit_code}" != "0" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${label}" "${child_status}" "${child_exit_code}" "${matrix_child_mode}" \
                "${child_output_dir}" "${child_archive_output}" "${overrides_csv}" >> "${matrix_summary_file}"
            manifest_candidates+="${manifest_separator}{\"label\":\"$(json_escape "${label}")\",\"status\":\"$(json_escape "${child_status}")\",\"exit_code\":${child_exit_code},\"mode\":\"$(json_escape "${matrix_child_mode}")\",\"output_dir\":\"$(json_escape "${child_output_dir}")\",\"archive_output\":\"$(json_escape "${child_archive_output}")\",\"overrides\":\"$(json_escape "${overrides_csv}")\"}"
            manifest_separator=","
            if [[ "${matrix_continue_on_error}" == "1" ]]; then
                continue
            fi
            break
        fi

        echo
        echo "### HGDN megakernel matrix candidate"
        echo "label=${label} mode=${matrix_child_mode} overrides=${overrides_csv}"
        local -a child_cmd
        child_cmd=("env")
        child_cmd+=("${child_env[@]}")
        child_cmd+=("bash" "${HGDN_REPO_ROOT}/scripts/run_h100_single_gpu_hgdn_megakernel.sh" "${matrix_child_mode}")
        hgdn_append_plain_command "${matrix_commands_file}" "${child_cmd[@]}"
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            printf '>>> '
            printf '%q ' "${child_cmd[@]}"
            printf '\n'
        else
            if "${child_cmd[@]}"; then
                child_exit_code=0
                child_status="ok"
            else
                child_exit_code=$?
                child_status="failed"
                overall_status=1
            fi
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${label}" "${child_status}" "${child_exit_code}" "${matrix_child_mode}" \
            "${child_output_dir}" "${child_archive_output}" "${overrides_csv}" >> "${matrix_summary_file}"
        manifest_candidates+="${manifest_separator}{\"label\":\"$(json_escape "${label}")\",\"status\":\"$(json_escape "${child_status}")\",\"exit_code\":${child_exit_code},\"mode\":\"$(json_escape "${matrix_child_mode}")\",\"output_dir\":\"$(json_escape "${child_output_dir}")\",\"archive_output\":\"$(json_escape "${child_archive_output}")\",\"overrides\":\"$(json_escape "${overrides_csv}")\"}"
        manifest_separator=","

        if [[ "${child_exit_code}" != "0" && "${matrix_continue_on_error}" != "1" ]]; then
            break
        fi
    done

    cat > "${matrix_manifest_file}" <<EOF
{
  "mode": "$(json_escape "${matrix_mode_label}")",
  "run_prefix": "$(json_escape "${run_prefix}")",
  "matrix_child_mode": "$(json_escape "${matrix_child_mode}")",
  "matrix_continue_on_error": ${matrix_continue_on_error},
  "torch_cuda_arch_list": "$(json_escape "${torch_arch_list}")",
  "mk_timing_repeats": ${timing_repeats},
  "mk_cases_dir": "$(json_escape "${cases_dir}")",
  "archive_output": "$(json_escape "${matrix_archive_output}")",
  "git_commit": "$(json_escape "${git_commit}")",
  "git_branch": "$(json_escape "${git_branch}")",
  "host_name": "$(json_escape "${host_name}")",
  "timestamp_utc": "$(json_escape "${timestamp_utc}")",
  "candidates": [
    ${manifest_candidates}
  ]
}
EOF

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo
        echo "matrix_bundle_dir=${matrix_output_dir}"
        echo "matrix_bundle_archive=${matrix_archive_output}"
        return "${overall_status}"
    fi

    create_7z_archive "${matrix_archive_output}" "${matrix_output_dir}"
    echo
    echo "matrix_bundle_dir=${matrix_output_dir}"
    echo "matrix_bundle_archive=${matrix_archive_output}"
    return "${overall_status}"
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    ensure_python_module py7zr py7zr
fi

run_cmd() {
    local logfile="$1"
    shift
    echo
    printf '>>> '
    printf '%q ' "$@"
    printf '\n'
    hgdn_append_plain_command "${commands_file}" "$@"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    (
        set -o pipefail
        "$@" 2>&1 | tee "${logfile}"
    )
}

build_bundle() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo
        echo "bundle_dir=${output_dir}"
        echo "bundle_archive=${archive_output}"
        return 0
    fi

    "${python_cmd[@]}" scripts/hgdn_helper_cli.py write-megakernel-bundle \
        --bundle-dir "${output_dir}" \
        --mode "${mode}" \
        --run-prefix "${run_prefix}" \
        --trainer-run-id "${trainer_run_id}" \
        --torch-cuda-arch-list "${torch_arch_list}" \
        --rec-chunk-t "${rec_chunk_t}" \
        --timing-repeats "${timing_repeats}" \
        --cases-dir "${cases_dir}" \
        --trainer-cache-dir "${trainer_cache_dir}" \
        --archive-output "${archive_output}" \
        --commands-file "${commands_file}" \
        --metadata-path "${metadata_file}" \
        --torch-logs "${torch_logs}" \
        --iterations "${ITERATIONS:-5}" \
        --compile-warmup-steps "${COMPILE_WARMUP_STEPS:-2}" \
        --train-seq-len "${TRAIN_SEQ_LEN:-2048}" \
        --train-batch-tokens "${TRAIN_BATCH_TOKENS:-524288}" \
        --val-loss-every "${VAL_LOSS_EVERY:-0}" \
        --train-log-every "${TRAIN_LOG_EVERY:-1}" \
        --git-commit "${git_commit}" \
        --git-branch "${git_branch}" \
        --host-name "${host_name}" \
        --timestamp-utc "${timestamp_utc}" \
        --exit-status "${_mk_exit_status}"

    create_7z_archive "${archive_output}" "${output_dir}"
    echo
    echo "bundle_dir=${output_dir}"
    echo "bundle_archive=${archive_output}"
}

build_bundle_once() {
    if [[ "${_mk_bundle_done}" == "1" ]]; then
        return 0
    fi
    _mk_bundle_done=1
    build_bundle || true
}

run_parity() {
    echo
    echo "### HGDN megakernel parity gate"
    echo "arch_list=${torch_arch_list} rec_chunk_t=${rec_chunk_t} timing_repeats=${timing_repeats}"
    echo "output_dir=${output_dir}"
    if [[ "${torch_arch_list}" != "9.0" && "${torch_arch_list}" != "9.0a" ]]; then
        echo "note: non-H100 arch override detected; treat this as command-path/correctness validation only"
    fi
    run_cmd "${output_dir}/build.log" env GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0 TORCH_CUDA_ARCH_LIST="${torch_arch_list}" "${python_cmd[@]}" \
        setup_hgdn_megakernel.py build_ext --inplace
    run_cmd "${output_dir}/audit.log" env GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0 "${python_cmd[@]}" scripts/audit_hgdn_megakernel_contract.py
    run_cmd "${output_dir}/parity.log" env GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0 "${python_cmd[@]}" hgdn_megakernel/test_megakernel.py \
        --case-dir "${cases_dir}" \
        --timing-repeats "${timing_repeats}" \
        --rec-chunk-t "${rec_chunk_t}" \
        --include-b2-t512 \
        --include-b1-t2048
}

run_trainer_smoke() {
    echo
    echo "### HGDN megakernel trainer smoke"
    echo "run_id=${trainer_run_id} arch_list=${torch_arch_list} rec_chunk_t=${rec_chunk_t}"
    echo "output_dir=${output_dir}"
    local -a train_env
    train_env=(
        env
        GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0
        TORCH_CUDA_ARCH_LIST="${torch_arch_list}"
        TORCHINDUCTOR_CACHE_DIR="${trainer_cache_dir}"
        RUN_ID="${trainer_run_id}"
        USE_WANDB="${USE_WANDB:-0}"
        WANDB_MODE="${WANDB_MODE:-offline}"
        WANDB_WATCH="${WANDB_WATCH:-none}"
        GDN_USE_CUDA_MEGAKERNEL=1
        GDN_MEGAKERNEL_REC_CHUNK_T="${rec_chunk_t}"
        GDN_CONTROL_PROJ_FP32=0
        GDN_USE_PACKED_QKV_CONV=1
        GDN_USE_PACKED_QKV_PROJ=1
        GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=0
        GDN_CONV_OUTPUT_CONTIGUOUS=1
        NUM_LAYERS="${NUM_LAYERS:-14}"
        MODEL_DIM="${MODEL_DIM:-384}"
        MLP_MULT="${MLP_MULT:-3.25}"
        GDN_RATIO="${GDN_RATIO:-1}"
        TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
        TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
        ITERATIONS="${ITERATIONS:-5}"
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
        TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}"
        PERF_SKIP_FINAL_EVAL="${PERF_SKIP_FINAL_EVAL:-1}"
        COMPILE="${COMPILE:-1}"
        COMPILE_STRATEGY="${COMPILE_STRATEGY:-hybrid}"
        COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS:-2}"
    )
    if [[ -n "${TORCH_LOGS+x}" ]]; then
        train_env+=(TORCH_LOGS="${TORCH_LOGS}")
    fi
    run_cmd "${output_dir}/trainer_smoke.log" "${train_env[@]}" \
        "${python_cmd[@]}" -m torch.distributed.run --standalone --nproc_per_node=1 \
        train_gpt_hybrid.py
}

if [[ "${mode}" == "help" || "${mode}" == "-h" || "${mode}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${mode}" == "matrix" || "${mode}" == "compare100" ]]; then
    run_matrix "${mode}"
    exit $?
fi

trap '_mk_exit_status=$?; build_bundle_once; exit ${_mk_exit_status}' EXIT

case "${mode}" in
parity)
    run_parity
    ;;
trainer-smoke)
    run_trainer_smoke
    ;;
all)
    run_parity
    run_trainer_smoke
    ;;
*)
    echo "Unknown mode: ${mode}" >&2
    usage >&2
    exit 1
    ;;
esac
