#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

mode="${1:-all}"

usage() {
    cat <<'EOF'
Usage: scripts/run_h100_single_gpu_hgdn_corekernel.sh {parity|trainer-smoke|all|compare100|help}

Purpose:
  Bounded 1xH100 helper for the active HGDN core-kernel direction.
  This helper is intentionally narrow:
  - build the extension for the requested CUDA arch list
  - run the static contract audit
  - run the isolated core-kernel parity / owned-launch / timing harness
  - optionally run one short trainer smoke in core-kernel mode
  - optionally run one fixed-step compare against the packed control

  The old full-block megakernel is research-only. It can still be compared by
  explicit candidate override, but it is not the default path here.

Modes:
  parity
    Build the extension, run the static audit, then run the isolated
    core-kernel parity / owned-launch / timing harness with:
    - B=1,T=8/32/128/512
    - optional B=2,T=512
    - optional B=1,T=2048
    - timing repeats defaulting to 3

  trainer-smoke
    Run one bounded trainer smoke. Defaults to the HGDN core-kernel path:
    - GDN_USE_CUDA_COREKERNEL=1
    - GDN_USE_CUDA_MEGAKERNEL=0
    - GDN_CONTROL_PROJ_FP32=0
    - packed QKV projection + packed QKV conv enabled

    Override `GDN_USE_CUDA_COREKERNEL` / `GDN_USE_CUDA_MEGAKERNEL` explicitly
    if you need a packed-control or full-block research run instead.

  all
    Run parity first, then the bounded trainer smoke.

  compare100
    Run parity once, then run a fixed-step candidate comparison under:
    - ITERATIONS=100
    - COMPILE_WARMUP_STEPS=20
    - TRAIN_LOG_EVERY=25
    - MAX_WALLCLOCK_SECONDS=0

    Default candidates:
    - packed_control:
        GDN_USE_CUDA_COREKERNEL=0,GDN_USE_CUDA_MEGAKERNEL=0
    - core_rc8:
        GDN_USE_CUDA_COREKERNEL=1,GDN_USE_CUDA_MEGAKERNEL=0,GDN_MEGAKERNEL_REC_CHUNK_T=8

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
  GDN_MEGAKERNEL_REC_CHUNK_T Shared runtime checkpoint cadence knob, defaults to 8.
  HK_CASES_DIR              Defaults to hgdn_megakernel/cases.
  HK_TIMING_REPEATS         Defaults to 3.
  HK_OUTPUT_DIR             Defaults to artifacts/hgdn_corekernel/<run>.
  HK_ARCHIVE_OUTPUT         Defaults to <HK_OUTPUT_DIR>.7z.
  HK_CANDIDATE_SPECS        Candidate specs for compare100.
                            Format: label:KEY=VALUE[,KEY=VALUE][;...]
                            Default:
                            packed_control:GDN_USE_CUDA_COREKERNEL=0,GDN_USE_CUDA_MEGAKERNEL=0;
                            core_rc8:GDN_USE_CUDA_COREKERNEL=1,GDN_USE_CUDA_MEGAKERNEL=0,GDN_MEGAKERNEL_REC_CHUNK_T=8
  RUN_PREFIX                Base prefix for trainer smoke run ids.
  RUN_ID                    Explicit trainer smoke run id.
  TORCH_LOGS                Defaults to graph_breaks,recompiles.
  TORCHINDUCTOR_CACHE_DIR   Defaults to /tmp/<run-id>_inductor.
  ITERATIONS                Defaults to 5 for trainer-smoke, 100 for compare100.
  COMPILE_WARMUP_STEPS      Defaults to 2 for trainer-smoke, 20 for compare100.
  TRAIN_LOG_EVERY           Defaults to 1 for trainer-smoke, 25 for compare100.
  MAX_WALLCLOCK_SECONDS     Defaults to 0 for compare100.
  USE_WANDB                 Defaults to 0.
  WANDB_MODE                Defaults to offline.
  WANDB_WATCH               Defaults to none.
  DRY_RUN                   If 1, print commands without executing them.

Examples:
  scripts/run_h100_single_gpu_hgdn_corekernel.sh all
  scripts/run_h100_single_gpu_hgdn_corekernel.sh compare100
  TORCH_CUDA_ARCH_LIST=8.9 scripts/run_h100_single_gpu_hgdn_corekernel.sh parity
  PYTHON_BIN=python3 scripts/run_h100_single_gpu_hgdn_corekernel.sh all
  GDN_MEGAKERNEL_REC_CHUNK_T=4 scripts/run_h100_single_gpu_hgdn_corekernel.sh trainer-smoke
  HK_CANDIDATE_SPECS='packed_control:GDN_USE_CUDA_COREKERNEL=0,GDN_USE_CUDA_MEGAKERNEL=0;core_rc4:GDN_USE_CUDA_COREKERNEL=1,GDN_USE_CUDA_MEGAKERNEL=0,GDN_MEGAKERNEL_REC_CHUNK_T=4' scripts/run_h100_single_gpu_hgdn_corekernel.sh compare100
  HK_OUTPUT_DIR=/tmp/h100core_case HK_ARCHIVE_OUTPUT=/tmp/h100core_case.7z scripts/run_h100_single_gpu_hgdn_corekernel.sh compare100
  DRY_RUN=1 scripts/run_h100_single_gpu_hgdn_corekernel.sh compare100
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
rec_chunk_t="${GDN_MEGAKERNEL_REC_CHUNK_T:-8}"
timing_repeats="${HK_TIMING_REPEATS:-3}"
cases_dir="${HK_CASES_DIR:-hgdn_megakernel/cases}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
run_prefix="${RUN_PREFIX:-h100core_${run_stamp}}"
trainer_run_id="${RUN_ID:-${run_prefix}_compile_smoke}"
trainer_cache_dir="${TORCHINDUCTOR_CACHE_DIR:-/tmp/${trainer_run_id}_inductor}"
bundle_name="${run_prefix}_${mode}"
output_dir="${HK_OUTPUT_DIR:-artifacts/hgdn_corekernel/${bundle_name}}"
archive_output="${HK_ARCHIVE_OUTPUT:-${output_dir}.7z}"
commands_file="${output_dir}/commands.sh"
metadata_file="${output_dir}/metadata.txt"
git_commit="$(git rev-parse HEAD)"
git_branch="$(git rev-parse --abbrev-ref HEAD)"
host_name="$(hostname)"
timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
allow_jit_build="${GDN_MEGAKERNEL_ALLOW_JIT_BUILD:-0}"
default_compare100_candidate_specs="packed_control:GDN_USE_CUDA_COREKERNEL=0,GDN_USE_CUDA_MEGAKERNEL=0;core_rc8:GDN_USE_CUDA_COREKERNEL=1,GDN_USE_CUDA_MEGAKERNEL=0,GDN_MEGAKERNEL_REC_CHUNK_T=8"
_hk_bundle_done=0
_hk_exit_status=0

if [[ "${allow_jit_build}" != "0" ]]; then
    echo "Refusing H100 helper run with GDN_MEGAKERNEL_ALLOW_JIT_BUILD=${allow_jit_build}" >&2
    exit 1
fi
export GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0

mkdir -p "${output_dir}"
: > "${commands_file}"
chmod +x "${commands_file}"
cat > "${metadata_file}" <<EOF
mode=${mode}
run_prefix=${run_prefix}
trainer_run_id=${trainer_run_id}
torch_cuda_arch_list=${torch_arch_list}
gdn_megakernel_rec_chunk_t=${rec_chunk_t}
hk_timing_repeats=${timing_repeats}
hk_cases_dir=${cases_dir}
torchinductor_cache_dir=${trainer_cache_dir}
hk_archive_output=${archive_output}
python_cmd=${python_cmd_rendered% }
gdn_megakernel_allow_jit_build=${allow_jit_build}
git_commit=${git_commit}
git_branch=${git_branch}
host_name=${host_name}
timestamp_utc=${timestamp_utc}
EOF

python_has_module() {
    local module_name="${1:?module name required}"
    "${python_cmd[@]}" - "${module_name}" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
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
    "${python_cmd[@]}" - "${archive_path}" "${source_path}" <<'PY'
from pathlib import Path
import sys

import py7zr

archive_output = Path(sys.argv[1])
source_path = Path(sys.argv[2])

with py7zr.SevenZipFile(archive_output, "w") as archive:
    archive.writeall(source_path, arcname=source_path.name)
PY
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

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    ensure_python_module py7zr py7zr
fi

run_cmd() {
    local logfile="$1"
    shift
    mkdir -p "$(dirname "${logfile}")"
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

append_metadata_line() {
    printf '%s\n' "$1" >> "${metadata_file}"
}

build_bundle() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo
        echo "bundle_dir=${output_dir}"
        echo "bundle_archive=${archive_output}"
        return 0
    fi

    "${python_cmd[@]}" - \
        "${output_dir}" \
        "${mode}" \
        "${run_prefix}" \
        "${trainer_run_id}" \
        "${torch_arch_list}" \
        "${rec_chunk_t}" \
        "${timing_repeats}" \
        "${cases_dir}" \
        "${trainer_cache_dir}" \
        "${archive_output}" \
        "${commands_file}" \
        "${metadata_file}" \
        "${TORCH_LOGS:-graph_breaks,recompiles}" \
        "${ITERATIONS:-5}" \
        "${COMPILE_WARMUP_STEPS:-2}" \
        "${TRAIN_SEQ_LEN:-2048}" \
        "${TRAIN_BATCH_TOKENS:-524288}" \
        "${VAL_LOSS_EVERY:-0}" \
        "${TRAIN_LOG_EVERY:-1}" \
        "${git_commit}" \
        "${git_branch}" \
        "${host_name}" \
        "${timestamp_utc}" \
        "${_hk_exit_status}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

bundle_dir = Path(sys.argv[1])
mode = sys.argv[2]
run_prefix = sys.argv[3]
trainer_run_id = sys.argv[4]
torch_cuda_arch_list = sys.argv[5]
rec_chunk_t = int(sys.argv[6])
timing_repeats = int(sys.argv[7])
cases_dir = sys.argv[8]
trainer_cache_dir = sys.argv[9]
archive_output = sys.argv[10]
commands_file = Path(sys.argv[11]).name
metadata_path = Path(sys.argv[12])
metadata_file = metadata_path.name
torch_logs = sys.argv[13]
iterations = int(sys.argv[14])
compile_warmup_steps = int(sys.argv[15])
train_seq_len = int(sys.argv[16])
train_batch_tokens = int(sys.argv[17])
val_loss_every = int(sys.argv[18])
train_log_every = int(sys.argv[19])
git_commit = sys.argv[20]
git_branch = sys.argv[21]
host_name = sys.argv[22]
timestamp_utc = sys.argv[23]
exit_status = int(sys.argv[24])

logs = sorted(
    str(path.relative_to(bundle_dir))
    for path in bundle_dir.rglob("*.log")
    if path.is_file()
)
extension_status = None
try:
    from hgdn_megakernel import extension_status as get_extension_status

    extension_status = get_extension_status()
except Exception as exc:  # pragma: no cover - best-effort bundle metadata
    extension_status = {"error": str(exc), "loaded": False}

with metadata_path.open("a", encoding="utf-8") as fh:
    fh.write(f"bundle_exit_status={exit_status}\n")
    if extension_status is not None:
        fh.write(
            "extension_status_json="
            + json.dumps(extension_status, sort_keys=True)
            + "\n"
        )

manifest = {
    "mode": mode,
    "run_prefix": run_prefix,
    "trainer_run_id": trainer_run_id,
    "archive_output": archive_output,
    "exit_status": exit_status,
    "paths": {
        "commands": commands_file,
        "metadata": metadata_file,
        "logs": logs,
    },
    "contract": {
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "gdn_megakernel_rec_chunk_t": rec_chunk_t,
        "hk_timing_repeats": timing_repeats,
        "hk_cases_dir": cases_dir,
        "torchinductor_cache_dir": trainer_cache_dir,
        "torch_logs": torch_logs or None,
        "iterations": iterations,
        "compile_warmup_steps": compile_warmup_steps,
        "train_seq_len": train_seq_len,
        "train_batch_tokens": train_batch_tokens,
        "val_loss_every": val_loss_every,
        "train_log_every": train_log_every,
    },
    "extension_status": extension_status,
    "provenance": {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "host_name": host_name,
        "timestamp_utc": timestamp_utc,
    },
}
(bundle_dir / "bundle_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

    create_7z_archive "${archive_output}" "${output_dir}"
    echo
    echo "bundle_dir=${output_dir}"
    echo "bundle_archive=${archive_output}"
}

build_bundle_once() {
    if [[ "${_hk_bundle_done}" == "1" ]]; then
        return 0
    fi
    _hk_bundle_done=1
    build_bundle || true
}

run_parity() {
    echo
    echo "### HGDN core-kernel parity gate"
    echo "arch_list=${torch_arch_list} rec_chunk_t=${rec_chunk_t} timing_repeats=${timing_repeats}"
    echo "output_dir=${output_dir}"
    if [[ "${torch_arch_list}" != "9.0" && "${torch_arch_list}" != "9.0a" ]]; then
        echo "note: non-H100 arch override detected; treat this as command-path/correctness validation only"
    fi
    run_cmd "${output_dir}/build.log" env GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0 TORCH_CUDA_ARCH_LIST="${torch_arch_list}" "${python_cmd[@]}" \
        setup_hgdn_megakernel.py build_ext --inplace
    run_cmd "${output_dir}/audit.log" env GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0 "${python_cmd[@]}" scripts/audit_hgdn_megakernel_contract.py
    run_cmd "${output_dir}/parity.log" env GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0 GDN_MEGAKERNEL_REC_CHUNK_T="${rec_chunk_t}" "${python_cmd[@]}" \
        hgdn_megakernel/test_corekernel.py \
        --case-dir "${cases_dir}" \
        --timing-repeats "${timing_repeats}" \
        --include-b2-t512 \
        --include-b1-t2048
}

run_trainer_smoke() {
    local logfile="${1:-${output_dir}/trainer_smoke.log}"
    shift || true
    local -a extra_env
    extra_env=("$@")
    local run_id="${RUN_ID:-${trainer_run_id}}"
    local cache_dir="${TORCHINDUCTOR_CACHE_DIR:-${trainer_cache_dir}}"
    local use_core="${GDN_USE_CUDA_COREKERNEL:-1}"
    local use_megakernel="${GDN_USE_CUDA_MEGAKERNEL:-0}"
    local runtime_rec_chunk_t="${GDN_MEGAKERNEL_REC_CHUNK_T:-${rec_chunk_t}}"
    local max_wallclock="${MAX_WALLCLOCK_SECONDS:-0}"

    for assignment in "${extra_env[@]}"; do
        case "${assignment}" in
            RUN_ID=*)
                run_id="${assignment#RUN_ID=}"
                ;;
            TORCHINDUCTOR_CACHE_DIR=*)
                cache_dir="${assignment#TORCHINDUCTOR_CACHE_DIR=}"
                ;;
            GDN_USE_CUDA_COREKERNEL=*)
                use_core="${assignment#GDN_USE_CUDA_COREKERNEL=}"
                ;;
            GDN_USE_CUDA_MEGAKERNEL=*)
                use_megakernel="${assignment#GDN_USE_CUDA_MEGAKERNEL=}"
                ;;
            GDN_MEGAKERNEL_REC_CHUNK_T=*)
                runtime_rec_chunk_t="${assignment#GDN_MEGAKERNEL_REC_CHUNK_T=}"
                ;;
            MAX_WALLCLOCK_SECONDS=*)
                max_wallclock="${assignment#MAX_WALLCLOCK_SECONDS=}"
                ;;
        esac
    done

    echo
    echo "### HGDN trainer smoke"
    echo "run_id=${run_id} arch_list=${torch_arch_list} core=${use_core} megakernel=${use_megakernel} rec_chunk_t=${runtime_rec_chunk_t}"
    echo "output_log=${logfile}"
    run_cmd "${logfile}" env \
        GDN_MEGAKERNEL_ALLOW_JIT_BUILD=0 \
        TORCH_CUDA_ARCH_LIST="${torch_arch_list}" \
        TORCHINDUCTOR_CACHE_DIR="${cache_dir}" \
        TORCH_LOGS="${TORCH_LOGS:-graph_breaks,recompiles}" \
        RUN_ID="${run_id}" \
        USE_WANDB="${USE_WANDB:-0}" \
        WANDB_MODE="${WANDB_MODE:-offline}" \
        WANDB_WATCH="${WANDB_WATCH:-none}" \
        GDN_USE_CUDA_COREKERNEL="${use_core}" \
        GDN_USE_CUDA_MEGAKERNEL="${use_megakernel}" \
        GDN_MEGAKERNEL_REC_CHUNK_T="${runtime_rec_chunk_t}" \
        GDN_CONTROL_PROJ_FP32=0 \
        GDN_USE_PACKED_QKV_CONV=1 \
        GDN_USE_PACKED_QKV_PROJ=1 \
        GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=0 \
        GDN_CONV_OUTPUT_CONTIGUOUS=1 \
        NUM_LAYERS="${NUM_LAYERS:-14}" \
        MODEL_DIM="${MODEL_DIM:-384}" \
        MLP_MULT="${MLP_MULT:-3.25}" \
        GDN_RATIO="${GDN_RATIO:-1}" \
        TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}" \
        TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}" \
        ITERATIONS="${ITERATIONS:-5}" \
        MAX_WALLCLOCK_SECONDS="${max_wallclock}" \
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
        TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}" \
        PERF_SKIP_FINAL_EVAL="${PERF_SKIP_FINAL_EVAL:-1}" \
        COMPILE="${COMPILE:-1}" \
        COMPILE_STRATEGY="${COMPILE_STRATEGY:-hybrid}" \
        COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS:-2}" \
        "${extra_env[@]}" \
        "${python_cmd[@]}" -m torch.distributed.run --standalone --nproc_per_node=1 \
        train_gpt_hybrid.py
}

run_compare100() {
    local candidate_specs="${HK_CANDIDATE_SPECS:-${default_compare100_candidate_specs}}"
    local summary_file="${output_dir}/compare100_summary.tsv"
    local manifest_file="${output_dir}/compare100_manifest.json"
    local compare_iterations="${ITERATIONS:-100}"
    local compare_warmup="${COMPILE_WARMUP_STEPS:-20}"
    local compare_train_log_every="${TRAIN_LOG_EVERY:-25}"
    local compare_max_wallclock="${MAX_WALLCLOCK_SECONDS:-0}"
    local candidate_root="${output_dir}/candidates"

    echo
    echo "### HGDN compare100"
    echo "candidate_specs=${candidate_specs}"
    append_metadata_line "compare100_candidate_specs=${candidate_specs}"
    append_metadata_line "compare100_iterations=${compare_iterations}"
    append_metadata_line "compare100_compile_warmup_steps=${compare_warmup}"
    append_metadata_line "compare100_train_log_every=${compare_train_log_every}"
    append_metadata_line "compare100_max_wallclock_seconds=${compare_max_wallclock}"

    printf 'label\tstatus\texit_code\toutput_dir\toverrides\n' > "${summary_file}"

    IFS=';' read -r -a candidate_list <<< "${candidate_specs}"
    local manifest_candidates=""
    local manifest_separator=""
    local overall_status=0

    for raw_spec in "${candidate_list[@]}"; do
        local spec="${raw_spec}"
        if [[ -z "${spec// }" ]]; then
            continue
        fi
        if [[ "${spec}" != *:* ]]; then
            echo "Invalid candidate spec: ${spec}. Expected label:KEY=VALUE[,KEY=VALUE]" >&2
            overall_status=1
            continue
        fi
        local label="${spec%%:*}"
        local overrides_csv="${spec#*:}"
        local child_dir="${candidate_root}/${label}"
        local child_log="${child_dir}/trainer_smoke.log"
        local child_run_id="${run_prefix}_${label}_compare100"
        local child_exit_code=0
        local child_status="ok"
        local -a override_env
        override_env=(
            "RUN_ID=${child_run_id}"
            "TORCHINDUCTOR_CACHE_DIR=/tmp/${child_run_id}_inductor"
            "ITERATIONS=${compare_iterations}"
            "COMPILE_WARMUP_STEPS=${compare_warmup}"
            "TRAIN_LOG_EVERY=${compare_train_log_every}"
            "MAX_WALLCLOCK_SECONDS=${compare_max_wallclock}"
        )

        IFS=',' read -r -a override_pairs <<< "${overrides_csv}"
        for override in "${override_pairs[@]}"; do
            if [[ -z "${override// }" ]]; then
                continue
            fi
            if [[ "${override}" != *=* ]]; then
                echo "Invalid override in candidate ${label}: ${override}" >&2
                child_exit_code=2
                child_status="invalid_override"
                break
            fi
            override_env+=("${override}")
        done

        echo
        echo "### compare100 candidate"
        echo "label=${label} overrides=${overrides_csv}"
        if [[ "${child_exit_code}" == "0" ]]; then
            mkdir -p "${child_dir}"
            if run_trainer_smoke "${child_log}" "${override_env[@]}"; then
                child_exit_code=0
            else
                child_exit_code=$?
                child_status="failed"
                overall_status=1
            fi
        else
            overall_status=1
        fi

        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${label}" "${child_status}" "${child_exit_code}" "${child_dir}" "${overrides_csv}" >> "${summary_file}"
        manifest_candidates+="${manifest_separator}{\"label\":\"$(json_escape "${label}")\",\"status\":\"$(json_escape "${child_status}")\",\"exit_code\":${child_exit_code},\"output_dir\":\"$(json_escape "${child_dir}")\",\"overrides\":\"$(json_escape "${overrides_csv}")\"}"
        manifest_separator=","
    done

    cat > "${manifest_file}" <<EOF
{
  "mode": "compare100",
  "run_prefix": "$(json_escape "${run_prefix}")",
  "torch_cuda_arch_list": "$(json_escape "${torch_arch_list}")",
  "compare_iterations": ${compare_iterations},
  "compare_compile_warmup_steps": ${compare_warmup},
  "compare_train_log_every": ${compare_train_log_every},
  "compare_max_wallclock_seconds": ${compare_max_wallclock},
  "candidates": [
    ${manifest_candidates}
  ]
}
EOF

    return "${overall_status}"
}

if [[ "${mode}" == "help" || "${mode}" == "-h" || "${mode}" == "--help" ]]; then
    usage
    exit 0
fi

trap '_hk_exit_status=$?; build_bundle_once; exit ${_hk_exit_status}' EXIT

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
compare100)
    ITERATIONS="${ITERATIONS:-100}"
    COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS:-20}"
    TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-25}"
    MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
    run_parity
    run_compare100
    ;;
*)
    echo "Unknown mode: ${mode}" >&2
    usage >&2
    exit 1
    ;;
esac
