#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

mode="${1:-all}"

usage() {
    cat <<'EOF'
Usage: scripts/run_h100_single_gpu_hgdn_megakernel.sh {parity|trainer-smoke|all|help}

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

Important notes:
  - Default target arch is H100: TORCH_CUDA_ARCH_LIST=9.0
  - For local command-path validation only, override TORCH_CUDA_ARCH_LIST=8.9
    or 12.0 and do not interpret the result as H100 timing evidence
  - Trainer smoke assumes challenge data/tokenizer paths are already available
    in the environment or via repo defaults

Environment overrides:
  PYTHON_BIN                Explicit Python executable to use.
  CONDA_ENV_NAME            Optional conda env name if conda is available,
                            defaults to pg.
  TORCH_CUDA_ARCH_LIST      Defaults to 9.0.
  GDN_MEGAKERNEL_REC_CHUNK_T Defaults to 8.
  MK_TIMING_REPEATS         Defaults to 3.
  MK_CASES_DIR              Defaults to hgdn_megakernel/cases.
  MK_OUTPUT_DIR             Defaults to artifacts/hgdn_megakernel/<run>.
  MK_ARCHIVE_OUTPUT         Defaults to <MK_OUTPUT_DIR>.7z.
  RUN_PREFIX                Base prefix for trainer smoke run ids.
  RUN_ID                    Explicit trainer smoke run id.
  TORCH_LOGS                Defaults to graph_breaks,recompiles.
  TORCHINDUCTOR_CACHE_DIR   Defaults to /tmp/<run-id>_inductor.
  ITERATIONS                Defaults to 5 for trainer-smoke.
  COMPILE_WARMUP_STEPS      Defaults to 2 for trainer-smoke.
  USE_WANDB                 Defaults to 0.
  WANDB_MODE                Defaults to offline.
  WANDB_WATCH               Defaults to none.
  DRY_RUN                   If 1, print commands without executing them.

Examples:
  scripts/run_h100_single_gpu_hgdn_megakernel.sh all
  TORCH_CUDA_ARCH_LIST=8.9 scripts/run_h100_single_gpu_hgdn_megakernel.sh parity
  PYTHON_BIN=python3 scripts/run_h100_single_gpu_hgdn_megakernel.sh all
  GDN_MEGAKERNEL_REC_CHUNK_T=4 scripts/run_h100_single_gpu_hgdn_megakernel.sh trainer-smoke
  MK_OUTPUT_DIR=/tmp/h100mk_case MK_ARCHIVE_OUTPUT=/tmp/h100mk_case.7z scripts/run_h100_single_gpu_hgdn_megakernel.sh all
  DRY_RUN=1 scripts/run_h100_single_gpu_hgdn_megakernel.sh all
EOF
}

resolve_python_cmd() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "${PYTHON_BIN}"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        local conda_env_name="${CONDA_ENV_NAME:-pg}"
        printf 'conda run -s --name %q python\n' "${conda_env_name}"
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
git_commit=${git_commit}
git_branch=${git_branch}
host_name=${host_name}
timestamp_utc=${timestamp_utc}
EOF
chmod +x "${commands_file}"

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
        "${timestamp_utc}" <<'PY'
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
metadata_file = Path(sys.argv[12]).name
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

logs = sorted(
    path.name for path in bundle_dir.glob("*.log") if path.is_file()
)

manifest = {
    "mode": mode,
    "run_prefix": run_prefix,
    "trainer_run_id": trainer_run_id,
    "archive_output": archive_output,
    "paths": {
        "commands": commands_file,
        "metadata": metadata_file,
        "logs": logs,
    },
    "contract": {
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "gdn_megakernel_rec_chunk_t": rec_chunk_t,
        "mk_timing_repeats": timing_repeats,
        "mk_cases_dir": cases_dir,
        "torchinductor_cache_dir": trainer_cache_dir,
        "torch_logs": torch_logs or None,
        "iterations": iterations,
        "compile_warmup_steps": compile_warmup_steps,
        "train_seq_len": train_seq_len,
        "train_batch_tokens": train_batch_tokens,
        "val_loss_every": val_loss_every,
        "train_log_every": train_log_every,
    },
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

run_parity() {
    echo
    echo "### HGDN megakernel parity gate"
    echo "arch_list=${torch_arch_list} rec_chunk_t=${rec_chunk_t} timing_repeats=${timing_repeats}"
    echo "output_dir=${output_dir}"
    if [[ "${torch_arch_list}" != "9.0" && "${torch_arch_list}" != "9.0a" ]]; then
        echo "note: non-H100 arch override detected; treat this as command-path/correctness validation only"
    fi
    run_cmd "${output_dir}/build.log" env TORCH_CUDA_ARCH_LIST="${torch_arch_list}" "${python_cmd[@]}" \
        setup_hgdn_megakernel.py build_ext --inplace
    run_cmd "${output_dir}/audit.log" "${python_cmd[@]}" scripts/audit_hgdn_megakernel_contract.py
    run_cmd "${output_dir}/parity.log" "${python_cmd[@]}" hgdn_megakernel/test_megakernel.py \
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
    run_cmd "${output_dir}/trainer_smoke.log" env \
        TORCH_CUDA_ARCH_LIST="${torch_arch_list}" \
        TORCHINDUCTOR_CACHE_DIR="${trainer_cache_dir}" \
        TORCH_LOGS="${TORCH_LOGS:-graph_breaks,recompiles}" \
        RUN_ID="${trainer_run_id}" \
        USE_WANDB="${USE_WANDB:-0}" \
        WANDB_MODE="${WANDB_MODE:-offline}" \
        WANDB_WATCH="${WANDB_WATCH:-none}" \
        GDN_USE_CUDA_MEGAKERNEL=1 \
        GDN_MEGAKERNEL_REC_CHUNK_T="${rec_chunk_t}" \
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
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
        TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-1}" \
        PERF_SKIP_FINAL_EVAL="${PERF_SKIP_FINAL_EVAL:-1}" \
        COMPILE="${COMPILE:-1}" \
        COMPILE_STRATEGY="${COMPILE_STRATEGY:-hybrid}" \
        COMPILE_WARMUP_STEPS="${COMPILE_WARMUP_STEPS:-2}" \
        "${python_cmd[@]}" -m torch.distributed.run --standalone --nproc_per_node=1 \
        train_gpt_hybrid.py
}

case "${mode}" in
parity)
    run_parity
    build_bundle
    ;;
trainer-smoke)
    run_trainer_smoke
    build_bundle
    ;;
all)
    run_parity
    run_trainer_smoke
    build_bundle
    ;;
help|-h|--help)
    usage
    ;;
*)
    echo "Unknown mode: ${mode}" >&2
    usage >&2
    exit 1
    ;;
esac
