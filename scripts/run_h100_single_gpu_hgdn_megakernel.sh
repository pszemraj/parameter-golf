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
  CONDA_ENV_NAME            Defaults to pg.
  TORCH_CUDA_ARCH_LIST      Defaults to 9.0.
  GDN_MEGAKERNEL_REC_CHUNK_T Defaults to 8.
  MK_TIMING_REPEATS         Defaults to 3.
  MK_CASES_DIR              Defaults to hgdn_megakernel/cases.
  MK_OUTPUT_DIR             Defaults to artifacts/hgdn_megakernel/<run>.
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
  GDN_MEGAKERNEL_REC_CHUNK_T=4 scripts/run_h100_single_gpu_hgdn_megakernel.sh trainer-smoke
  DRY_RUN=1 scripts/run_h100_single_gpu_hgdn_megakernel.sh all
EOF
}

hgdn_require_cmd conda

conda_env_name="${CONDA_ENV_NAME:-pg}"
python_cmd=(conda run -s --name "${conda_env_name}" python)
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
commands_file="${output_dir}/commands.sh"
metadata_file="${output_dir}/metadata.txt"

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
EOF
chmod +x "${commands_file}"

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
    ;;
trainer-smoke)
    run_trainer_smoke
    ;;
all)
    run_parity
    run_trainer_smoke
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
