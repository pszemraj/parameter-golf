#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export TORCH_BLAS_PREFER_CUBLASLT=1

OUT_ROOT="${OUT_ROOT:-./runs_allama_validation/nsight_compute}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${STAMP}"
mkdir -p "${RUN_DIR}"

CASES=("$@")
if [[ ${#CASES[@]} -eq 0 ]]; then
  CASES=(
    allama_mlp_up
    allama_mlp_down
    allama_attn_proj
    allama_qkv
    allama_flash_bwd
  )
fi

for CASE_NAME in "${CASES[@]}"; do
  REP_BASE="${RUN_DIR}/${CASE_NAME}"
  echo "ncu_profile case=${CASE_NAME} run_dir=${RUN_DIR}"
  /usr/local/cuda/bin/ncu \
    --force-overwrite \
    --target-processes all \
    --kernel-name-base demangled \
    --set speedOfLight \
    --export "${REP_BASE}" \
    --capture-range cudaProfilerApi \
    --capture-range-end stop \
    conda run -s --name train python scripts/profile_hot_kernels.py \
      --case "${CASE_NAME}" \
      --warmup-iters 10 \
      --measured-iters 1 \
      --out-dir "${RUN_DIR}/summaries" \
      --cuda-profiler-range

  /usr/local/cuda/bin/ncu \
    --import "${REP_BASE}.ncu-rep" \
    --page details > "${REP_BASE}.txt"
done
