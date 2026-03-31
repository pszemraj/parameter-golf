#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export TORCH_BLAS_PREFER_CUBLASLT=1

MODEL="${1:-allama_anchor}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-./runs_allama_validation/nsight_systems}"
RUN_DIR="${OUT_ROOT}/${STAMP}_${MODEL}"
REP_BASE="${RUN_DIR}/${MODEL}"
mkdir -p "${RUN_DIR}"

echo "nsys_profile model=${MODEL} run_dir=${RUN_DIR}"
/usr/local/cuda/bin/nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output="${REP_BASE}" \
  conda run -s --name train python scripts/profile_training_step.py \
    --model "${MODEL}" \
    --mode benchmark \
    --out-dir "${RUN_DIR}/harness" \
    --warmup-steps 3 \
    --measured-steps 1 \
    --train-seq-len 1024 \
    --train-batch-tokens 262144 \
    --microbatches 64 \
    --sdpa-backend auto \
    --compile \
    --fullgraph \
    --cuda-profiler-range

/usr/local/cuda/bin/nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum \
  --format csv \
  "${REP_BASE}.nsys-rep" > "${RUN_DIR}/nsys_stats.csv"
