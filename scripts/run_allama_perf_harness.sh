#!/usr/bin/env bash
set -euo pipefail
# run this from the repo root

export TORCH_BLAS_PREFER_CUBLASLT=1

OUT_DIR="${OUT_DIR:-./runs_allama_validation/perf_harness}"
MODE="${MODE:-benchmark}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
MEASURED_STEPS="${MEASURED_STEPS:-5}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
MICROBATCHES="${MICROBATCHES:-64}"
SDPA_BACKEND="${SDPA_BACKEND:-auto}"
if [[ "${SDPA_BACKEND}" == "cudnn" ]]; then
  unset TORCH_BLAS_PREFER_CUBLASLT
else
  export TORCH_BLAS_PREFER_CUBLASLT=1
fi
COMPILE_FLAG="${COMPILE_FLAG:---compile}"
FULLGRAPH_FLAG="${FULLGRAPH_FLAG:---fullgraph}"
CUDA_GRAPH_FLAG="${CUDA_GRAPH_FLAG:-}"
if [[ $# -eq 0 ]]; then
  MODELS=(allama_anchor gpt_reference)
else
  MODELS=("$@")
fi

for MODEL in "${MODELS[@]}"; do
  echo "perf_harness model=${MODEL} mode=${MODE} out_dir=${OUT_DIR}"
  conda run -s --name train python scripts/profile_training_step.py \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --out-dir "${OUT_DIR}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --measured-steps "${MEASURED_STEPS}" \
    --train-seq-len "${TRAIN_SEQ_LEN}" \
    --train-batch-tokens "${TRAIN_BATCH_TOKENS}" \
    --microbatches "${MICROBATCHES}" \
    --sdpa-backend "${SDPA_BACKEND}" \
    "${COMPILE_FLAG}" \
    "${FULLGRAPH_FLAG}" \
    ${CUDA_GRAPH_FLAG}
done
