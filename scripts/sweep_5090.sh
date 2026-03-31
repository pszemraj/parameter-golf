#!/usr/bin/env bash
# sweep_5090.sh — Single-target 5090 quality-comparison harness for GPT/hconv.
#
# The trainer config is passed via CLI flags, not trainer env vars. The only
# env var this script relies on for the actual training stack is
# TORCH_BLAS_PREFER_CUBLASLT, because that is a library/runtime knob.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HCONV_TRAINER="${ROOT_DIR}/train_hconv.py"
GPT_TRAINER="${ROOT_DIR}/train_gpt.py"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs_hconv_quality_5090}"

DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${ROOT_DIR}/data/tokenizers/fineweb_1024_bpe.model}"
SDPA_BACKEND="${SDPA_BACKEND:-auto}"
COMPILE_DISABLE="${COMPILE_DISABLE:-0}"
TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT:-1}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pg-hconv-ablations}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-hconv_quality_5090}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_WATCH_LOG="${WANDB_WATCH_LOG:-gradients}"
WANDB_WATCH_LOG_FREQ="${WANDB_WATCH_LOG_FREQ:-25}"
WANDB_TAGS="${WANDB_TAGS:-5090,quality,hconv}"
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-}"
WANDB_RUN_SUFFIX="${WANDB_RUN_SUFFIX:-${RUN_ID_SUFFIX}}"

QUALITY_MAX_STEPS="${QUALITY_MAX_STEPS:-750}"
QUALITY_TRAIN_BATCH_TOKENS="${QUALITY_TRAIN_BATCH_TOKENS:-262144}"
QUALITY_SEQ_LEN="${QUALITY_SEQ_LEN:-1024}"
QUALITY_VAL_BATCH_SIZE="${QUALITY_VAL_BATCH_SIZE:-8192}"
QUALITY_VAL_BATCHES="${QUALITY_VAL_BATCHES:-8}"
QUALITY_VAL_FIRST_STEP="${QUALITY_VAL_FIRST_STEP:-100}"
QUALITY_VAL_LOSS_EVERY="${QUALITY_VAL_LOSS_EVERY:-250}"
QUALITY_TRAIN_LOG_EVERY="${QUALITY_TRAIN_LOG_EVERY:-25}"
QUALITY_WALLCLOCK_SECONDS="${QUALITY_WALLCLOCK_SECONDS:-0}"
QUALITY_WARMUP_STEPS="${QUALITY_WARMUP_STEPS:-20}"

SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-10}"
SMOKE_TRAIN_BATCH_TOKENS="${SMOKE_TRAIN_BATCH_TOKENS:-32768}"
SMOKE_VAL_BATCH_SIZE="${SMOKE_VAL_BATCH_SIZE:-8192}"
SMOKE_VAL_BATCHES="${SMOKE_VAL_BATCHES:-1}"
SMOKE_VAL_FIRST_STEP="${SMOKE_VAL_FIRST_STEP:-0}"
SMOKE_VAL_LOSS_EVERY="${SMOKE_VAL_LOSS_EVERY:-0}"
SMOKE_TRAIN_LOG_EVERY="${SMOKE_TRAIN_LOG_EVERY:-5}"
SMOKE_WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-0}"

TARGET="${1:-}"

mkdir -p "${RUNS_ROOT}"
cd "${ROOT_DIR}"

print_usage() {
    cat <<'EOF'
Usage: bash scripts/sweep_5090.sh TARGET

Targets:
  SMOKE_HCONV   quick trainer smoke test
  GPT_REF       matched GPT baseline under the quality contract
  GPT_12L       size-matched GPT gate against T2/T3 budget
  B1            vanilla hybrid (10 conv, 3 attn)
  C2            pure conv
  T2            tied-deep main bet
  T3            tied + MLP 3x in attention blocks
  I1            T2 + dilated conv
  I2            T2 + squared gating
  I4            T2 + dilated conv + squared gating
  I4H           I4 + hippo init

Requested quality-comparison order:
  GPT_REF, B1, C2, T2, T3, I1, I2, I4, I4H
  Size-match gate: GPT_12L

Optional launcher env overrides:
  DATA_PATH=/abs/path/to/fineweb10B_sp1024
  TOKENIZER_PATH=/abs/path/to/fineweb_1024_bpe.model
  SDPA_BACKEND=auto|flash|cudnn|mem_efficient|math
  COMPILE_DISABLE=0|1
  TORCH_BLAS_PREFER_CUBLASLT=0|1
  WANDB_ENABLE=0|1
  WANDB_PROJECT=pg-hconv-ablations
  WANDB_ENTITY=<optional>
  WANDB_GROUP=hconv_quality_5090
  WANDB_MODE=online|offline|disabled
  WANDB_WATCH_LOG=gradients|all
  WANDB_WATCH_LOG_FREQ=25
  WANDB_TAGS=comma,separated,tags
  RUN_ID_SUFFIX=_3K
  WANDB_RUN_SUFFIX=_3k
  QUALITY_MAX_STEPS=3000
  QUALITY_WARMUP_STEPS=80
EOF
}

require_data() {
    local val_file="${DATA_PATH}/fineweb_val_000000.bin"
    if [[ ! -f "${val_file}" ]]; then
        echo "Missing dataset shard: ${val_file}" >&2
        echo "Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10" >&2
        exit 1
    fi
}

resolve_wandb_project() {
    local target="$1"
    echo "${WANDB_PROJECT}"
}

resolve_wandb_group() {
    local target="$1"
    echo "${WANDB_GROUP}"
}

resolve_wandb_tags() {
    local target="$1"
    local base_tags="${WANDB_TAGS}"
    if [[ "${target}" == "SMOKE_HCONV" && "${base_tags}" == "5090,quality,hconv" ]]; then
        base_tags="5090,smoke,hconv"
    fi
    echo "${base_tags},${target}"
}

resolve_target_id() {
    local target="$1"
    echo "${target}${RUN_ID_SUFFIX}"
}

resolve_wandb_run_name() {
    local target="$1"
    local base_name
    case "${target}" in
        SMOKE_HCONV) base_name="SMOKE_hconv_attn5_uconv6_conv18_mlp2" ;;
        GPT_REF) base_name="GPT_REF_gpt_layers9_dim512_mlp2_tiedemb" ;;
        GPT_12L) base_name="GPT_12L_gpt_layers12_dim512_mlp2_tiedemb" ;;
        B1) base_name="B1_hconv_hybrid_attn3_uconv10_conv10_mlp2" ;;
        C2) base_name="C2_hconv_pureconv_attn0_uconv15_conv15_mlp2" ;;
        T2) base_name="T2_hconv_tieddepth_attn5_uconv6_conv18_mlp2" ;;
        T3) base_name="T3_hconv_tieddepth_attn5_uconv4_conv16_mlp3" ;;
        I1) base_name="I1_hconv_dilated_attn5_uconv6_conv18_mlp2" ;;
        I2) base_name="I2_hconv_sqgate_attn5_uconv6_conv18_mlp2" ;;
        I4) base_name="I4_hconv_dilated_sqgate_attn5_uconv6_conv18_mlp2" ;;
        I4H) base_name="I4H_hconv_dilated_sqgate_hippo_attn5_uconv6_conv18_mlp2" ;;
        *) base_name="${target}" ;;
    esac
    echo "${base_name}${WANDB_RUN_SUFFIX}"
}

run_complete() {
    local run_dir="$1"
    local expected_steps="$2"
    local train_log="${run_dir}/train.log"
    [[ -f "${train_log}" ]] || return 1
    grep -Eq "step:${expected_steps}/${expected_steps} val_loss:" "${train_log}" || return 1
    grep -q "final_int8_zlib_roundtrip" "${train_log}" || return 1
}

guard_run_dir() {
    local run_dir="$1"
    local expected_steps="$2"
    if [[ -d "${run_dir}" ]]; then
        if run_complete "${run_dir}" "${expected_steps}"; then
            echo "Run already completed: ${run_dir}"
            echo "Refusing to rerun the same protocol over an existing completed directory."
            exit 0
        fi
        if find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
            echo "Found incomplete or mismatched existing run directory: ${run_dir}" >&2
            echo "Move or delete it before re-running this target." >&2
            exit 2
        fi
    fi
}

write_launch_summary() {
    local summary_file="$1"
    local target="$2"
    local trainer="$3"
    local run_dir="$4"
    local wandb_project="$5"
    local wandb_group="$6"
    local wandb_run_name="$7"
    local wandb_tags="$8"
    local max_steps="$9"
    local train_batch_tokens="${10}"
    local seq_len="${11}"
    local eval_mode="${12}"
    local val_batch_size="${13}"
    local val_batches="${14}"
    local val_first_step="${15}"
    local val_loss_every="${16}"
    local train_log_every="${17}"
    local warmup_steps="${18}"
    local planned_train_tokens=$(( train_batch_tokens * max_steps ))
    local local_batch_size=$(( train_batch_tokens / (8 * seq_len) ))
    cat > "${summary_file}" <<EOF
target=${target}
trainer=${trainer}
run_dir=${run_dir}
data_path=${DATA_PATH}
tokenizer_path=${TOKENIZER_PATH}
train_seq_len=${seq_len}
train_batch_tokens=${train_batch_tokens}
grad_accum_steps=8
local_batch_size=${local_batch_size}
max_steps=${max_steps}
planned_train_tokens=${planned_train_tokens}
eval_mode=${eval_mode}
val_batch_size=${val_batch_size}
val_batches=${val_batches}
val_first_step=${val_first_step}
val_loss_every=${val_loss_every}
train_log_every=${train_log_every}
max_wallclock_seconds=${QUALITY_WALLCLOCK_SECONDS}
warmup_steps=${warmup_steps}
sdpa_backend=${SDPA_BACKEND}
compile_disable=${COMPILE_DISABLE}
TORCH_BLAS_PREFER_CUBLASLT=${TORCH_BLAS_PREFER_CUBLASLT}
wandb_enable=${WANDB_ENABLE}
wandb_project=${wandb_project}
wandb_entity=${WANDB_ENTITY}
wandb_group=${wandb_group}
wandb_run_name=${wandb_run_name}
wandb_mode=${WANDB_MODE}
wandb_watch_log=${WANDB_WATCH_LOG}
wandb_watch_log_freq=${WANDB_WATCH_LOG_FREQ}
wandb_tags=${wandb_tags}
EOF
}

run_target() {
    local target="$1"
    local trainer="$2"
    local max_steps="$3"
    local train_batch_tokens="$4"
    local seq_len="$5"
    local eval_mode="$6"
    local val_batch_size="$7"
    local val_batches="$8"
    local val_first_step="$9"
    local val_loss_every="${10}"
    local train_log_every="${11}"
    local warmup_steps="${12}"
    shift 12
    local -a trainer_args=("$@")
    local target_id
    target_id="$(resolve_target_id "${target}")"
    local run_dir="${RUNS_ROOT}/${target_id}"
    local stdout_log="${run_dir}/stdout.log"
    local summary_file="${run_dir}/launch_summary.txt"
    local planned_train_tokens=$(( train_batch_tokens * max_steps ))
    local local_batch_size=$(( train_batch_tokens / (8 * seq_len) ))
    local wandb_project
    local wandb_group
    local wandb_run_name
    local wandb_tags
    wandb_project="$(resolve_wandb_project "${target}")"
    wandb_group="$(resolve_wandb_group "${target}")"
    wandb_run_name="$(resolve_wandb_run_name "${target}")"
    wandb_tags="$(resolve_wandb_tags "${target}")"

    guard_run_dir "${run_dir}" "${max_steps}"
    mkdir -p "${run_dir}"
    write_launch_summary \
        "${summary_file}" \
        "${target_id}" \
        "${trainer}" \
        "${run_dir}" \
        "${wandb_project}" \
        "${wandb_group}" \
        "${wandb_run_name}" \
        "${wandb_tags}" \
        "${max_steps}" \
        "${train_batch_tokens}" \
        "${seq_len}" \
        "${eval_mode}" \
        "${val_batch_size}" \
        "${val_batches}" \
        "${val_first_step}" \
        "${val_loss_every}" \
        "${train_log_every}" \
        "${warmup_steps}"

    echo
    echo "================================================================"
    echo "TARGET: ${target}"
    echo "TARGET_ID: ${target_id}"
    echo "TRAINER: ${trainer}"
    echo "RUN_DIR: ${run_dir}"
    echo "planned_train_tokens=${planned_train_tokens}"
    echo "local_batch_size=${local_batch_size}"
    echo "train_batch_tokens=${train_batch_tokens} seq_len=${seq_len} max_steps=${max_steps}"
    echo "eval_mode=${eval_mode} val_batch_size=${val_batch_size} val_batches=${val_batches}"
    echo "val_first_step=${val_first_step} val_loss_every=${val_loss_every} train_log_every=${train_log_every}"
    echo "sdpa_backend=${SDPA_BACKEND} compile_disable=${COMPILE_DISABLE}"
    echo "TORCH_BLAS_PREFER_CUBLASLT=${TORCH_BLAS_PREFER_CUBLASLT}"
    echo "wandb_enable=${WANDB_ENABLE} wandb_project=${wandb_project} wandb_group=${wandb_group}"
    echo "wandb_run_name=${wandb_run_name}"
    echo "wandb_mode=${WANDB_MODE} watch=${WANDB_WATCH_LOG}@${WANDB_WATCH_LOG_FREQ}"
    echo "================================================================"

    env TORCH_BLAS_PREFER_CUBLASLT="${TORCH_BLAS_PREFER_CUBLASLT}" \
        torchrun --standalone --nproc_per_node=1 "${trainer}" \
        --data-path "${DATA_PATH}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --run-id "${target_id}" \
        --output-dir "${run_dir}" \
        --vocab-size 1024 \
        --model-dim 512 \
        --num-heads 8 \
        --num-kv-heads 4 \
        --train-batch-tokens "${train_batch_tokens}" \
        --train-seq-len "${seq_len}" \
        --max-steps "${max_steps}" \
        --val-batch-size "${val_batch_size}" \
        --val-first-step "${val_first_step}" \
        --val-loss-every "${val_loss_every}" \
        --train-log-every "${train_log_every}" \
        --eval-mode "${eval_mode}" \
        --val-batches "${val_batches}" \
        --eval-batch-tokens 0 \
        --max-wallclock-seconds "${QUALITY_WALLCLOCK_SECONDS}" \
        --warmup-steps "${warmup_steps}" \
        --sdpa-backend "${SDPA_BACKEND}" \
        --compile-disable "${COMPILE_DISABLE}" \
        --wandb "${WANDB_ENABLE}" \
        --wandb-project "${wandb_project}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-group "${wandb_group}" \
        --wandb-run-name "${wandb_run_name}" \
        --wandb-tags "${wandb_tags}" \
        --wandb-mode "${WANDB_MODE}" \
        --wandb-watch-log "${WANDB_WATCH_LOG}" \
        --wandb-watch-log-freq "${WANDB_WATCH_LOG_FREQ}" \
        "${trainer_args[@]}" \
        2>&1 | tee "${stdout_log}"

    echo "train log: ${run_dir}/train.log"
    echo "stdout log: ${stdout_log}"
}

run_smoke_hconv() {
    run_target \
        "SMOKE_HCONV" \
        "${HCONV_TRAINER}" \
        "${SMOKE_MAX_STEPS}" \
        "${SMOKE_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${SMOKE_VAL_BATCH_SIZE}" \
        "${SMOKE_VAL_BATCHES}" \
        "${SMOKE_VAL_FIRST_STEP}" \
        "${SMOKE_VAL_LOSS_EVERY}" \
        "${SMOKE_TRAIN_LOG_EVERY}" \
        "${SMOKE_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 6 \
        --n-unique-attn 5 \
        --n-conv-effective 18
}

run_gpt_ref() {
    run_target \
        "GPT_REF" \
        "${GPT_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --num-layers 9 \
        --mlp-mult 2 \
        --tie-embeddings 1
}

run_gpt_12l() {
    run_target \
        "GPT_12L" \
        "${GPT_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --num-layers 12 \
        --mlp-mult 2 \
        --tie-embeddings 1
}

run_b1() {
    run_target \
        "B1" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 10 \
        --n-unique-attn 3 \
        --n-conv-effective 10 \
        --squared-gate 0 \
        --dilated-conv 0 \
        --hippo-init 0
}

run_c2() {
    run_target \
        "C2" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 15 \
        --n-unique-attn 0 \
        --n-conv-effective 15 \
        --squared-gate 0 \
        --dilated-conv 0 \
        --hippo-init 0
}

run_t2() {
    run_target \
        "T2" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 6 \
        --n-unique-attn 5 \
        --n-conv-effective 18 \
        --squared-gate 0 \
        --dilated-conv 0 \
        --hippo-init 0
}

run_t3() {
    run_target \
        "T3" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 3 \
        --n-unique-conv 4 \
        --n-unique-attn 5 \
        --n-conv-effective 16 \
        --squared-gate 0 \
        --dilated-conv 0 \
        --hippo-init 0
}

run_i1() {
    run_target \
        "I1" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 6 \
        --n-unique-attn 5 \
        --n-conv-effective 18 \
        --squared-gate 0 \
        --dilated-conv 1 \
        --hippo-init 0
}

run_i2() {
    run_target \
        "I2" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 6 \
        --n-unique-attn 5 \
        --n-conv-effective 18 \
        --squared-gate 1 \
        --dilated-conv 0 \
        --hippo-init 0
}

run_i4() {
    run_target \
        "I4" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 6 \
        --n-unique-attn 5 \
        --n-conv-effective 18 \
        --squared-gate 1 \
        --dilated-conv 1 \
        --hippo-init 0
}

run_i4h() {
    run_target \
        "I4H" \
        "${HCONV_TRAINER}" \
        "${QUALITY_MAX_STEPS}" \
        "${QUALITY_TRAIN_BATCH_TOKENS}" \
        "${QUALITY_SEQ_LEN}" \
        "sampled" \
        "${QUALITY_VAL_BATCH_SIZE}" \
        "${QUALITY_VAL_BATCHES}" \
        "${QUALITY_VAL_FIRST_STEP}" \
        "${QUALITY_VAL_LOSS_EVERY}" \
        "${QUALITY_TRAIN_LOG_EVERY}" \
        "${QUALITY_WARMUP_STEPS}" \
        --mlp-mult 2 \
        --n-unique-conv 6 \
        --n-unique-attn 5 \
        --n-conv-effective 18 \
        --squared-gate 1 \
        --dilated-conv 1 \
        --hippo-init 1
}

if [[ -z "${TARGET}" ]]; then
    print_usage
    exit 1
fi

require_data

case "${TARGET}" in
    SMOKE_HCONV) run_smoke_hconv ;;
    GPT_REF) run_gpt_ref ;;
    GPT_12L) run_gpt_12l ;;
    B1) run_b1 ;;
    C2) run_c2 ;;
    T2) run_t2 ;;
    T3) run_t3 ;;
    I1) run_i1 ;;
    I2) run_i2 ;;
    I4) run_i4 ;;
    I4H) run_i4h ;;
    *)
        echo "Unknown target: ${TARGET}" >&2
        print_usage
        exit 1
        ;;
esac
