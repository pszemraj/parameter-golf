#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_hgdn_local_phase1.sh

Purpose:
  Run the local RTX 4070 HGDN phase-1 investigation bundle sequentially:
  1. CUDA preflight
  2. bare GDN hotpath profile
  3. HybridGPT forward/backward hotpath profile
  4. HybridGPT optimizer-only hotpath profile
  5. full trainer eager hybrid profile
  6. bucket-attribution + boundary-audit analysis

Defaults:
  - USE_WANDB=0
  - WANDB_MODE=offline
  - RUN_PREFIX=rtx4070_phase1
  - TRAIN_BATCH_TOKENS=131072
  - TRAIN_SEQ_LEN=2048
  - PROFILE_WAIT=1
  - PROFILE_WARMUP=1
  - PROFILE_ACTIVE=2
  - ITERATIONS=6
  - LOCAL_HOTPATH_BATCH_SIZE=2
  - LOCAL_SEQ_LEN follows TRAIN_SEQ_LEN
  - output root: profiles/${RUN_PREFIX}

Examples:
  bash scripts/run_hgdn_local_phase1.sh
  RUN_PREFIX=rtx4070_phase1b TRAIN_SEQ_LEN=1024 bash scripts/run_hgdn_local_phase1.sh
EOF
}

case "${1:-run}" in
help|-h|--help)
    usage
    exit 0
    ;;
run)
    ;;
*)
    echo "Unknown mode: $1" >&2
    usage >&2
    exit 1
    ;;
esac

export USE_WANDB="${USE_WANDB:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
python_bin="${PYTHON_BIN:-python}"

run_prefix="${RUN_PREFIX:-rtx4070_phase1}"
profile_root="${PROFILE_ROOT:-$HGDN_REPO_ROOT/profiles/$run_prefix}"
local_seq_len="${LOCAL_SEQ_LEN:-${TRAIN_SEQ_LEN:-2048}}"
local_hotpath_batch_size="${LOCAL_HOTPATH_BATCH_SIZE:-2}"
trainer_seq_len="${TRAIN_SEQ_LEN:-2048}"
hybrid_gdn_ratio="${HYBRID_GDN_RATIO:-1}"
hybrid_mlp_mult="${HYBRID_MLP_MULT:-${MLP_MULT:-3.25}}"
trainer_run_dir="$profile_root/trainer/${run_prefix}_profile_eager_hybrid_r${hybrid_gdn_ratio}_mlp${hybrid_mlp_mult}_seq${trainer_seq_len}"
boundary_audit_path="$profile_root/hybrid_fwd_bwd.boundaries.jsonl"
gdn_env=(
    "GDN_USE_PACKED_QKV_CONV=${GDN_USE_PACKED_QKV_CONV:-0}"
    "GDN_USE_PACKED_QKV_PROJ=${GDN_USE_PACKED_QKV_PROJ:-0}"
    "GDN_USE_Q_CONV=${GDN_USE_Q_CONV:-1}"
    "GDN_USE_K_CONV=${GDN_USE_K_CONV:-1}"
    "GDN_USE_V_CONV=${GDN_USE_V_CONV:-1}"
    "GDN_CONV_OUTPUT_CONTIGUOUS=${GDN_CONV_OUTPUT_CONTIGUOUS:-0}"
    "GDN_Q_CONV_OUTPUT_CONTIGUOUS=${GDN_Q_CONV_OUTPUT_CONTIGUOUS:-${GDN_CONV_OUTPUT_CONTIGUOUS:-0}}"
    "GDN_K_CONV_OUTPUT_CONTIGUOUS=${GDN_K_CONV_OUTPUT_CONTIGUOUS:-${GDN_CONV_OUTPUT_CONTIGUOUS:-0}}"
    "GDN_V_CONV_OUTPUT_CONTIGUOUS=${GDN_V_CONV_OUTPUT_CONTIGUOUS:-${GDN_CONV_OUTPUT_CONTIGUOUS:-0}}"
    "GDN_CONTROL_PROJ_FP32=${GDN_CONTROL_PROJ_FP32:-1}"
    "GDN_GATES_FP32=${GDN_GATES_FP32:-1}"
    "GDN_OUTPUT_NORM_FP32=${GDN_OUTPUT_NORM_FP32:-1}"
    "GDN_USE_CUDA_FRONTEND_NCT=${GDN_USE_CUDA_FRONTEND_NCT:-0}"
    "GDN_USE_CUDA_PACKED_CONV=${GDN_USE_CUDA_PACKED_CONV:-0}"
    "GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD=${GDN_USE_CUDA_PACKED_CONV_ATEN_BACKWARD:-0}"
    "GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD=${GDN_USE_CUDA_PACKED_CONV_ATEN_WEIGHT_BACKWARD:-0}"
    "GDN_USE_CUDA_SPLIT_NORM=${GDN_USE_CUDA_SPLIT_NORM:-0}"
    "GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=${GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD:-0}"
    "GDN_PACKED_QKV_SINGLE_CONTIG=${GDN_PACKED_QKV_SINGLE_CONTIG:-0}"
    "GDN_PACKED_QKV_SPLIT_COPY=${GDN_PACKED_QKV_SPLIT_COPY:-0}"
)

print_gdn_env() {
    printf '%s\n' "${gdn_env[@]}"
}

phase1_record_command() {
    hgdn_append_command "$commands_path" "$@"
}

phase1_run_step() {
    local label="$1"
    shift
    echo
    echo ">>> $label"
    hgdn_run_with_env "$@"
}

phase1_run_logged_step() {
    local label="$1"
    local logfile="$2"
    shift 2
    echo
    echo ">>> $label"
    hgdn_run_with_env "$@" | tee "$logfile"
}

mkdir -p "$profile_root"/{preflight,hotpath,trainer,analysis}

commands_path="$profile_root/commands.sh"
hotpath_script="$HGDN_REPO_ROOT/scripts/profile_hgdn_local_hotpath.py"
analyze_script="$HGDN_REPO_ROOT/scripts/analyze_hgdn_phase1.py"
preflight_script="$HGDN_REPO_ROOT/scripts/hgdn_cuda_preflight.py"
trainer_profile_script="$HGDN_REPO_ROOT/scripts/run_h100_single_gpu_hgdn_profile.sh"

preflight_env=("${gdn_env[@]}" "USE_WANDB=$USE_WANDB" "WANDB_MODE=$WANDB_MODE")
preflight_cmd=("$python_bin" "$preflight_script")

gdn_hotpath_env=("${gdn_env[@]}")
gdn_hotpath_cmd=(
    "$python_bin" "$hotpath_script"
    --mode gdn
    --batch-size "$local_hotpath_batch_size"
    --seq-len "$local_seq_len"
    --output-dir "$profile_root/hotpath"
)

hybrid_hotpath_env=(
    "${gdn_env[@]}"
    "GDN_AUDIT_BOUNDARIES=1"
    "GDN_AUDIT_BOUNDARIES_PATH=$boundary_audit_path"
    "GDN_AUDIT_BOUNDARIES_LIMIT=${GDN_AUDIT_BOUNDARIES_LIMIT:-1}"
)
hybrid_hotpath_cmd=(
    "$python_bin" "$hotpath_script"
    --mode hybrid-fwd-bwd
    --batch-size "$local_hotpath_batch_size"
    --seq-len "$local_seq_len"
    --output-dir "$profile_root/hotpath"
)

hybrid_opt_env=("${gdn_env[@]}")
hybrid_opt_cmd=(
    "$python_bin" "$hotpath_script"
    --mode hybrid-opt
    --batch-size "$local_hotpath_batch_size"
    --seq-len "$local_seq_len"
    --output-dir "$profile_root/hotpath"
)

trainer_env=(
    "${gdn_env[@]}"
    "RUN_PREFIX=$run_prefix"
    "PROFILE_DIR=$profile_root/trainer"
    "USE_WANDB=0"
    "WANDB_MODE=offline"
    "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-131072}"
    "TRAIN_SEQ_LEN=$trainer_seq_len"
    "PROFILE_WAIT=${PROFILE_WAIT:-1}"
    "PROFILE_WARMUP=${PROFILE_WARMUP:-1}"
    "PROFILE_ACTIVE=${PROFILE_ACTIVE:-2}"
    "ITERATIONS=${ITERATIONS:-6}"
    "TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-1}"
)
trainer_cmd=(bash "$trainer_profile_script" hybrid-eager)

analysis_cmd=(
    "$python_bin" "$analyze_script"
    --gdn "$profile_root/hotpath/gdn_fwd_bwd.json"
    --hybrid-fwd-bwd "$profile_root/hotpath/hybrid_fwd_bwd.json"
    --hybrid-opt "$profile_root/hotpath/hybrid_optimizer.json"
    --trainer "$trainer_run_dir"
    --boundary-audit "$boundary_audit_path"
    --output-dir "$profile_root/analysis"
)

git rev-parse HEAD > "$profile_root/git_sha.txt"
{
    echo "RUN_PREFIX=$run_prefix"
    echo "PROFILE_ROOT=$profile_root"
    echo "USE_WANDB=$USE_WANDB"
    echo "WANDB_MODE=$WANDB_MODE"
    echo "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-131072}"
    echo "TRAIN_SEQ_LEN=$trainer_seq_len"
    echo "PROFILE_WAIT=${PROFILE_WAIT:-1}"
    echo "PROFILE_WARMUP=${PROFILE_WARMUP:-1}"
    echo "PROFILE_ACTIVE=${PROFILE_ACTIVE:-2}"
    echo "ITERATIONS=${ITERATIONS:-6}"
    echo "LOCAL_HOTPATH_BATCH_SIZE=$local_hotpath_batch_size"
    echo "LOCAL_SEQ_LEN=$local_seq_len"
    echo "HYBRID_GDN_RATIO=$hybrid_gdn_ratio"
    echo "HYBRID_MLP_MULT=$hybrid_mlp_mult"
    print_gdn_env
} > "$profile_root/env_snapshot.txt"
: > "$commands_path"
phase1_record_command "${preflight_env[@]}" "${preflight_cmd[@]}"
phase1_record_command "${gdn_hotpath_env[@]}" "${gdn_hotpath_cmd[@]}"
phase1_record_command "${hybrid_hotpath_env[@]}" "${hybrid_hotpath_cmd[@]}"
phase1_record_command "${hybrid_opt_env[@]}" "${hybrid_opt_cmd[@]}"
phase1_record_command "${trainer_env[@]}" "${trainer_cmd[@]}"
phase1_record_command "${analysis_cmd[@]}"

phase1_run_logged_step \
    "HGDN phase-1 preflight" \
    "$profile_root/preflight/preflight.log" \
    "${preflight_env[@]}" \
    "${preflight_cmd[@]}"
phase1_run_step "HGDN phase-1 hotpath: bare GDN" "${gdn_hotpath_env[@]}" "${gdn_hotpath_cmd[@]}"
phase1_run_step \
    "HGDN phase-1 hotpath: hybrid forward/backward" \
    "${hybrid_hotpath_env[@]}" \
    "${hybrid_hotpath_cmd[@]}"
phase1_run_step \
    "HGDN phase-1 hotpath: hybrid optimizer" \
    "${hybrid_opt_env[@]}" \
    "${hybrid_opt_cmd[@]}"
phase1_run_step \
    "HGDN phase-1 trainer eager profile" \
    "${trainer_env[@]}" \
    "${trainer_cmd[@]}"
phase1_run_step "HGDN phase-1 analysis" "${analysis_cmd[@]}"

echo
echo "phase1_bundle:$profile_root"
