#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hgdn_shell_common.sh"
hgdn_setup_repo_root "${BASH_SOURCE[0]}"

NGPU="${NGPU:-1}"
if (( 8 % NGPU != 0 )); then
    echo "NGPU must evenly divide 8 so grad accumulation stays valid: NGPU=$NGPU" >&2
    exit 1
fi

trainer_path="$HGDN_REPO_ROOT/train_gpt_hybrid.py"
sweep_agent_path="$HGDN_REPO_ROOT/scripts/sweep_agent.sh"
sweep_config_path="$HGDN_REPO_ROOT/scripts/sweep_config.yaml"

export WANDB_PROJECT="${WANDB_PROJECT:-pg-hgdn-ablations}"
export DATA_PATH="${DATA_PATH:-$HGDN_REPO_ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$HGDN_REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export USE_WANDB="${USE_WANDB:-1}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export COMPILE_STRATEGY="${COMPILE_STRATEGY:-model}"
chmod +x "$sweep_agent_path"

setup_perf_env() {
    unset TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCHINDUCTOR_FORCE_DISABLE_CACHES
    if [[ "${PERF_ISOLATE_COMPILE_CACHE:-0}" == "1" ]]; then
        local cache_root="$HGDN_REPO_ROOT/.cache/perf/${RUN_ID}"
        export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
        export TRITON_CACHE_DIR="$cache_root/triton"
        rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
        mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    fi
    if [[ "${PERF_FORCE_DISABLE_COMPILE_CACHES:-0}" == "1" ]]; then
        export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
    fi
}

print_launch_summary() {
    local label="$1"
    local grad_accum_steps=$((8 / NGPU))
    local planned_train_tokens=$((TRAIN_BATCH_TOKENS * ITERATIONS))
    local local_batch_size=$((TRAIN_BATCH_TOKENS / (NGPU * grad_accum_steps * TRAIN_SEQ_LEN)))
    local wallclock="${MAX_WALLCLOCK_SECONDS:-600.0}"

    echo "=== $label: $RUN_ID ==="
    echo "planned_train_tokens=$planned_train_tokens train_batch_tokens=$TRAIN_BATCH_TOKENS train_seq_len=$TRAIN_SEQ_LEN"
    echo "ngpu=$NGPU grad_accum_steps=$grad_accum_steps local_batch_size=$local_batch_size iterations=$ITERATIONS max_wallclock_seconds=$wallclock"
    echo "seed=$SEED pythonhashseed=$PYTHONHASHSEED cudnn_benchmark=$CUDNN_BENCHMARK"
    echo "data_path=$DATA_PATH"
    echo "tokenizer_path=$TOKENIZER_PATH"
    echo "compile_strategy=$COMPILE_STRATEGY perf_timing=${PERF_TIMING:-0} perf_ignore_steps=${PERF_IGNORE_STEPS:-0}"
    echo "torch_logs=${TORCH_LOGS:-<unset>} torch_trace=${TORCH_TRACE:-<unset>}"
    if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" || -n "${TRITON_CACHE_DIR:-}" ]]; then
        echo "inductor_cache_dir=${TORCHINDUCTOR_CACHE_DIR:-<default>}"
        echo "triton_cache_dir=${TRITON_CACHE_DIR:-<default>}"
    fi
    if [[ -n "${TORCHINDUCTOR_FORCE_DISABLE_CACHES:-}" ]]; then
        echo "TORCHINDUCTOR_FORCE_DISABLE_CACHES=${TORCHINDUCTOR_FORCE_DISABLE_CACHES}"
    fi
}

launch_train() {
    export_default SEED 1337
    export_default PYTHONHASHSEED "$SEED"
    export_default CUDNN_BENCHMARK 0
    setup_perf_env
    print_launch_summary "$1"
    torchrun --standalone --nproc_per_node="$NGPU" "$trainer_path"
}

export_default() {
    local name="$1"
    local value="$2"
    export "$name=${!name:-$value}"
}

hgdn_require_cmd torchrun

case "${1:-single}" in
# ── hybrid_tight: 8h Dk48 Dv48 mlp3.0 r3 (15.8MB, 1.1% HR) ──
single)
    export_default RUN_ID "hybrid_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
    export_default NUM_LAYERS 16
    export_default MODEL_DIM 384
    export_default NUM_HEADS 8
    export_default NUM_KV_HEADS 4
    export_default GDN_N_HEADS 8
    export_default GDN_HEAD_K_DIM 48
    export_default GDN_EXPAND_V 1.0
    export_default GDN_RATIO 3
    export_default GDN_ALLOW_NEG_EIGVAL 1
    export_default GDN_CONV_SIZE 4
    export_default MLP_MULT 3.0
    export_default LEAKY_SLOPE 0.5
    export_default TRAIN_SEQ_LEN 2048
    export_default MATRIX_LR 0.04
    export_default SCALAR_LR 0.04
    export_default TIED_EMBED_LR 0.05
    export_default MUON_MOMENTUM 0.95
    export_default WEIGHT_DECAY 0.04
    export_default WARMDOWN_ITERS 1200
    export_default TRAIN_BATCH_TOKENS 524288
    export_default ITERATIONS 20000
    export_default VAL_LOSS_EVERY 1000
    export_default TRAIN_LOG_EVERY 200
    launch_train "Hybrid tight (${NUM_LAYERS}Lx${MODEL_DIM}d GDN_RATIO=${GDN_RATIO} ${GDN_N_HEADS}h Dk${GDN_HEAD_K_DIM} mlp${MLP_MULT})" ;;

# ── baseline_fill: 11L×512d mlp2.75 attn (15.4MB, 3.2% HR) ──
baseline)
    export_default RUN_ID "baseline_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
    export_default NUM_LAYERS 11
    export_default MODEL_DIM 512
    export_default NUM_HEADS 8
    export_default NUM_KV_HEADS 4
    export_default GDN_RATIO 0
    export_default MLP_MULT 2.75
    export_default LEAKY_SLOPE 0.5
    export_default TRAIN_SEQ_LEN 2048
    export_default MATRIX_LR 0.04
    export_default SCALAR_LR 0.04
    export_default TIED_EMBED_LR 0.05
    export_default MUON_MOMENTUM 0.95
    export_default WEIGHT_DECAY 0.04
    export_default WARMDOWN_ITERS 1200
    export_default TRAIN_BATCH_TOKENS 524288
    export_default ITERATIONS 20000
    export_default VAL_LOSS_EVERY 1000
    export_default TRAIN_LOG_EVERY 200
    launch_train "Baseline fill (${NUM_LAYERS}Lx${MODEL_DIM}d 0G+${NUM_LAYERS}A mlp${MLP_MULT})" ;;

# ── depth_control: 16L×384d mlp3.75 attn (15.5MB, 2.7% HR) ──
depth)
    export_default RUN_ID "depth_ctrl_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
    export_default NUM_LAYERS 16
    export_default MODEL_DIM 384
    export_default NUM_HEADS 8
    export_default NUM_KV_HEADS 4
    export_default GDN_RATIO 0
    export_default MLP_MULT 3.75
    export_default LEAKY_SLOPE 0.5
    export_default TRAIN_SEQ_LEN 2048
    export_default MATRIX_LR 0.04
    export_default SCALAR_LR 0.04
    export_default TIED_EMBED_LR 0.05
    export_default MUON_MOMENTUM 0.95
    export_default WEIGHT_DECAY 0.04
    export_default WARMDOWN_ITERS 1200
    export_default TRAIN_BATCH_TOKENS 524288
    export_default ITERATIONS 20000
    export_default VAL_LOSS_EVERY 1000
    export_default TRAIN_LOG_EVERY 200
    launch_train "Attention-only baseline (${NUM_LAYERS}Lx${MODEL_DIM}d 0G+${NUM_LAYERS}A mlp${MLP_MULT})" ;;

# ── A/B/C: reviewer-recommended 3-way comparison ──
abc)
    echo "=== 3-way A/B/C: hybrid vs width baseline vs attention-only baseline ==="
    echo "Runs execute sequentially with the current env overrides."
    "$0" single   # hybrid
    "$0" baseline  # width baseline
    "$0" depth     # attention-only baseline
    echo "=== Compare all three in wandb ===" ;;

sweep)
    export NGPU
    SWEEP_ID=$(wandb sweep --project "$WANDB_PROJECT" "$sweep_config_path" 2>&1 | grep -oP 'wandb agent \K\S+')
    echo "Sweep: $SWEEP_ID"
    wandb agent --count "${SWEEP_COUNT:-10}" "$SWEEP_ID" ;;

quick)
    export_default RUN_ID "quick_$(date +%s)"
    export_default USE_WANDB 0
    export_default NUM_LAYERS 4
    export_default MODEL_DIM 128
    export_default NUM_HEADS 4
    export_default NUM_KV_HEADS 2
    export_default GDN_N_HEADS 4
    export_default GDN_HEAD_K_DIM 16
    export_default GDN_EXPAND_V 2.0
    export_default GDN_RATIO 3
    export_default GDN_ALLOW_NEG_EIGVAL 1
    export_default GDN_CONV_SIZE 4
    export_default MLP_MULT 2
    export_default LEAKY_SLOPE 0.5
    export_default TRAIN_SEQ_LEN 512
    export_default ITERATIONS 100
    export_default MAX_WALLCLOCK_SECONDS 120
    export_default VAL_LOSS_EVERY 50
    export_default TRAIN_LOG_EVERY 25
    export_default TRAIN_BATCH_TOKENS 32768
    export_default WARMUP_STEPS 3
    export_default WARMDOWN_ITERS 30
    launch_train "Quick smoke" ;;

*) echo "Usage: $0 {single|baseline|depth|abc|sweep|quick}"; exit 1 ;;
esac
