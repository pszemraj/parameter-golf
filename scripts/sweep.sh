#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
NGPU="${NGPU:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-param-golf-hybrid}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export USE_WANDB="${USE_WANDB:-1}" VOCAB_SIZE=1024
chmod +x sweep_agent.sh

case "${1:-single}" in
# ── hybrid_tight: 8h Dk48 Dv48 mlp3.0 r3 (15.8MB, 1.1% HR) ──
single)
    export RUN_ID="hybrid_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
    export NUM_LAYERS=16 MODEL_DIM=384 NUM_HEADS=8 NUM_KV_HEADS=4
    export GDN_N_HEADS=8 GDN_HEAD_K_DIM=48 GDN_EXPAND_V=1.0 GDN_RATIO=3
    export GDN_ALLOW_NEG_EIGVAL=1 GDN_CONV_SIZE=4
    export MLP_MULT=3.0 LEAKY_SLOPE=0.5 TRAIN_SEQ_LEN=2048
    export MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05
    export MUON_MOMENTUM=0.95 WEIGHT_DECAY=0.04 WARMDOWN_ITERS=1200
    export TRAIN_BATCH_TOKENS=524288 ITERATIONS=20000
    export VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200
    echo "=== Hybrid tight: $RUN_ID (16L×384d 12G+4A 8h Dk48 Dv48 mlp3.0) ==="
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt_hybrid.py ;;

# ── baseline_fill: 11L×512d mlp2.75 attn (15.4MB, 3.2% HR) ──
baseline)
    export RUN_ID="baseline_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
    export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
    export GDN_RATIO=0 MLP_MULT=2.75 LEAKY_SLOPE=0.5 TRAIN_SEQ_LEN=2048
    export MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05
    export MUON_MOMENTUM=0.95 WEIGHT_DECAY=0.04 WARMDOWN_ITERS=1200
    export TRAIN_BATCH_TOKENS=524288 ITERATIONS=20000
    export VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200
    echo "=== Baseline fill: $RUN_ID (11L×512d 0G+11A mlp2.75) ==="
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt_hybrid.py ;;

# ── depth_control: 16L×384d mlp3.75 attn (15.5MB, 2.7% HR) ──
depth)
    export RUN_ID="depth_ctrl_${NGPU}gpu_$(date +%Y%m%d_%H%M%S)"
    export NUM_LAYERS=16 MODEL_DIM=384 NUM_HEADS=8 NUM_KV_HEADS=4
    export GDN_RATIO=0 MLP_MULT=3.75 LEAKY_SLOPE=0.5 TRAIN_SEQ_LEN=2048
    export MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05
    export MUON_MOMENTUM=0.95 WEIGHT_DECAY=0.04 WARMDOWN_ITERS=1200
    export TRAIN_BATCH_TOKENS=524288 ITERATIONS=20000
    export VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200
    echo "=== Depth control: $RUN_ID (16L×384d 0G+16A mlp3.75) ==="
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt_hybrid.py ;;

# ── A/B/C: reviewer-recommended 3-way comparison ──
abc)
    echo "=== 3-way A/B/C: hybrid vs baseline vs depth_control ==="
    "$0" single   # hybrid
    "$0" baseline  # width baseline
    "$0" depth     # depth control
    echo "=== Compare all three in wandb ===" ;;

sweep)
    export NGPU
    SWEEP_ID=$(wandb sweep --project "$WANDB_PROJECT" sweep_config.yaml 2>&1 | grep -oP 'wandb agent \K\S+')
    echo "Sweep: $SWEEP_ID"
    wandb agent --count "${SWEEP_COUNT:-10}" "$SWEEP_ID" ;;

quick)
    export RUN_ID="quick_$(date +%s)" USE_WANDB=0
    export NUM_LAYERS=4 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2
    export GDN_N_HEADS=4 GDN_HEAD_K_DIM=16 GDN_EXPAND_V=2.0
    export GDN_RATIO=3 GDN_ALLOW_NEG_EIGVAL=1 GDN_CONV_SIZE=4
    export MLP_MULT=2 LEAKY_SLOPE=0.5 TRAIN_SEQ_LEN=512
    export ITERATIONS=100 MAX_WALLCLOCK_SECONDS=120
    export VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=25
    export TRAIN_BATCH_TOKENS=32768 WARMUP_STEPS=3 WARMDOWN_ITERS=30
    echo "=== Quick smoke: $RUN_ID ==="
    torchrun --standalone --nproc_per_node="$NGPU" train_gpt_hybrid.py ;;

*) echo "Usage: $0 {single|baseline|depth|abc|sweep|quick}"; exit 1 ;;
esac
