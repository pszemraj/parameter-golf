# ALlama Reborn trainer

This README is for **`train_allama_reborn.py`** only.

It replaces the earlier mismatched notes. The goal here is simple: one trainer file, one README, same behavior.

## What is in this trainer

`train_allama_reborn.py` is a decoder-only ALBERT-style trainer aimed at the OpenAI Parameter Golf setup.

Main features:

- configurable cross-layer sharing with `NUM_SHARED_BLOCKS`
- sharing layouts:
  - `SHARE_PATTERN=chunk` or `contiguous`
  - `SHARE_PATTERN=cycle` or `round_robin`
  - `SHARE_PATTERN=repeat_2`, `repeat_4`, etc.
- `NORM_LAYOUT=postnorm` or `prenorm`
- `NORM_KIND=layernorm` or `rmsnorm`
- factorized embeddings with `EMBED_DIM != MODEL_DIM`
- optional tied embeddings
- global RoPE cache shared by all blocks
- **ALBERT-style `x0` shortcut / `resid_mix`**, controlled by:
  - `USE_X0_SHORTCUT=1`
  - `RESID_MIX_INIT=0.1`
- GQA using `enable_gqa=True` inside SDPA when the local PyTorch supports it
- full fixed-validation scanning for Parameter Golf via `EVAL_MODE=full`
- faster proxy eval via `EVAL_MODE=sampled`
- optional `q_deltas` / `v_deltas` hooks and `loss_reduction=none` path for TTT-style work
- checkpoint save/load
- compact int8 payload export/load for local iteration
- optional W&B logging built directly into the trainer

## Setup

From the Parameter Golf repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# your CUDA torch install here; example if you already have it, skip this
# pip install torch torchvision

pip install sentencepiece wandb huggingface-hub datasets tqdm
```

Notes:

- `sentencepiece` is needed if you want correct `val_bpb` on the official SP-1024 tokenizer.
- `wandb` is optional. If missing, the trainer still runs.
- On a 5090, use `DTYPE=bf16` unless you have a specific reason not to.

## Get the official Parameter Golf data

From the same repo root:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

This should populate paths like:

- `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `./data/tokenizers/fineweb_1024_bpe.model`

For larger confirmatory runs later, rerun with more train shards, for example:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

## Minimal single-run command

This is a good first real run on one GPU:

```bash
RUN_ID=allama_post_ln_share1 \
OUT_DIR=./runs_allama_reborn \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
DEVICE=cuda \
DTYPE=bf16 \
NUM_LAYERS=12 \
NUM_SHARED_BLOCKS=1 \
SHARE_PATTERN=chunk \
NORM_LAYOUT=postnorm \
NORM_KIND=layernorm \
MODEL_DIM=512 \
EMBED_DIM=128 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2.0 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=65536 \
ITERATIONS=2000 \
VAL_LOSS_EVERY=250 \
EVAL_MODE=sampled \
VAL_BATCH_SIZE=8 \
VAL_BATCHES=8 \
SAVE_PATH=./runs_allama_reborn/allama_post_ln_share1/model.pt \
EXPORT_INT8_PATH=./runs_allama_reborn/allama_post_ln_share1/model_int8.pt \
python train_allama_reborn.py
```

### Important eval note

`EVAL_MODE` behaves like this:

- `auto`: `full` on Parameter Golf data, `sampled` on enwik8
- `full`: exhaustive fixed-val scan
- `sampled`: faster proxy eval using evenly spaced windows

For sweeps, use `EVAL_MODE=sampled`.

For final reruns and honest local numbers, use `EVAL_MODE=full`.

Also note the split in eval knobs:

- `VAL_BATCH_SIZE` is the **sampled-eval batch size in sequences**
- `VAL_BATCHES` is the number of sampled-eval batches
- `EVAL_BATCH_TOKENS` is the **throughput knob for full eval**

## W&B logging

The trainer logs to W&B if either of these is true:

- `WANDB=1`
- `WANDB_PROJECT` is set

Recommended W&B env for your use case:

```bash
export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP=allama-core-5090
export WANDB_TAGS=5090,allama,parameter-golf
wandb login
```

Logged metrics include:

- `train/loss`
- `train/lr`
- `train/tokens_per_s`
- `eval/loss`
- `eval/bpb` when byte accounting is available
- `artifact/code_bytes`
- `artifact/int8_zlib_model_bytes`
- `artifact/total_artifact_bytes`
- `model/param_count`

## Recommended 5090 sweep

I would do this in two passes.

### Pass 1: cheap ranking pass

Use:

- `train-shards=10`
- `ITERATIONS=2000`
- `TRAIN_BATCH_TOKENS=65536`
- `EVAL_MODE=sampled`
- `VAL_LOSS_EVERY=250`
- W&B project `param-golf-ablations`

Recommended first ablation set:

1. `postnorm + layernorm + share1 + chunk`
2. `postnorm + layernorm + share2 + chunk`
3. `postnorm + layernorm + share2 + cycle`
4. `postnorm + layernorm + share4 + chunk`
5. `postnorm + layernorm + share4 + cycle`
6. `postnorm + layernorm + share2 + repeat_2`
7. `prenorm + layernorm + share1 + chunk`
8. `prenorm + layernorm + share2 + cycle`

A bash loop that runs exactly those under W&B:

```bash
export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP=allama-core-5090
export WANDB_TAGS=5090,allama,core

BASE_ENV=(
  OUT_DIR=./runs_allama_reborn
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  DEVICE=cuda
  DTYPE=bf16
  MODEL_DIM=512
  EMBED_DIM=128
  NUM_LAYERS=12
  NUM_HEADS=8
  NUM_KV_HEADS=4
  MLP_MULT=2.0
  TRAIN_SEQ_LEN=1024
  EVAL_SEQ_LEN=1024
  TRAIN_BATCH_TOKENS=65536
  ITERATIONS=2000
  VAL_LOSS_EVERY=250
  TRAIN_LOG_EVERY=25
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
  USE_X0_SHORTCUT=1
  RESID_MIX_INIT=0.1
)

CONFIGS=(
  "RUN_ID=post_ln_share1_chunk NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=1 SHARE_PATTERN=chunk"
  "RUN_ID=post_ln_share2_chunk NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=chunk"
  "RUN_ID=post_ln_share2_cycle NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=cycle"
  "RUN_ID=post_ln_share4_chunk NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=4 SHARE_PATTERN=chunk"
  "RUN_ID=post_ln_share4_cycle NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=4 SHARE_PATTERN=cycle"
  "RUN_ID=post_ln_share2_repeat2 NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=repeat_2"
  "RUN_ID=pre_ln_share1_chunk NORM_LAYOUT=prenorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=1 SHARE_PATTERN=chunk"
  "RUN_ID=pre_ln_share2_cycle NORM_LAYOUT=prenorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=cycle"
)

for cfg in "${CONFIGS[@]}"; do
  echo "Running: $cfg"
  env "${BASE_ENV[@]}" $cfg \
    SAVE_PATH=./runs_allama_reborn/$(echo $cfg | sed -n 's/.*RUN_ID=\([^ ]*\).*/\1/p')/model.pt \
    EXPORT_INT8_PATH=./runs_allama_reborn/$(echo $cfg | sed -n 's/.*RUN_ID=\([^ ]*\).*/\1/p')/model_int8.pt \
    python train_allama_reborn.py
 done
```

If your shell chokes on that substitution inside `env`, use the simpler manual version below instead.

### Cleaner manual loop version

```bash
run_one () {
  local RUN_ID="$1"
  shift
  env \
    WANDB=1 \
    WANDB_PROJECT=param-golf-ablations \
    WANDB_GROUP=allama-core-5090 \
    WANDB_TAGS=5090,allama,core \
    OUT_DIR=./runs_allama_reborn \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    DEVICE=cuda \
    DTYPE=bf16 \
    MODEL_DIM=512 \
    EMBED_DIM=128 \
    NUM_LAYERS=12 \
    NUM_HEADS=8 \
    NUM_KV_HEADS=4 \
    MLP_MULT=2.0 \
    TRAIN_SEQ_LEN=1024 \
    EVAL_SEQ_LEN=1024 \
    TRAIN_BATCH_TOKENS=65536 \
    ITERATIONS=2000 \
    VAL_LOSS_EVERY=250 \
    TRAIN_LOG_EVERY=25 \
    EVAL_MODE=sampled \
    VAL_BATCH_SIZE=8 \
    VAL_BATCHES=8 \
    USE_X0_SHORTCUT=1 \
    RESID_MIX_INIT=0.1 \
    RUN_ID="$RUN_ID" \
    SAVE_PATH=./runs_allama_reborn/$RUN_ID/model.pt \
    EXPORT_INT8_PATH=./runs_allama_reborn/$RUN_ID/model_int8.pt \
    "$@" \
    python train_allama_reborn.py
}

run_one post_ln_share1_chunk  NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=1 SHARE_PATTERN=chunk
run_one post_ln_share2_chunk  NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=chunk
run_one post_ln_share2_cycle  NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=cycle
run_one post_ln_share4_chunk  NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=4 SHARE_PATTERN=chunk
run_one post_ln_share4_cycle  NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=4 SHARE_PATTERN=cycle
run_one post_ln_share2_repeat2 NORM_LAYOUT=postnorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=repeat_2
run_one pre_ln_share1_chunk   NORM_LAYOUT=prenorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=1 SHARE_PATTERN=chunk
run_one pre_ln_share2_cycle   NORM_LAYOUT=prenorm NORM_KIND=layernorm NUM_SHARED_BLOCKS=2 SHARE_PATTERN=cycle
```

### Pass 2: confirm the winners

Take the best 2 or 3 runs from Pass 1 and rerun them with:

- `train-shards=80`
- `ITERATIONS=4000` to `8000`
- `EVAL_MODE=full`
- `EVAL_BATCH_TOKENS=65536` or `131072`, depending on memory

Example:

```bash
RUN_ID=post_ln_share2_cycle_full \
OUT_DIR=./runs_allama_reborn \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
DEVICE=cuda \
DTYPE=bf16 \
NUM_LAYERS=12 \
NUM_SHARED_BLOCKS=2 \
SHARE_PATTERN=cycle \
NORM_LAYOUT=postnorm \
NORM_KIND=layernorm \
MODEL_DIM=512 \
EMBED_DIM=128 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2.0 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=65536 \
EVAL_BATCH_TOKENS=65536 \
ITERATIONS=4000 \
VAL_LOSS_EVERY=1000 \
EVAL_MODE=full \
WANDB=1 \
WANDB_PROJECT=param-golf-ablations \
WANDB_GROUP=allama-confirm-5090 \
SAVE_PATH=./runs_allama_reborn/post_ln_share2_cycle_full/model.pt \
EXPORT_INT8_PATH=./runs_allama_reborn/post_ln_share2_cycle_full/model_int8.pt \
python train_allama_reborn.py
```

## Eval-only mode

The trainer can load a saved checkpoint or its own compact payload and evaluate it.

### Eval a saved checkpoint

```bash
RUN_ID=eval_post_ln_share2_cycle \
OUT_DIR=./runs_allama_reborn \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
DEVICE=cuda \
DTYPE=bf16 \
EVAL_ONLY=1 \
LOAD_PATH=./runs_allama_reborn/post_ln_share2_cycle/model.pt \
NUM_LAYERS=12 \
NUM_SHARED_BLOCKS=2 \
SHARE_PATTERN=cycle \
NORM_LAYOUT=postnorm \
NORM_KIND=layernorm \
MODEL_DIM=512 \
EMBED_DIM=128 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2.0 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=1024 \
EVAL_MODE=full \
EVAL_BATCH_TOKENS=65536 \
python train_allama_reborn.py
```

### Eval a compact payload exported by the same trainer

```bash
RUN_ID=eval_compact_post_ln_share2_cycle \
OUT_DIR=./runs_allama_reborn \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
DEVICE=cuda \
DTYPE=bf16 \
EVAL_ONLY=1 \
LOAD_PATH=./runs_allama_reborn/post_ln_share2_cycle/model_int8.pt \
NUM_LAYERS=12 \
NUM_SHARED_BLOCKS=2 \
SHARE_PATTERN=cycle \
NORM_LAYOUT=postnorm \
NORM_KIND=layernorm \
MODEL_DIM=512 \
EMBED_DIM=128 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2.0 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=1024 \
EVAL_MODE=full \
EVAL_BATCH_TOKENS=65536 \
python train_allama_reborn.py
```

## Most useful environment variables

### Data

- `DATA_BACKEND=auto|parameter_golf|enwik8`
- `DATA_PATH=...`
- `TRAIN_FILES=...`
- `VAL_FILES=...`
- `TOKENIZER_PATH=...`
- `VOCAB_SIZE=...`

### Training

- `TRAIN_SEQ_LEN`
- `TRAIN_BATCH_TOKENS`
- `GRAD_ACCUM_STEPS`
- `ITERATIONS`
- `LEARNING_RATE`
- `MIN_LR`
- `WARMUP_STEPS`
- `WEIGHT_DECAY`
- `GRAD_CLIP_NORM`
- `MAX_WALLCLOCK_SECONDS`

### Eval

- `EVAL_MODE=auto|full|sampled`
- `EVAL_SEQ_LEN`
- `EVAL_BATCH_TOKENS`
- `VAL_BATCH_SIZE`
- `VAL_BATCHES`
- `VAL_LOSS_EVERY`

### Model

- `NUM_LAYERS`
- `NUM_SHARED_BLOCKS`
- `SHARE_PATTERN`
- `MODEL_DIM`
- `EMBED_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `MLP_MULT`
- `NORM_LAYOUT=postnorm|prenorm`
- `NORM_KIND=layernorm|rmsnorm`
- `TIE_EMBEDDINGS=1|0`
- `QK_NORM=1|0`
- `USE_X0_SHORTCUT=1|0`
- `RESID_MIX_INIT=0.1`
- `LAYER_SCALE_INIT`
- `LOGIT_SOFTCAP`

### Save / load

- `SAVE_PATH`
- `EXPORT_INT8_PATH`
- `LOAD_PATH`
- `EVAL_ONLY=1`
- `STRICT_LOAD=1|0`

### W&B

- `WANDB=1`
- `WANDB_PROJECT=param-golf-ablations`
- `WANDB_ENTITY=...`
- `WANDB_GROUP=...`
- `WANDB_RUN_NAME=...`
- `WANDB_TAGS=tag1,tag2`
- `WANDB_NOTES=...`
- `WANDB_MODE=online|offline|disabled`

## Outputs

For each run, the trainer writes:

- `OUT_DIR/RUN_ID/config.json`
- `OUT_DIR/RUN_ID/train.log`

And optionally:

- `SAVE_PATH` for the regular checkpoint
- `EXPORT_INT8_PATH` for the compact payload

## Behavior details worth remembering

### 1. Full eval is the honest local check

On Parameter Golf data, `EVAL_MODE=full` scans the fixed validation tokens exhaustively in `EVAL_SEQ_LEN` chunks.

### 2. Sampled eval is only a proxy

`EVAL_MODE=sampled` is intended for fast sweep ranking. Do not treat those BPB numbers as final.

### 3. `x0` shortcut is on by default in this trainer

This is the ALBERT-style residual mixing path that helps repeated/shared depth avoid washing out the original signal.

### 4. GQA kernel behavior depends on your local torch

If your PyTorch SDPA supports `enable_gqa=True`, the trainer uses it. Otherwise it falls back to explicit KV expansion.

## Local smoke test example

If you just want to see that the trainer works end-to-end before a real run, use a tiny config on CPU:

```bash
RUN_ID=cpu_smoke \
OUT_DIR=./runs_allama_reborn \
DATA_BACKEND=enwik8 \
ENWIK8_PATH=./data/enwik8.gz \
DEVICE=cpu \
DTYPE=fp32 \
NUM_LAYERS=4 \
NUM_SHARED_BLOCKS=2 \
SHARE_PATTERN=repeat_2 \
MODEL_DIM=64 \
EMBED_DIM=32 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2.0 \
TRAIN_SEQ_LEN=64 \
EVAL_SEQ_LEN=64 \
TRAIN_BATCH_TOKENS=256 \
ITERATIONS=2 \
VAL_LOSS_EVERY=1 \
EVAL_MODE=sampled \
VAL_BATCH_SIZE=2 \
VAL_BATCHES=2 \
python train_allama_reborn.py
```

## My default recommendation

For ALlama on this challenge, I would start here:

- `NORM_LAYOUT=postnorm`
- `NORM_KIND=layernorm`
- `NUM_SHARED_BLOCKS=1` or `2`
- `SHARE_PATTERN=chunk` first, then `cycle`
- `USE_X0_SHORTCUT=1`
- `EMBED_DIM=128`, `MODEL_DIM=512`
- `NUM_HEADS=8`, `NUM_KV_HEADS=4`
- `MLP_MULT=2.0`

Then expand only after the sweep tells you where the real win is.
