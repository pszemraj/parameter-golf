# ALlama

## What this trainer fixes

This version is the one meant for actual local ablations on a 5090:

- decoder-only ALBERT-style sharing with `ALL`, `SOME`, or untied-ish schedules
- `x0` / `resid_mix` shortcut for deep virtual depth
- one global RoPE cache
- GQA via `enable_gqa=True` when the local PyTorch SDPA path supports it
- epoch-driven training over **all available** downloaded train shards by default
- sampled or full validation, both wired into the same trainer
- W&B logging of the metrics that actually matter for shared models:
  - `stored_params`
  - `functional_params`
  - `sharing_ratio`
  - `stored_parameter_bytes`
  - `checkpoint_bytes`
  - `checkpoint_zlib_bytes`
  - `int8_payload_bytes`
  - `int8_payload_zlib_bytes`
  - `artifact_bytes`
- hierarchical deduped model summary saved per run

## Defaults

The defaults are no longer the toy baseline-sized smoke settings.

The trainer now defaults to a serious ALlama family:

```text
MODEL_DIM=1024
EMBED_DIM=256
NUM_LAYERS=24
NUM_HEADS=16
NUM_KV_HEADS=4
MLP_MULT=2.5
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=65536
GRAD_ACCUM_STEPS=4
DTYPE=bf16 on CUDA
NUM_EPOCHS=1
```

That means the default behavior is:
- one full pass over all downloaded train shards
- validation every `VAL_LOSS_EVERY` steps
- no arbitrary `ITERATIONS` cap

`MAX_STEPS` still exists, but only as a debug cap.

## Working directory

Work inside a local clone or fork of `openai/parameter-golf` and place `train_allama_reborn.py` in the repo root.

You are using the repo mainly for:
- `data/cached_challenge_fineweb.py`
- the expected tokenizer/data layout
- easy baseline comparison

The trainer itself is otherwise self-contained.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install sentencepiece wandb huggingface-hub datasets tqdm
```

Use your existing CUDA PyTorch install on the 5090.

## Download the official challenge data

Small first pass:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

Larger pass once the sweep is behaving:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

Expected layout:

```text
./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin
./data/tokenizers/fineweb_1024_bpe.model
```

## What gets recorded to W&B

Per run, the config includes the model identity and size fields you actually want for ALlama-style sharing:

- `model_stored_params`
- `model_stored_trainable_params`
- `model_functional_params`
- `model_functional_trainable_params`
- `sharing_ratio`
- `model_stored_parameter_bytes_init`
- `model_checkpoint_bytes_init`
- `model_checkpoint_zlib_bytes_init`
- `model_int8_payload_zlib_bytes_init`
- `model_artifact_bytes_init`

These are intentionally recorded in W&B config rather than `wandb.log()` history, because they are derived static values, not training metrics.

The metrics stream includes only actual run-time signals:

- `train/*`
- `eval/*`

So for every run you can directly compare:
- what is **stored once on disk**
- what is **functionally traversed** after virtual unrolling
- how close the export path is to the 16MB cap

## Output files per run

For a run with `RUN_ID=my_run`, the trainer writes:

```text
<OUT_DIR>/my_run/train.log
<OUT_DIR>/my_run/model_summary.txt
<OUT_DIR>/my_run/model.pt            # if SAVE_PATH is set
<OUT_DIR>/my_run/model_int8.pt       # if EXPORT_INT8_PATH is set
```

The log prints concise init/final lines:

- `model_init ...`
- `size_init ...`
- `payload_init ...`
- `model_final ...`
- `size_final ...`

## Recommended flow on a 5090

Do this in two phases.

### Phase 1: size probes

The point is to stop wasting time on tiny models that leave most of the 16MB artifact budget unused.

Use `EVAL_ONLY=1` to get immediate model-size accounting without training.

```bash
export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP=allama-size-probes
export WANDB_TAGS=5090,allama,size-probe

BASE_ENV=(
  OUT_DIR=./runs_allama
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  DEVICE=cuda
  DTYPE=bf16
  TRAIN_SEQ_LEN=1024
  EVAL_SEQ_LEN=1024
  TRAIN_BATCH_TOKENS=65536
  GRAD_ACCUM_STEPS=4
  NUM_EPOCHS=1
  EVAL_ONLY=1
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
  USE_X0_SHORTCUT=1
  RESID_MIX_INIT=0.1
)

probe_one () {
  local RUN_ID="$1"
  local MODEL_DIM="$2"
  local EMBED_DIM="$3"
  local NUM_LAYERS="$4"
  local NUM_HEADS="$5"
  local NUM_KV_HEADS="$6"
  local MLP_MULT="$7"
  local NUM_SHARED_BLOCKS="$8"
  local SHARE_PATTERN="$9"

  env "${BASE_ENV[@]}" \
    RUN_ID="$RUN_ID" \
    MODEL_DIM="$MODEL_DIM" \
    EMBED_DIM="$EMBED_DIM" \
    NUM_LAYERS="$NUM_LAYERS" \
    NUM_HEADS="$NUM_HEADS" \
    NUM_KV_HEADS="$NUM_KV_HEADS" \
    MLP_MULT="$MLP_MULT" \
    NUM_SHARED_BLOCKS="$NUM_SHARED_BLOCKS" \
    SHARE_PATTERN="$SHARE_PATTERN" \
    NORM_LAYOUT=postnorm \
    NORM_KIND=layernorm \
    python train_allama_reborn.py
}

probe_one probe_m1024_l24_s1 1024 256 24 16 4 2.5 1 chunk
probe_one probe_m1024_l24_s2 1024 256 24 16 4 2.5 2 cycle
probe_one probe_m1152_l24_s1 1152 288 24 18 6 2.5 1 chunk
probe_one probe_m1152_l24_s2 1152 288 24 18 6 2.5 2 cycle
```

Interpretation:
- if `model_artifact_bytes_init` in W&B config or `size_init ... artifact_bytes=...` in the local log is nowhere near 16MB, go bigger
- if it is too high, trim width or `MLP_MULT`
- once you are near budget, **then** do real ablations inside that size regime

### Phase 2: serious ablations

These are the actual first ablations I would run once the size probe says the family is sensible.

```bash
export WANDB=1
export WANDB_PROJECT=param-golf-ablations
export WANDB_GROUP=allama-initial-ablations
export WANDB_TAGS=allama,5090,initial

BASE_ENV=(
  OUT_DIR=./runs_allama
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  DEVICE=cuda
  DTYPE=bf16
  MODEL_DIM=1024
  EMBED_DIM=256
  NUM_LAYERS=24
  NUM_HEADS=16
  NUM_KV_HEADS=4
  MLP_MULT=2.5
  TRAIN_SEQ_LEN=1024
  EVAL_SEQ_LEN=1024
  TRAIN_BATCH_TOKENS=65536
  GRAD_ACCUM_STEPS=4
  NUM_EPOCHS=1
  VAL_LOSS_EVERY=250
  TRAIN_LOG_EVERY=25
  EVAL_MODE=sampled
  VAL_BATCH_SIZE=8
  VAL_BATCHES=8
  USE_X0_SHORTCUT=1
  RESID_MIX_INIT=0.1
)

run_one () {
  local RUN_ID="$1"
  local NORM_LAYOUT="$2"
  local NORM_KIND="$3"
  local NUM_SHARED_BLOCKS="$4"
  local SHARE_PATTERN="$5"

  env "${BASE_ENV[@]}" \
    RUN_ID="$RUN_ID" \
    NORM_LAYOUT="$NORM_LAYOUT" \
    NORM_KIND="$NORM_KIND" \
    NUM_SHARED_BLOCKS="$NUM_SHARED_BLOCKS" \
    SHARE_PATTERN="$SHARE_PATTERN" \
    SAVE_PATH="./runs_allama/${RUN_ID}/model.pt" \
    EXPORT_INT8_PATH="./runs_allama/${RUN_ID}/model_int8.pt" \
    python train_allama_reborn.py
}

run_one post_ln_share1_chunk   postnorm layernorm 1 chunk
run_one post_ln_share2_chunk   postnorm layernorm 2 chunk
run_one post_ln_share2_cycle   postnorm layernorm 2 cycle
run_one post_ln_share4_chunk   postnorm layernorm 4 chunk
run_one post_ln_share4_cycle   postnorm layernorm 4 cycle
run_one post_ln_share2_repeat2 postnorm layernorm 2 repeat_2
run_one pre_ln_share1_chunk    prenorm  layernorm 1 chunk
run_one pre_ln_share2_cycle    prenorm  layernorm 2 cycle
```

That sweep is finally testing something real:
- same serious width/depth family
- same artifact regime
- actual sharing/norm decisions
- no baseline-sized toy model pretending to be an ALBERT experiment

## Full validation vs sampled validation

For local ranking, use:

```bash
EVAL_MODE=sampled
```

For an honest full pass over the validation tokens, use:

```bash
EVAL_MODE=full
```

`VAL_LOSS_EVERY` still controls cadence in either mode.

Sampled mode is faster for sweep triage.
Full mode is what you should run on the finalists.

## Single-run sanity check

```bash
RUN_ID=allama_sanity \
OUT_DIR=./runs_allama \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
DEVICE=cuda \
DTYPE=bf16 \
NUM_EPOCHS=1 \
MAX_STEPS=20 \
VAL_LOSS_EVERY=10 \
TRAIN_LOG_EVERY=5 \
SAVE_PATH=./runs_allama/allama_sanity/model.pt \
EXPORT_INT8_PATH=./runs_allama/allama_sanity/model_int8.pt \
WANDB=1 \
WANDB_PROJECT=param-golf-ablations \
WANDB_GROUP=allama-sanity \
WANDB_TAGS=5090,allama,sanity \
python train_allama_reborn.py
```

## Notes

- `MAX_STEPS` is for debug and sanity checks only.
- On CUDA, the trainer uses `bf16` autocast. There is no user-facing half-precision path to care about here.
- `model_summary.txt` is deduped by parameter identity, so shared blocks are counted once in the stored-parameter view.
- The meaningful comparison for this project is not just loss. It is **loss at a given artifact size**.
