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

The metrics stream includes only actual run-time signals that can move during the run:

- `train/*`
- `eval/*`

That means invariant run descriptors such as resolved eval mode, epoch count, and planned per-step token counts belong in W&B config and the local startup log, not in W&B history.

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

Do this in three phases.

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
- use these only to bracket rough architecture families; `size_init ... artifact_bytes=...` is not reliable enough to pick final training runs by itself
- if init size is far below budget, go bigger; if it is wildly above budget, trim width or `MLP_MULT`
- once a family is in the right ballpark, switch to short training-based sizing rather than trusting init compression
- with the current coarse probe results, `probe_m1024_l24_s2` was a sensible first budget-closing starting point at about 14.5MB init, while the one-shared-block variants were undersized and `probe_m1152_l24_s2` overshot the cap
- do not assume the best near-cap family is balanced by default; wide, short-fat, tall, and higher-unique-block variants can all fit under the cap with different `MLP_MULT` choices
- ignore tiny `val_loss` / `val_bpb` differences in this phase because `EVAL_ONLY=1` makes these runs size probes, not learned-model comparisons

### Phase 2: final-size calibration

Per [README.md](README.md), the thing that counts is compressed artifact size, defined as:

```text
artifact_bytes = code_bytes + int8_payload_zlib_bytes
```

Two things matter here:

- `size_final` drifts upward quickly during training because the int8 payload becomes much less compressible
- that drift is not caused by checkpoint junk; the payload/checkpoint metadata is tiny, and the growth comes from model entropy

Representative drift on an old near-cap anchor using repeated short runs:

- step 0: `artifact_bytes=15,498,554`
- step 50: `artifact_bytes=22,148,785`
- step 100: `artifact_bytes=22,488,958`
- step 150: `artifact_bytes=22,653,383`

What changed was compression, not file structure:

- `saved_int8_payload_bytes` stayed constant across those checkpoints
- `int8_payload_zlib_bytes` rose from `15,399,405` to `22,554,234`
- checkpoint config overhead was only about `1.7 KB` compressed
- payload metadata overhead was tiny; the large delta was the quantized tensors becoming less compressible

Across 50-step runs, the trained payload landed very close to a fixed fraction of the raw payload bytes:

- `wide_ff15` old anchor: `0.9355`
- `shortfat_ff20` old anchor: `0.9373`
- `balanced_ff25` old anchor: `0.9380`
- `tall_ff30` old anchor: `0.9365`
- `wide_ff15` resized anchor: `0.9395`
- `balanced_ff25` resized-but-too-large anchor (`896/224/24`): `0.9426`
- `tall_ff30` resized anchor: `0.9412`

So the practical sizing proxy is:

```text
artifact_proxy_bytes ~= code_bytes + 0.945 * int8_payload_bytes_init
```

Use `0.945` instead of the tighter `~0.937` mean because the smaller near-cap models came in slightly worse.
Then validate any near-cap finalist with a real short run and actual `size_final`.

### Phase 3: serious ablations

These are the actual first ablations I would run once the family has been made final-size-aware instead of init-size-aware.
The current blocked sweep uses four aligned anchors:

- `wide_ff15`: `MODEL_DIM=1024`, `EMBED_DIM=256`, `NUM_LAYERS=16`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `NUM_SHARED_BLOCKS=2`, `MLP_MULT=1.5`, `MLP_MULTIPLE_OF=128`, `artifact_proxy_bytes=15896523`
- `shortfat_ff20`: `MODEL_DIM=896`, `EMBED_DIM=1152`, `NUM_LAYERS=20`, `NUM_HEADS=14`, `NUM_KV_HEADS=2`, `NUM_SHARED_BLOCKS=2`, `MLP_MULT=2.0`, `MLP_MULTIPLE_OF=128`, `artifact_proxy_bytes=15935714`
- `balanced_ff25`: `MODEL_DIM=768`, `EMBED_DIM=1792`, `NUM_LAYERS=24`, `NUM_HEADS=12`, `NUM_KV_HEADS=4`, `NUM_SHARED_BLOCKS=2`, `MLP_MULT=2.5`, `MLP_MULTIPLE_OF=128`, `artifact_proxy_bytes=15969643`
- `tall_ff30`: `MODEL_DIM=768`, `EMBED_DIM=1088`, `NUM_LAYERS=32`, `NUM_HEADS=12`, `NUM_KV_HEADS=4`, `NUM_SHARED_BLOCKS=2`, `MLP_MULT=3.0`, `MLP_MULTIPLE_OF=128`, `artifact_proxy_bytes=15986759`

Why these and not the earlier near-cap anchors:

- the old anchors were all shape-sloppy for the GPU: head dims `66`, `60`, `54`, and `52`
- the new anchors keep the important matmul shapes on `64`/`128` boundaries, with head dims `128`, `64`, `64`, and `64`
- they still span FF ratios `1.5`, `2.0`, `2.5`, and `3.0`, but they now do it with near-cap final-size-aware proxies instead of init-only artifact guesses

Cold `--compile` 4-step speed audit on the 5090, using the same sweep batch settings, favored the aligned family set across the board:

- `wide_ff15`: `25704.60 -> 35657.95` avg `tok/s`, `119427.79 -> 131400.50` last-step `tok/s`
- `shortfat_ff20`: `27425.31 -> 29549.36` avg `tok/s`, `105703.23 -> 118886.17` last-step `tok/s`
- `balanced_ff25`: `23216.30 -> 24842.72` avg `tok/s`, `90864.47 -> 105067.74` last-step `tok/s`
- `tall_ff30`: `16913.22 -> 18408.88` avg `tok/s`, `67216.41 -> 75220.66` last-step `tok/s`

The Inductor `Online softmax is disabled on the fly since Inductor decides to split the reduction` warning still appeared on every old and new family in that audit, so the shape cleanup improved speed but did not eliminate the warning.

The training helper `scripts/run_allama_ablations.sh` is intentionally training-only and uses blocked ablations over those final-size-aware anchors:

```bash
bash scripts/run_allama_ablations.sh
```

It also exports `TORCH_BLAS_PREFER_CUBLASLT=1` for 5090-friendly CUDA BLAS selection.
The trainer now supports `SDPA_BACKEND=auto|flash|efficient|math|cudnn` for explicit SDPA backend experiments.

For short diagnostic runs it also enables W&B parameter/gradient watching by default:

- `WANDB_WATCH=all`
- `WANDB_WATCH_LOG_FREQ=100`

Batch semantics matter here:

- in [train_gpt.py](train_gpt.py), `TRAIN_BATCH_TOKENS=524288` is the effective optimizer-step batch, not the one-GPU microbatch
- on `WORLD_SIZE=1`, that trainer also uses `grad_accum_steps=8`, so the per-microstep token count is `524288 / 8 = 65536`
- the old allama `65536` setting came from that microstep number, not from `train_gpt.py`'s actual effective batch

So the allama training sweep now uses the `train_gpt.py` effective batch target directly:

- `TRAIN_BATCH_TOKENS=524288`

The trainer's own default env values are still smaller; this is a sweep-script override, not a global trainer-default change.

Current 5090 compile/VRAM check says the compile-safe accumulation setting is:

- idle desktop load on this box was about `2.9 GiB`
- `GRAD_ACCUM_STEPS=32` (`local_batch_size=16`) failed under `--compile` with `No valid triton configs`
- `GRAD_ACCUM_STEPS=64` (`local_batch_size=8`) failed with the same Triton shared-memory limit
- `GRAD_ACCUM_STEPS=128` (`local_batch_size=4`) succeeded under `--compile`
- at that passing point, a 1-step smoke hit `max_memory_reserved=4.834 GiB` and `max_memory_allocated=4.776 GiB`

The current aligned family set also passed serialized cold `--compile` 4-step smokes at those same batch settings.

So the sweep defaults are:

- `TRAIN_BATCH_TOKENS=524288`
- `GRAD_ACCUM_STEPS=128`
- `RUN_COMPILE=1`
- `VAL_LOSS_EVERY=500`

That is conservative on VRAM, but it is the first verified compile-safe point for the worst-case family on the current 5090 stack. If you disable compile, you can experiment with lower accumulation separately, but the training helper keeps the compile-safe default.

One backend knob is worth keeping available but not forcing by default:

- cold-cache 1-step compile smoke on `balanced_ff25` at `TRAIN_BATCH_TOKENS=524288`, `GRAD_ACCUM_STEPS=128` showed no meaningful gain from forcing flash
- isolated-cache result:
- `SDPA_BACKEND=auto`: `elapsed_s=74.18`, `tokens_per_s=7068`
- `SDPA_BACKEND=flash`: `elapsed_s=74.14`, `tokens_per_s=7071`
- both runs emitted Inductor's `Online softmax is disabled on the fly since Inductor decides to split the reduction` warning
- the later old-vs-aligned family audit showed that the warning still appears even on the cleaned-up `64`/`128`-aligned shapes

So the sweep helper leaves `SDPA_BACKEND=auto` as the default, but the trainer still exposes the override for targeted experiments.

It is structured as:

- baseline block: all four families at the same canonical settings
- share-pattern block: `chunk` and `repeat_2`, applied to the same four families
- norm block: `prenorm+layernorm` and `postnorm+rmsnorm`, applied to the same four families
- shortcut block: `resid_mix_init=0.05` and `no_x0`, applied to the same four families

That means family comparisons and factor comparisons are not confounded by silently changing the family set underneath them.

You can disable blocks explicitly if you want to stage the work:

```bash
RUN_BASELINE_BLOCK=1 \
RUN_SHARE_BLOCK=0 \
RUN_NORM_BLOCK=0 \
RUN_SHORTCUT_BLOCK=0 \
bash scripts/run_allama_ablations.sh
```

If you want to disable compile and test a different accumulation regime manually:

```bash
RUN_COMPILE=0 bash scripts/run_allama_ablations.sh
```

That sweep is finally testing something real:
- near-cap compressed artifacts under the actual 16,000,000-byte rule, using a conservative final-size proxy instead of init-only compression
- shape families spanning wide, short-fat, balanced, and tall variants
- FF ratio effects across `1.5`, `2.0`, `2.5`, and `3.0`
- one factor at a time within each ablation block

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
