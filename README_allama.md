# ALlama

## What this trainer fixes

This version is the one meant for actual local ablations on a 5090:

- decoder-only ALBERT-style sharing with `ALL`, `SOME`, or untied-ish schedules
- `x0` / `resid_mix` shortcut for deep virtual depth
- one global RoPE cache
- GQA via `enable_gqa=True` when the local PyTorch SDPA path supports it
- epoch-driven training over **all available** downloaded train shards by default
- fail-fast BPB setup for Parameter Golf: no silent `val_bpb=disabled` fallback if `sentencepiece`, `TOKENIZER_PATH`, or `VOCAB_SIZE` is wrong
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
- one full pass over all downloaded train shards, except for a tiny end-of-epoch tail smaller than `TRAIN_SEQ_LEN * WORLD_SIZE` that is dropped to keep fixed-length microbatches
- validation every `VAL_LOSS_EVERY` steps
- no arbitrary `MAX_STEPS` cap

`MAX_STEPS` still exists, but only as a debug cap.
The startup log prints both `dropped_target_tokens_per_epoch` and a `train_tail_drop ... reason=fixed_train_seq_len` line when that tail-drop path is active.

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
- `int8_payload_bytes_init`
- `model_code_bytes`

These are intentionally recorded in W&B config rather than `wandb.log()` history, because they are derived static values, not training metrics.

For trained runs, W&B summary records only the final artifact fields that matter for the rule:

- `artifact_limit_bytes`
- `artifact_headroom_bytes_final`
- `artifact_over_limit_final`
- `artifact_status_final`
- `artifact_warning_final`
- `artifact/code_bytes_final`
- `artifact/int8_payload_zlib_bytes_final`
- `artifact_bytes_final`

Do not use saved file size fields such as `artifact/saved_int8_payload_bytes` to judge challenge compliance. The cap check is `artifact_bytes_final`.

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
- the default export should not waste bytes on float passthrough tensors that do not justify it; keeping `resid_mix_logits` float was too expensive for these near-cap runs, so the default control pattern is now just `depth_gains`

Representative drift on an old near-cap anchor using the older max-abs exporter:

- step 0: `artifact_bytes=15,498,554`
- step 50: `artifact_bytes=22,148,785`
- step 100: `artifact_bytes=22,488,958`
- step 150: `artifact_bytes=22,653,383`

What changed was compression, not file structure:

- `saved_int8_payload_bytes` stayed constant across those checkpoints
- `int8_payload_zlib_bytes` rose from `15,399,405` to `22,554,234`
- checkpoint config overhead was only about `1.7 KB` compressed
- payload metadata overhead was tiny; the large delta was the quantized tensors becoming less compressible

That old behavior was not just a training effect. The exporter itself was wasteful compared with the baseline `train_gpt.py` policy. Auditing the same finished ALlama checkpoints against the baseline-style clipped quantizer showed a huge gap:

- old allama exporter on `wide_ff15_baseline`: `15,961,356`
- baseline-style clipped exporter on the same checkpoint: `7,379,594`
- old allama exporter on `balanced_ff25_baseline`: `16,023,736`
- baseline-style clipped exporter on the same checkpoint: `7,651,896`

The meaningful conclusion is:

- the older allama export policy was leaving massive artifact budget on the table
- the corrected exporter now uses clipped int8 quantization instead of the old max-abs scheme
- any size proxy or family search derived from the old exporter is invalid now
- the old `0.945 * int8_payload_bytes_init` proxy is obsolete
- the useful proxy under the corrected exporter is the architecture-fixed raw payload size, because the raw int8 payload bytes do not depend on trained values and the 750-step corrected re-exports landed in a fairly tight `final_zlib / raw_payload` band of about `0.431-0.451`

### Phase 3: serious ablations

Fresh family-size calibration under the corrected exporter is now done.
The blocked sweep now uses near-cap families that keep `head_dim=64` and spend size budget primarily through `NUM_SHARED_BLOCKS`, because that grows stored capacity far more than logical `NUM_LAYERS` while leaving per-step FLOPs mostly unchanged.

Default calibrated family anchors:

- `wide_ff15`: `MODEL_DIM=1024`, `EMBED_DIM=128`, `NUM_LAYERS=16`, `NUM_HEADS=16`, `NUM_KV_HEADS=2`, `NUM_SHARED_BLOCKS=5`, `MLP_MULT=1.5`, `raw_payload_bytes=35,950,734`, `predicted_artifact_bytes=15,818,966`, headroom `181,034`
- `shortfat_ff20`: `MODEL_DIM=896`, `EMBED_DIM=896`, `NUM_LAYERS=20`, `NUM_HEADS=14`, `NUM_KV_HEADS=2`, `NUM_SHARED_BLOCKS=5`, `MLP_MULT=2.0`, `raw_payload_bytes=34,340,378`, `predicted_artifact_bytes=15,342,304`, headroom `657,696`
- `balanced_ff25`: `MODEL_DIM=768`, `EMBED_DIM=1792`, `NUM_LAYERS=24`, `NUM_HEADS=12`, `NUM_KV_HEADS=4`, `NUM_SHARED_BLOCKS=5`, `MLP_MULT=2.5`, `raw_payload_bytes=34,736,590`, `predicted_artifact_bytes=15,797,101`, headroom `202,899`
- `tall_ff30`: `MODEL_DIM=768`, `EMBED_DIM=960`, `NUM_LAYERS=32`, `NUM_HEADS=12`, `NUM_KV_HEADS=2`, `NUM_SHARED_BLOCKS=5`, `MLP_MULT=3.0`, `raw_payload_bytes=36,063,118`, `predicted_artifact_bytes=15,842,554`, headroom `157,446`

Calibration notes:

- `NUM_SHARED_BLOCKS` is now the primary size lever, not `NUM_LAYERS`
- `shortfat_ff20` now uses the cleaner `896/14q/2kv` shape instead of the earlier `960/15q/5kv` compromise; this leaves more size headroom, but it stays aligned with the AGENTS guidance and preserves `head_dim=64`
- the older 20-step checked-size table and the stale-family guard are obsolete after this recalibration

Cold `--compile` 4-step speed audit on the 5090, using the same sweep batch settings, favored the aligned family set across the board:

- `wide_ff15`: `25704.60 -> 35657.95` avg `tok/s`, `119427.79 -> 131400.50` last-step `tok/s`
- `shortfat_ff20`: `27425.31 -> 29549.36` avg `tok/s`, `105703.23 -> 118886.17` last-step `tok/s`
- `balanced_ff25`: `23216.30 -> 24842.72` avg `tok/s`, `90864.47 -> 105067.74` last-step `tok/s`
- `tall_ff30`: `16913.22 -> 18408.88` avg `tok/s`, `67216.41 -> 75220.66` last-step `tok/s`

The Inductor `Online softmax is disabled on the fly since Inductor decides to split the reduction` warning still appeared on every old and new family in that audit, so the shape cleanup improved speed but did not eliminate the warning.

I also checked trained saved size directly with 20-step one-GPU runs at the actual sweep batch settings. The current anchors are not lazily undersized:

- `wide_ff15`: `14,959,072`
- `shortfat_ff20`: `14,990,094`
- `balanced_ff25`: `15,236,675`
- `tall_ff30`: `15,115,258`

Those older 20-step values are now useful only as historical evidence that changing logical depth is the wrong way to fill budget. The current sweep families above replace them.

The training helper `scripts/run_allama_ablations.sh` is intentionally
training-only and runs a `train_gpt.py` reference baseline before any ALlama
ablations. That reference run now uses the same resolved dataset path,
sequence length, effective batch, eval cadence, compile setting, and artifact
reporting contract as the ALlama sweep, so the local comparison is actually
apples-to-apples:

```bash
bash scripts/run_allama_ablations.sh
```

The `train_gpt.py` reference run is written under
`runs_allama/sbcal_v2_gpt_baseline_reference/` by default and logs the same
useful summary fields:

- `artifact_limit_bytes`
- `artifact_headroom_bytes_final`
- `artifact_over_limit_final`
- `artifact_status_final`
- `artifact_warning_final`
- `artifact/code_bytes_final`
- `artifact/int8_payload_zlib_bytes_final`
- `artifact_bytes_final`

It prints its launch summary before starting, including planned vs scheduled
run counts, the resolved batch settings, local batch size, compile mode, and
wallclock cap.

There are now four intended sweep profiles:

- default `SWEEP_PROFILE=ablate`: 29 planned runs total, `1` `train_gpt.py`
  reference plus `28` ALlama runs across all blocks, using the 1x5090 local
  proxy contract
- `SWEEP_PROFILE=screen`: 5 planned runs total, `1` `train_gpt.py` reference
  plus `4` ALlama baseline-block runs
- `SWEEP_PROFILE=explore`: 29 planned runs total, `1` `train_gpt.py`
  reference plus `28` ALlama runs across all blocks
- `SWEEP_PROFILE=full`: 29 planned runs total, `1` `train_gpt.py` reference
  plus `28` ALlama runs across all blocks

It prefixes run IDs with `sbcal_v2_` by default so the cleaner `14/2` recalibrated sweep does not collide with the earlier rejected `15/5` shortfat directories.

It skips already-completed run directories only when the artifacts exist,
`train.log` shows the expected terminal step or intentional wallclock-capped
stop for the current plan, and `.run_spec` matches the resolved launch
contract for the current run. That means a shorter `ablate`, `screen`, or
`explore` run no longer falsely satisfies a later `full` run with the same
`RUN_ID`. If a directory already exists but does not satisfy the current
completion check, the script stops instead of silently overwriting mixed logs
and artifacts. Use `FORCE_RERUN=1` if you want to overwrite anyway.

That `.run_spec` fingerprint now includes the launch settings that materially
change behavior, including eval mode, validation batching, SDPA backend, MLP
alignment, device/dtype, and the core batch/shape settings. So changing one of
those knobs no longer aliases an old run as "complete".

Completion checks respect intentional wallclock-capped runs. If
`MAX_WALLCLOCK_SECONDS` is part of the run spec, a matching
`train_stop ... reason=max_wallclock_seconds=...` is accepted as completion for
that capped plan, but it still does not alias an uncapped or longer plan with
a different run spec.

It also exports `TORCH_BLAS_PREFER_CUBLASLT=1` for 5090-friendly CUDA BLAS selection.
The trainer now supports `SDPA_BACKEND=auto|flash|efficient|math|cudnn` for explicit SDPA backend experiments.

The sweep exports W&B watch knobs:

- `WANDB_WATCH=all`
- `WANDB_WATCH_LOG_FREQ=100`

When `torch.compile` is active, both trainers use a manual histogram backend
instead of W&B module hooks. That keeps scalar logging intact and preserves
parameter and gradient visibility at `WANDB_WATCH_LOG_FREQ` without injecting
hook state into the compiled graph and triggering Dynamo recompile-limit
failures. Eager runs still use normal `wandb.watch` hooks.

The important batch rule is:

- `local_batch_size = TRAIN_BATCH_TOKENS / (GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN)`

The local model-selection profile is now the explicit 1x5090 proxy contract:

- default `SWEEP_PROFILE=ablate`
- `TRAIN_BATCH_TOKENS=131072`
- `GRAD_ACCUM_STEPS=32`
- `local_batch_size=4`
- `MAX_STEPS=1500`
- `MAX_WALLCLOCK_SECONDS=180`
- `VAL_LOSS_EVERY=100`
- `RUN_COMPILE=1`

That profile is meant to be representative of the eventual 8xH100 submission
contract while still being materially cheaper on one 5090. Both trainers honor
that same resolved batch contract now:

- `train_gpt.py` no longer hardcodes `grad_accum_steps=8 // WORLD_SIZE` when
  `GRAD_ACCUM_STEPS` is explicitly set
- `RUN_COMPILE=0|1` now applies to both the GPT baseline and ALlama runs
- both trainers exclude compile warmup and eval time from the ranked training
  window, so `elapsed_s`, `tokens_per_s`, and `MAX_WALLCLOCK_SECONDS` reflect
  train-only time rather than cold compile time
- ALlama now derives its default `EVAL_BATCH_TOKENS` from the eval microbatch
  (`sampled_eval_batch_size * EVAL_SEQ_LEN * WORLD_SIZE`) instead of the full
  optimizer-step token budget, so default full-eval memory matches the intended
  microbatch contract rather than secretly jumping to a much larger batch
- the trainer startup log now prints `eval_plan ...` and the final log prints
  peak CUDA allocated/reserved memory so hidden phase spikes are easier to spot

The older blocked-sweep profiles are still available unchanged as explicit
alternatives:

- `SWEEP_PROFILE=screen`: `TRAIN_BATCH_TOKENS=262144`,
  `GRAD_ACCUM_STEPS=128`, `local_batch_size=2`, `MAX_STEPS=750`,
  baseline block only
- `SWEEP_PROFILE=explore`: `TRAIN_BATCH_TOKENS=262144`,
  `GRAD_ACCUM_STEPS=128`, `local_batch_size=2`, `MAX_STEPS=750`
- `SWEEP_PROFILE=full`: `TRAIN_BATCH_TOKENS=262144`,
  `GRAD_ACCUM_STEPS=128`, `local_batch_size=2`, `MAX_STEPS=2000`

Current 5090 compile/VRAM checks for the aligned family set say:

- `local_batch_size=16` failed under `--compile` with `No valid triton configs`
- `local_batch_size=8` failed with the same Triton shared-memory limit
- `local_batch_size=4` is the largest validated compile-safe local microbatch on
  this box

Compile remains the right default for the local proxy contract. On the current
1x5090 `ablate` settings, 100-step end-to-end checks beat eager in both cases I
measured:

- GPT reference: compile `45.5s` vs eager `51.1s`, with train-only throughput
  `426k tok/s` vs `289k tok/s`
- `wide_ff15/share_chunk`: compile `148.7s` vs eager `155.8s`, with train-only
  throughput `136.7k tok/s` vs `103.0k tok/s`

Using those same measurements to estimate an uncapped 750-step local run gives:

- GPT reference: compile about `245s` vs eager about `346s`
- `wide_ff15/share_chunk`: compile about `772s` vs eager about `983s`

So compile is already faster even when you count its real warmup cost, and it
only gets better as the run length grows. The wallclock-capped `ablate` profile
is still the correct local screening mode, though, because exact 750-step parity
on one 5090 would be too expensive for early-round ablations.

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
