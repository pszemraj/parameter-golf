# ALlama

See [PERFORMANCE.md](PERFORMANCE.md) for the current speed/quality state and
the running improvement log.

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
The startup log now distinguishes:
- `rank_aligned_target_tokens_per_epoch`: targets left after any `WORLD_SIZE` alignment trim
- `usable_target_tokens_per_epoch`: targets left after the fixed-`TRAIN_SEQ_LEN` trim
- `planned_target_tokens_per_epoch`: targets actually scheduled into epoch microbatches

When any tail-drop path is active, `train_tail_drop ...` now reports the separate
rank-alignment and fixed-sequence drop counts instead of folding them into one
ambiguous number.

## Working directory

Work inside a local clone or fork of `openai/parameter-golf` and place `train_allama.py` in the repo root.

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
  X0_GATE_INIT=-6.0
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
    NORM_KIND=rmsnorm \
    python train_allama.py
}

probe_one probe_wide_s4_e384_ff10 1024 384 16 16 2 1.0 4 cycle
probe_one probe_shortfat_s4_ff15 896 896 20 14 2 1.5 4 cycle
probe_one probe_balanced_s4_e1472_ff175 768 1472 24 12 4 1.75 4 cycle
probe_one probe_tall_s4_e832_ff2125 768 832 32 12 2 2.125 4 cycle

# obvious near-cap challengers that now bracket the calibrated anchors
probe_one probe_wide_s4_e448_ff10 1024 448 16 16 2 1.0 4 cycle
probe_one probe_shortfat_s5_ff1125 896 896 20 14 2 1.125 5 cycle
probe_one probe_balanced_s4_e1536_ff175 768 1536 24 12 4 1.75 4 cycle
probe_one probe_tall_s4_e896_ff2125 768 896 32 12 2 2.125 4 cycle
```

Interpretation:
- use these only to bracket rough architecture families; `size_init ... artifact_bytes=...` is not reliable enough to pick final training runs by itself
- if init size is far below budget, go bigger; if it is wildly above budget, trim width or `MLP_MULT`
- once a family is in the right ballpark, switch to short training-based sizing rather than trusting init compression
- the current useful probes are the actual near-cap family anchors plus their obvious one-notch challengers, not the old generic `m1024_l24` / `m1152_l24` shapes
- do not assume the best near-cap family is balanced by default; wide, short-fat, tall, and higher-unique-block variants can all fit under the cap with different `MLP_MULT` choices
- ignore tiny `val_loss` / `val_bpb` differences in this phase because `EVAL_ONLY=1` makes these runs size probes, not learned-model comparisons

### Phase 2: final-size calibration

Per [README.md](../README.md), the thing that counts is compressed artifact
size, defined as:

```text
artifact_bytes = code_bytes + int8_payload_zlib_bytes
```

Two things matter here:

- `size_final` drifts upward quickly during training because the int8 payload becomes much less compressible
- that drift is not caused by checkpoint junk; the payload/checkpoint metadata is tiny, and the growth comes from model entropy
- the default export should not waste bytes on float passthrough tensors that do not justify it; the v4 shared path keeps only the low-dimensional control tensors in fp32 by default: `attn_scale,mlp_scale,q_gain,x0_gate,norm`

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
- the useful proxy under the corrected exporter is the architecture-fixed raw payload size, because the raw int8 payload bytes do not depend on trained values
- on the current trainer/model, the completed 750-step runs so far have landed at a much higher `final_zlib / raw_payload` band than the old README assumed:
- `wide_ff15_baseline`: `22,878,654 / 36,116,777 = 0.6335`
- `shortfat_ff20_baseline`: `22,283,930 / 34,521,333 = 0.6455`
- `shortfat_s5_ff1125_baseline`: `15,892,707 / 24,184,053 = 0.6572`
- `balanced_s4_ff175_baseline`: `16,284,392 / 24,176,327 = 0.6736`
- with `code_bytes=157,752`, the useful conservative raw-payload target on the current trainer is about `23.35M-23.55M`; that is the band used for the new family table below

### Phase 3: serious ablations

Fresh family-size calibration under the current exporter is now done.
The blocked sweep now uses near-cap families that keep `head_dim=64` and spend size budget through CUDA-friendly hidden sizes first, then embed width only where it actually helps, and only then extra shared blocks. A broader search showed that `wide s8`, `balanced s6/s8`, and `tall s8` all overshoot badly even after lowering `MLP_MULT`; the only short-fat higher-block upgrade that stayed interesting was `shortfat s5`, but its best current form still landed slightly over cap on a real 750-step run.

Default calibrated family anchors:

- `wide_s4_e384_ff10`: `MODEL_DIM=1024`, `EMBED_DIM=384`, `NUM_LAYERS=16`, `NUM_HEADS=16`, `NUM_KV_HEADS=2`, `NUM_SHARED_BLOCKS=4`, `MLP_MULT=1.0`, `raw_payload_bytes=23,502,151`, conservative `predicted_artifact_bytes=15,987,999`, headroom `12,001`
- `shortfat_s4_ff15`: `MODEL_DIM=896`, `EMBED_DIM=896`, `NUM_LAYERS=20`, `NUM_HEADS=14`, `NUM_KV_HEADS=2`, `NUM_SHARED_BLOCKS=4`, `MLP_MULT=1.5`, `raw_payload_bytes=23,711,251`, conservative `predicted_artifact_bytes=16,128,842`, headroom `-128,842`; this is the one intentionally aggressive anchor because the short-fat family has also shown a lower observed ratio (`0.6572`) that would project this shape back under cap
- `balanced_s4_e1472_ff175`: `MODEL_DIM=768`, `EMBED_DIM=1472`, `NUM_LAYERS=24`, `NUM_HEADS=12`, `NUM_KV_HEADS=4`, `NUM_SHARED_BLOCKS=4`, `MLP_MULT=1.75`, `raw_payload_bytes=23,356,487`, conservative `predicted_artifact_bytes=15,889,885`, headroom `110,115`
- `tall_s4_e832_ff2125`: `MODEL_DIM=768`, `EMBED_DIM=832`, `NUM_LAYERS=32`, `NUM_HEADS=12`, `NUM_KV_HEADS=2`, `NUM_SHARED_BLOCKS=4`, `MLP_MULT=2.125`, `raw_payload_bytes=23,365,831`, conservative `predicted_artifact_bytes=15,896,178`, headroom `103,822`

Calibration notes:

- all four anchors preserve `head_dim=64` and `MLP_MULTIPLE_OF=128`
- `wide` was underusing size at `EMBED_DIM=128` once the current compression band was measured, so the new anchor spends those bytes through embed width while keeping the `hidden=1024` bucket
- `balanced` and `tall` both work better as "keep the fatter hidden bucket, trim embed width" families than as "drop `MLP_MULT` and keep oversized embeds" families
- `shortfat_s5_ff1125` remains the main over-cap challenger, but the default short-fat anchor now trims raw payload by dropping back to `s4` while keeping the stronger `hidden=1408` bucket
- the obvious one-notch challengers now are `wide_s4_e448_ff10`, `shortfat_s5_ff1125`, `balanced_s4_e1536_ff175`, and `tall_s4_e896_ff2125`
- the older v3 family table is obsolete after this recalibration

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
`runs_allama/sbcal_v4_gpt_baseline_reference/` by default and logs the same
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
run counts, the resolved batch settings, local batch size, compile mode,
planned train-token budget, and wallclock cap.

That wallclock cap still defaults to uncapped local ablations. If you
explicitly set `MAX_WALLCLOCK_SECONDS>0`, both trainers count compile warmup
against that cap instead of treating it as a free prelude. Evaluation time
remains separate.

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

It prefixes run IDs with `sbcal_v4_` by default so the current near-cap family set does not collide with the older rejected calibration passes.

It skips already-completed run directories only when the artifacts exist,
`train.log` shows the expected terminal step or intentional wallclock-capped
stop for the current plan, and `.run_spec` matches the resolved launch
contract for the current run. For the GPT reference, completion is stricter
than `train_stop`: the log must also contain `size_final` and
`final_int8_zlib_roundtrip_exact`, so a run that dies after training but before
final post-quantization validation does not get reused as "complete". That
means a shorter `ablate`, `screen`, or `explore` run no longer falsely
satisfies a later `full` run with the same `RUN_ID`. If a directory already
exists but does not satisfy the current completion check, the script stops
instead of silently overwriting mixed logs and artifacts. Use `FORCE_RERUN=1`
if you want to overwrite anyway.

That `.run_spec` fingerprint now includes the launch settings that materially
change behavior, including eval mode, validation batching, explicit
`EVAL_BATCH_TOKENS`, tokenizer/vocab identity, SDPA backend, MLP alignment,
device/dtype, and the core batch/shape settings. So changing one of those
knobs no longer aliases an old run as "complete".

The wrapper now sanitizes trainer-affecting env vars before launch and then
re-applies the resolved sweep contract explicitly. That closes the experiment
protocol against inherited shell state: an exported trainer override such as
`EVAL_BATCH_TOKENS`, `TOKENIZER_PATH`, or a GPT-only Muon hyperparameter can no
longer silently alter a run without also appearing in the sweep summary and
`.run_spec`.

Completion checks respect intentional wallclock-capped runs. If
`MAX_WALLCLOCK_SECONDS` is part of the run spec, a matching
`train_stop ... reason=max_wallclock_seconds=...` is accepted as completion for
that capped plan, but it still does not alias an uncapped or longer plan with
a different run spec. This only affects explicitly capped runs; the default
ablation contract remains uncapped.

The active torch 2.11+ scripts no longer force the old 5090 `cublaslt` workaround.
The trainer supports `SDPA_BACKEND=auto|flash|efficient|math|cudnn` for explicit SDPA backend experiments.

The sweep exports W&B watch knobs:

- `WANDB_WATCH=all`
- `WANDB_WATCH_LOG_FREQ=100`

Those watch settings are part of the printed launch summary and `.run_spec`, so
changing them does not silently reuse a run directory from a different
observability contract.

When `torch.compile` is active, both trainers fall back to manual histogram
logging for `WANDB_WATCH` instead of using W&B module hooks. Eager runs still
use normal `wandb.watch(...)` hooks, while compiled runs keep observability out
of the compiled graph on purpose.

The important batch rule is:

- `local_batch_size = TRAIN_BATCH_TOKENS / (GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN)`

The local model-selection profile is now the explicit 1x5090 fixed-data proxy
contract:

- default `SWEEP_PROFILE=ablate`
- `TRAIN_BATCH_TOKENS=262144`
- `GRAD_ACCUM_STEPS=64`
- `local_batch_size=4`
- `MAX_STEPS=750`
- `planned_train_tokens=196608000`
- `MAX_WALLCLOCK_SECONDS=0`
- `VAL_LOSS_EVERY=100`
- `RUN_COMPILE=1`

That profile is meant to be representative of the eventual 8xH100 submission
contract while still being materially cheaper on one 5090, and it is intended
for model-quality comparisons first, not speed screening. Baseline and ALlama
should both consume the same token budget under this profile. Both trainers
honor that same resolved batch contract now:

- `train_gpt.py` no longer hardcodes `grad_accum_steps=8 // WORLD_SIZE` when
  `GRAD_ACCUM_STEPS` is explicitly set
- `train_gpt.py` and `train_allama.py` now resolve full-eval batch size
  from the same global token-budget contract:
  `EVAL_BATCH_TOKENS` when explicitly set, otherwise
  `sampled_eval_batch_size * EVAL_SEQ_LEN * WORLD_SIZE`
- `RUN_COMPILE=0|1` now applies to both the GPT baseline and ALlama runs
- `train_allama.py` now compiles before optional DDP wrapping instead of
  silently downgrading distributed runs to eager, so the multi-GPU ALlama path
  matches the intended 8xH100-style launch topology
- both trainers exclude compile warmup and eval time from their logged
  `elapsed_s` and `tokens_per_s`, so those stay train-only speed signals
- if `MAX_WALLCLOCK_SECONDS>0` is explicitly set, both trainers still count
  compile warmup against that cap, so capped runs do not get free compile time
- ALlama now derives its default `EVAL_BATCH_TOKENS` from the eval microbatch
  (`sampled_eval_batch_size * EVAL_SEQ_LEN * WORLD_SIZE`) instead of the full
  optimizer-step token budget, so default full-eval memory matches the intended
  microbatch contract rather than secretly jumping to a much larger batch
- GPT full eval no longer divides `EVAL_BATCH_TOKENS` by
  `GRAD_ACCUM_STEPS`, so switching the aligned sweep contract from sampled eval
  to full eval uses a sensible per-rank validation batch instead of collapsing
  to an invalid token budget
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

Compile remains the right default for the local proxy contract. Earlier
`local_batch_size=4` checks showed compile beating eager for both the GPT
reference and a representative ALlama run, but those wallclock numbers depend
on the exact batch contract. Keep that as a separate speed audit. The default
`ablate` protocol is fixed-data first, and any wallclock-capped screening should
be treated as a different experiment mode.

One backend knob is worth keeping available but not forcing by default:

- cold-cache 1-step compile smoke on a previous balanced near-cap family at `TRAIN_BATCH_TOKENS=524288`, `GRAD_ACCUM_STEPS=128` showed no meaningful gain from forcing flash
- isolated-cache result:
- `SDPA_BACKEND=auto`: `elapsed_s=74.18`, `tokens_per_s=7068`
- `SDPA_BACKEND=flash`: `elapsed_s=74.14`, `tokens_per_s=7071`
- both runs emitted Inductor's `Online softmax is disabled on the fly since Inductor decides to split the reduction` warning
- the later old-vs-aligned family audit showed that the warning still appears even on the cleaned-up `64`/`128`-aligned shapes

So the sweep helper leaves `SDPA_BACKEND=auto` as the default, but the trainer still exposes the override for targeted experiments.

It is structured as:

- baseline block: all four families at the same canonical settings
- share-pattern block: `chunk` and `repeat_2`, applied to the same four families
- norm block: `prenorm+rmsnorm` and `postnorm+rmsnorm`, applied to the same four families
- shortcut block: `x0_gate_init=sigmoid^-1(0.05)` and `no_x0`, applied to the same four families

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
python train_allama.py
```

## Notes

- `MAX_STEPS` is for debug and sanity checks only.
- On CUDA, the trainer uses `bf16` autocast. There is no user-facing half-precision path to care about here.
- `model_summary.txt` is deduped by parameter identity, so shared blocks are counted once in the stored-parameter view.
- The meaningful comparison for this project is not just loss. It is **loss at a given artifact size**.

## sbcal_v4 sweep results

The `sbcal_v4` ablate sweep finished cleanly on the 5090 with the current
near-cap family set. All `29/29` expected runs completed, and all ALlama runs
landed under the `16,000,000` byte artifact cap.

Important historical note:

- these `sbcal_v4` numbers were produced before `layernorm` support was added
  to the shared model
- they also predate the postnorm `x0_gate` symmetry fix, so they should not be
  treated as the final word on postnorm
- `baseline` and `norm_post_rms` in that sweep were the same configuration, so
  that specific block only established that `postnorm+rmsnorm` matched the
  existing baseline control

These numbers are from the default local sweep contract:

- `EVAL_MODE=sampled`
- `TRAIN_BATCH_TOKENS=262144`
- `GRAD_ACCUM_STEPS=64`
- `MAX_STEPS=750`
- `--compile`

So treat them as local ranking signals, not final full-validation results.

### Best runs

Best overall ALlama run:

- `shortfat_s4_ff15 + shortcut_gate005`
- `val_bpb=1.434684`
- `artifact_bytes=15,655,289`
- `tokens_per_s=131,832`

Very close runner-up:

- `shortfat_s4_ff15 + prenorm+rmsnorm`
- `val_bpb=1.436112`
- `artifact_bytes=15,674,897`
- `tokens_per_s=133,391`

GPT reference for context:

- `train_gpt.py` reference: `val_bpb=1.442339`
- `artifact_bytes=11,403,868`
- `tokens_per_s=444,921`

Meaning:

- the best ALlama variant beat the matched GPT reference on sampled `val_bpb`
- GPT is still dramatically faster and much smaller
- the quality frontier is real, but the speed frontier is still not competitive

### Family takeaways

- `shortfat_s4_ff15` is the clear quality winner. Its baseline was already the
  best ALlama baseline at `1.448978`, and both `shortcut_gate005` and
  `prenorm+rmsnorm` improved it further.
- `wide_s4_e384_ff10` is the best speed/quality ALlama tradeoff. Its best run,
  `wide + prenorm`, reached `1.450962` at `183,667 tok/s`.
- `balanced_s4_e1472_ff175` and `tall_s4_e832_ff2125` are dominated after this
  sweep. Their best variants improved meaningfully over their own baselines,
  but they still lost to the best shortfat and wide variants while being slower
  and/or larger.

### Ablation signal

The strongest consistent ablation in this sweep was `prenorm+rmsnorm`.
It improved every family:

- `wide`: `1.479425 -> 1.450962` (`-0.028463`)
- `shortfat`: `1.448978 -> 1.436112` (`-0.012866`)
- `balanced`: `1.499955 -> 1.456177` (`-0.043778`)
- `tall`: `1.495737 -> 1.457064` (`-0.038673`)

Other patterns were weaker:

- `norm_post_rms` was effectively a no-op versus baseline
- `share_chunk` was mostly harmful
- `share_repeat2` was occasionally slightly helpful, but not a headline win
- shortcut changes were family-dependent:
  - `shortcut_gate005` was best for shortfat and clearly helped wide/tall
  - `no_x0` helped wide and balanced, but not shortfat

### Current frontier

The useful local frontier after `sbcal_v4` is:

- `shortfat_s4_ff15 + shortcut_gate005`
- `shortfat_s4_ff15 + prenorm+rmsnorm`
- `wide_s4_e384_ff10 + prenorm+rmsnorm`
- `wide_s4_e384_ff10 + no_x0`

Everything else is currently behind those on quality, speed, or both.

### Recommended next steps

1. Prune the family set to `shortfat` and `wide`.
2. Keep `prenorm+rmsnorm` as a default candidate, not just an ablation.
3. Keep `shortcut_gate005` for shortfat and keep both `shortcut_gate005` and
   `no_x0` alive for wide.
4. Drop `norm_post_rms` from the default sweep space.
5. Drop `share_chunk` from the default sweep space.
6. Keep `share_repeat2` only if you want one cheap residual check for
   speed/quality interaction.
7. Run combination tests on the frontier families, because this sweep only
   changed one factor at a time. The missing obvious combinations are:
   - `shortfat + prenorm + shortcut_gate005`
   - `wide + prenorm + shortcut_gate005`
   - `wide + prenorm + no_x0`
8. Run `EVAL_MODE=full` on the finalists before treating the local ranking as
   final.

If the goal is the next serious ALlama sweep rather than another wide search,
the most defensible reduced matrix is:

- families: `shortfat_s4_ff15`, `wide_s4_e384_ff10`
- norm layouts: `prenorm` and current baseline only if you want a control
- shortcuts: `shortcut_gate005`, `no_x0`, baseline
- share patterns: baseline plus at most `repeat_2`

That keeps the next pass focused on the only parts of the space that actually
showed frontier behavior here.

## Focused norm-kind check

To answer the narrower question "is postnorm being artificially hampered?", use
the dedicated reduced sweep instead of the full `sbcal_v4` matrix:

```bash
bash scripts/run_allama_reduced_norm_sweep.sh
```

That script:

- keeps the family set fixed to the two frontier families:
  `wide_s4_e384_ff10` and `shortfat_s4_ff15`
- holds sharing and shortcut settings aligned
- runs the four direct norm comparisons:
  - `postnorm+rmsnorm`
  - `prenorm+rmsnorm`
  - `postnorm+layernorm`
  - `prenorm+layernorm`

The shared model now supports both `rmsnorm` and `layernorm`, and the same
sigmoided `x0` gate is applied in both prenorm and postnorm paths so the norm
comparison is not confounded by a different shortcut parameterization.

### Reduced norm-kind results

The reduced norm-kind sweep establishes a narrower, more useful conclusion than
`sbcal_v4`:

- drop `layernorm` from the next best-model sweep; it lost to `rmsnorm` in both
  families and both layouts
- keep `postnorm+rmsnorm` alive for `wide_s4_e384_ff10`; it was nearly tied
  with `prenorm+rmsnorm`
- prefer `prenorm+rmsnorm` for `shortfat_s4_ff15`; it still won clearly there

Best runs from that reduced sweep:

- `shortfat + prenorm + rmsnorm`: `val_bpb=1.435873`
- `wide + prenorm + rmsnorm`: `val_bpb=1.447867`
- `wide + postnorm + rmsnorm`: `val_bpb=1.449017`

That means the old broad statement "postnorm is worse" was too strong.
The corrected statement is:

- `layernorm` does not help here
- `postnorm+rmsnorm` is family-dependent and still viable for `wide`
- `prenorm+rmsnorm` remains the safest default for quality

## Frontier v1 sweep

The next serious best-model sweep should be a combination sweep, not another
generic one-factor ablation pass. The dedicated script is:

```bash
bash scripts/run_allama_frontier_sweep.sh
```

This script intentionally does not repurpose `sbcal_v4`. It runs a new,
frontier-only matrix under the same local training contract:

- families: `shortfat_s4_ff15`, `wide_s4_e384_ff10`
- norm kind: `rmsnorm` only
- layouts:
  - `shortfat`: `postnorm` and `prenorm`
  - `wide`: `postnorm` and `prenorm`
- shortcut combinations:
  - `shortfat`: baseline gate and `shortcut_gate005`
  - `wide`: baseline shortcut and `shortcut_no_x0`

The resulting Allama matrix is:

- `shortfat post_rms control`
- `shortfat pre_rms control`
- `shortfat post_rms gate005`
- `shortfat pre_rms gate005`
- `wide post_rms control`
- `wide pre_rms control`
- `wide post_rms no_x0`
- `wide pre_rms no_x0`

plus the optional GPT reference run.

Why this is the right next sweep:

- it keeps only the families that still sit on the local frontier
- it drops `layernorm`, which did not pay off
- it retests `shortcut_gate005` for `shortfat` under the corrected norm/gate
  implementation instead of trusting the historical pre-fix result
- it tests `shortcut_no_x0` for `wide` in both viable norm layouts
- it focuses on missing combinations that can still plausibly beat the current
  best ALlama run instead of spending runs on obviously dominated regions

## Frontier v1 results

Timestamp:

- `2026-03-30T01:51:27-04:00`

The `frontier_v1` sweep completed cleanly on the local 5090. All `9/9` runs
finished, including the GPT reference, and every ALlama run stayed under the
`16,000,000` byte artifact cap.

### Best run

Best overall ALlama run from this sweep:

- `shortfat_s4_ff15 + prenorm + rmsnorm + shortcut_gate005`
- `val_bpb=1.434141`
- `tokens_per_s=133,833`
- `artifact_bytes=15,683,554`

Compared with the matched GPT reference from the same sweep:

- GPT reference: `val_bpb=1.443775`
- GPT reference: `tokens_per_s=442,941`
- GPT reference: `artifact_bytes=11,404,805`

So the best ALlama run beat GPT on sampled `val_bpb` by `0.009634`, while GPT
remained dramatically faster and smaller.

Compared with the previous best ALlama run from `sbcal_v4`:

- old best: `shortfat_s4_ff15 + shortcut_gate005`
- old best `val_bpb=1.434684`
- new best improved by `0.000543`

### Sweep takeaways

- `shortfat + shortcut_gate005` is real, and it works better with
  `prenorm+rmsnorm` than with `postnorm+rmsnorm`
- `shortfat pre_rms control -> pre_rms gate005` improved from `1.438043` to
  `1.434141`
- `shortfat post_rms control -> post_rms gate005` improved from `1.447484` to
  `1.444011`
- `wide + shortcut_no_x0` did not pay off on quality in either norm layout
- `wide` still offered decent speed, but it no longer looked like the best path
  for the strongest model

Sorted ALlama runs from this sweep:

- `shortfat pre_rms gate005`: `1.434141`
- `shortfat pre_rms control`: `1.438043`
- `shortfat post_rms gate005`: `1.444011`
- `shortfat post_rms control`: `1.447484`
- `wide post_rms control`: `1.450800`
- `wide pre_rms control`: `1.450943`
- `wide pre_rms no_x0`: `1.451374`
- `wide post_rms no_x0`: `1.451696`

### Updated recommendation

- keep `shortfat_s4_ff15 + prenorm + rmsnorm + shortcut_gate005` as the new
  primary ALlama candidate
- drop `wide` from best-model quality sweeps unless the goal is explicitly
  speed/quality tradeoff study rather than strongest `val_bpb`
- drop `shortcut_no_x0` from the quality search space
- use the remaining artifact headroom on the shortfat winner instead of
  spending more runs on clearly dominated combinations
