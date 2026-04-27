# 5090 Final Week Plan

Last updated: `2026-04-27`

The active local path is a non-transformer Core/Amplifier LM: frozen statistical
structure, dense trigram memory inside `spec.pt`, and a small recurrent
minGRU controller. Do not spend remaining time on transformer-like attention or
token-token mixing.

Shape rationale: [5090_shape_reassessment.md](5090_shape_reassessment.md)

## Current Candidate

The current local leader is:

```text
blocks0_d128_l5_i512
trigram_top_k = 6
seq_len = 2048
batch_size = 32
bptt_chunks = 2
steps = 8192
lr = 0.0035
lr_hold_steps = 7000
seed = 1337
```

Completed full-validation reads:

| Run | Full-val BPB | Steady tok/s | Artifact bytes | Read |
|---|---:|---:|---:|---|
| K2 `d128_l5_i512` | `2.0031207874` | `1,137,730` | `8,830,483` | geometry winner |
| K4 `seq2048` | `1.9731361526` | `1,182,049` | `11,371,671` | context/top-K gain |
| K4 `seq2048` BPTT2 | `1.9722313128` | `1,177,934` | `11,405,945` | small BPTT2 gain |
| K6 `seq2048` BPTT2 | `1.9572908661` | `1,169,965` | `13,798,090` | current leader |

K6 beats K4 `seq2048` BPTT2 by about `0.0149` BPB and leaves about `2.20 MB`
artifact headroom. K6 is the active local finalist unless K7 fits and improves.

## Protocol

Serious maintained 5090 runs require:

- W&B project `pg-core-amp`
- `SCAN_BACKEND=auto`, resolving to `assoc_accel` on CUDA
- `TORCH_BLAS_PREFER_CUBLASLT=1`
- `COMPILE=0`
- `GRADIENT_CHECKPOINTING=0`
- no `SPEC_MAX_TOKENS`, `TRIGRAM_MAX_TOKENS`, or `DATA_MAX_TOKENS` caps
- explicit validation shard, not a train-fraction split
- exact byte-normalized `val_bpb`
- artifact estimate under `16,000,000` total bytes

Frozen statistics and trigram memory use all local train shards:

```text
train shards = 195
train tokens = 19,473,201,340
validation shards = 1
validation tokens = 62,021,846
```

Validation tokens are never used for frozen statistics or trigram-memory
counts. Check local shard coverage before long cache builds:

```bash
conda run -s --name train python tools/check_dataset_shards.py \
  data/datasets/fineweb10B_sp1024 \
  --expected-train-files 195 \
  --expected-val-files 1
```

## Seed Policy

Seeds are not a search axis.

- Use seed `1337` for screens, confirmations, top-K selection, and finalist
  closeout.
- Do not add additional seeds for LR selection, geometry selection, top-K
  selection, or normal finalist closeout.
- A multi-seed finalist run requires both an explicit user request and
  `--finalist-stability-check`. Treat it as a stability report, not model
  selection.
- Never pick a winner by best seed.

The planner enforces this for finalist closeout: multiple `--finalist-seeds`
without `--finalist-stability-check` returns `blocked`.

## Active Commands

Single-seed K6 finalist closeout:

```bash
bash scripts/run_5090_finalist_closeout.sh \
  --run-id k6_finalist_seed1337_v1 \
  -- \
  --run-version geom1_seq2048_bptt2_k6 \
  --label blocks0_d128_l5_i512 \
  --finalist-run-version geom1_seq2048_bptt2_k6 \
  --finalist-seeds 1337 \
  --finalist-trigram-top-k 6 \
  --finalist-seq-len 2048 \
  --finalist-batch-size 32 \
  --finalist-bptt-chunks 2 \
  --finalist-steps 8192 \
  --finalist-hold-steps 7000 \
  --finalist-train-label 1b_seq2048_bptt2_k6 \
  --count-workers 4
```

This should no-op if seed `1337` already satisfies the exact contract.

K7 artifact preflight:

```bash
bash scripts/run_5090_finalist_closeout.sh \
  --run-id k7_preflight_v1 \
  -- \
  --run-version geom1_seq2048_bptt2_k6 \
  --label blocks0_d128_l5_i512 \
  --finalist-run-version geom1_seq2048_bptt2_k7_preflight \
  --finalist-seeds 1337 \
  --finalist-trigram-top-k 7 \
  --finalist-seq-len 2048 \
  --finalist-batch-size 32 \
  --finalist-bptt-chunks 2 \
  --finalist-steps 8192 \
  --finalist-hold-steps 7000 \
  --finalist-train-label preflight_seq2048_bptt2_k7 \
  --finalist-preflight-only \
  --count-workers 4
```

Train K7 only if preflight stays under the artifact cap with enough headroom for
code growth and trainable payload. Do not run K8 before K7 proves both artifact
viability and a real quality gain.

Optional context probe:

```bash
bash scripts/run_5090_finalist_closeout.sh \
  --run-id k6_seq4096_probe_v1 \
  -- \
  --run-version geom1_seq4096_bptt1_k6 \
  --label blocks0_d128_l5_i512 \
  --finalist-run-version geom1_seq4096_bptt1_k6 \
  --finalist-seeds 1337 \
  --finalist-trigram-top-k 6 \
  --finalist-seq-len 4096 \
  --finalist-batch-size 32 \
  --finalist-bptt-chunks 1 \
  --finalist-steps 8192 \
  --finalist-hold-steps 7000 \
  --finalist-train-label 1b_seq4096_k6 \
  --count-workers 4
```

Run this only after K6/K7 top-K decisions. Promote it only if it beats K6
`seq2048` BPTT2 by at least `0.004` BPB.

## Stop Rules

Keep the best confirmed K6/K7 candidate and stop architecture churn if:

- K7 is over artifact budget or improves K6 by less than `0.004` BPB.
- `seq4096` does not beat K6 `seq2048` BPTT2 by at least `0.004` BPB.
- Diagnostics show no hard-token-bucket improvement after top-K/context-memory
  changes.

Do not spend remaining time on:

- more pre-trigram gate/router/EMA variants
- replaying old `core_dim=48` controller ladders
- arbitrary geometry sweeps beyond the completed `d128_l5_i512` result
- larger frozen block stacks
- attention-like machinery

## Diagnostics

Use diagnostics on completed or partial runs before reviving secondary adapter
ideas:

```bash
conda run -s --name train python tools/analyze_core_amp_run.py \
  /path/to/run_dir \
  --checkpoint /path/to/run_dir/final.pt \
  --steps 64 \
  --batch-size 64 \
  --device cuda
```

Base-bigram delta and residual readout delta remain in the code, but their old
launchers are not part of the active script surface. Recover them from git only
if diagnostics show calibration or frozen-readout capacity is the bottleneck.
