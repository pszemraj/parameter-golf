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
trigram_top_k = 7
seq_len = 2048
batch_size = 32
bptt_chunks = 2
steps = 8192
lr = 0.0035
lr_hold_steps = 7000
stability seeds = 1337, 2027, 3141
```

Completed full-validation reads:

| Run | Full-val BPB | Steady tok/s | Artifact bytes | Read |
|---|---:|---:|---:|---|
| K2 `d128_l5_i512` | `2.0031207874` | `1,137,730` | `8,830,483` | geometry winner |
| K4 `seq2048` | `1.9731361526` | `1,182,049` | `11,371,671` | context/top-K gain |
| K4 `seq2048` BPTT2 | `1.9722313128` | `1,177,934` | `11,405,945` | small BPTT2 gain |
| K6 `seq2048` BPTT2 | `1.9572908661` | `1,169,965` | `13,798,090` | prior finalist |
| K7 `seq2048` BPTT2 | `1.9499987725` | `1,176,814` | `14,916,615` | current leader |
| K6 `seq4096` probe | `2.0023792949` | `1,177,137` | `13,766,373` | reject |

K7 beats K6 `seq2048` BPTT2 by about `0.0073` BPB at seed `1337`, and the
three-seed K7 mean beats the three-seed K6 mean by about `0.0063` BPB. K7 leaves
about `1.08 MB` artifact headroom. K8 is not a default next step because the
K6 to K7 fixed-spec gzip increase was already about the same size as the
remaining headroom.

Completed finalist stability rows:

| Top-K | Seed | Full-val BPB | Artifact bytes | Artifact status |
|---:|---:|---:|---:|---|
| `6` | `1337` | `1.9572908661` | `13,798,090` | `LEFT_ON_TABLE` |
| `6` | `2027` | `1.9551450488` | `13,816,234` | `LEFT_ON_TABLE` |
| `6` | `3141` | `1.9546511147` | `13,816,463` | `LEFT_ON_TABLE` |
| `7` | `1337` | `1.9499987725` | `14,916,615` | `LEFT_ON_TABLE` |
| `7` | `2027` | `1.9506748607` | `14,915,789` | `LEFT_ON_TABLE` |
| `7` | `3141` | `1.9474536988` | `14,916,246` | `LEFT_ON_TABLE` |

K6 mean BPB is `1.9556956765`; stdev is `0.0014033763`. K7 mean BPB is
`1.9493757773`; stdev is `0.0016985474`. All current K7 rows completed step
`8192`, used exact BPB and full validation coverage, reported
`validation_source=explicit_val_shard`, used `assoc_accel`, and stayed below
the 16 MB artifact limit.

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

The local adaptive closeout path completed and promoted K7. Keep using the
adaptive runner when changing planner logic, but the next evidence step is H100
portability for the K7 finalist, not another local architecture search.

Tiny local path after changing adaptive planning logic:

```bash
bash scripts/run_5090_adaptive_closeout.sh \
  --smoke-test \
  --run-id adaptive_closeout_smoke \
  --no-run-benchmark \
  --stop-after seq4096 \
  --count-workers 1
```

## Fresh H100 Bring-Up

Assumptions:

- Linux H100 host with `nvidia-smi` working.
- At least `120 GB` free local disk for the repo, dataset, cache, and run
  artifacts. The SP1024 dataset directory is about `85 GiB`.
- Python `3.11` is preferred. The launchers now use explicit `PYTHON=...` or
  the active `python3`/`python`.
- The first serious host must download the shards and build the K7 shared spec
  once. Later seeds should reuse the shared spec under
  `~/.cache/experiments/param-golf-coreamp/shared_specs/`.

Clone and install:

```bash
git clone --branch exp/coreamp https://github.com/pszemraj/parameter-golf.git parameter-golf
cd parameter-golf

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import torch
import accelerated_scan
import wandb

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("accelerated_scan", accelerated_scan.__version__)
print("wandb", wandb.__version__)
PY
```

Authenticate services if the host does not already have credentials:

```bash
python -m wandb login
# Only needed if the dataset repo requires auth:
# huggingface-cli login
```

Download and verify data:

```bash
python data/cached_challenge_fineweb.py \
  --variant sp1024 \
  --train-shards 195

python tools/check_dataset_shards.py \
  data/datasets/fineweb10B_sp1024 \
  --expected-train-files 195 \
  --expected-val-files 1
```

Run a small stack smoke after data is present:

```bash
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_5090_adaptive_closeout.sh \
  --smoke-test \
  --run-id h100_stack_smoke \
  --no-run-benchmark \
  --start-at k7-preflight \
  --stop-after k7-preflight \
  --count-workers 1
```

Build the full K7 shared spec and artifact preflight:

```bash
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_5090_trigram_aligned_geometry_screen.sh \
  --run-version h100_seq2048_bptt2_k7_preflight \
  --seeds 1337 \
  --geometry-label blocks0_d128_l5_i512 \
  --geometry-core-dim 128 \
  --geometry-core-layers 5 \
  --geometry-core-inner-dim 512 \
  --trigram-top-k 7 \
  --geometry-seq-len 2048 \
  --geometry-batch-size 32 \
  --geometry-bptt-chunks 2 \
  --target-effective-step-tokens 131072 \
  --num-steps 8192 \
  --lr-hold-steps 7000 \
  --val-every 1024 \
  --log-every 512 \
  --log-state-every 4096 \
  --save-every 4096 \
  --geometry-train-label preflight_seq2048_bptt2_k7_h100 \
  --preflight-only \
  --preflight-trainable-payload-bytes 1267367 \
  --full-val-final \
  --count-workers 4
```

Run the single-H100 K7 portability check:

```bash
PYTHON="$PWD/.venv/bin/python" \
bash scripts/run_5090_trigram_aligned_geometry_screen.sh \
  --run-version h100_seq2048_bptt2_k7 \
  --seeds 1337 \
  --geometry-label blocks0_d128_l5_i512 \
  --geometry-core-dim 128 \
  --geometry-core-layers 5 \
  --geometry-core-inner-dim 512 \
  --trigram-top-k 7 \
  --geometry-seq-len 2048 \
  --geometry-batch-size 32 \
  --geometry-bptt-chunks 2 \
  --target-effective-step-tokens 131072 \
  --num-steps 8192 \
  --lr-hold-steps 7000 \
  --val-every 1024 \
  --log-every 512 \
  --log-state-every 4096 \
  --save-every 4096 \
  --geometry-train-label 1b_seq2048_bptt2_k7_h100 \
  --preflight-trainable-payload-bytes 1267367 \
  --full-val-final \
  --count-workers 4
```

The H100 portability gate is:

- `completed=true`, `final_step=8192`, exact full-validation BPB.
- `validation_source=explicit_val_shard`.
- `scan_backend=assoc_accel`.
- artifact estimate below `16,000,000`.
- seed `1337` BPB within roughly `0.003` to `0.005` of the 5090 K7 seed
  `1337` read, `1.9499987725`.

After one H100 seed passes, reuse the built shared spec and run the remaining
stability seeds on separate visible GPUs:

```bash
seeds=(2027 3141)
gpus=(1 2)

for i in "${!seeds[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" \
  PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_5090_trigram_aligned_geometry_screen.sh \
    --run-version h100_seq2048_bptt2_k7 \
    --seeds "${seeds[$i]}" \
    --geometry-label blocks0_d128_l5_i512 \
    --geometry-core-dim 128 \
    --geometry-core-layers 5 \
    --geometry-core-inner-dim 512 \
    --trigram-top-k 7 \
    --geometry-seq-len 2048 \
    --geometry-batch-size 32 \
    --geometry-bptt-chunks 2 \
    --target-effective-step-tokens 131072 \
    --num-steps 8192 \
    --lr-hold-steps 7000 \
    --val-every 1024 \
    --log-every 512 \
    --log-state-every 4096 \
    --save-every 4096 \
    --geometry-train-label 1b_seq2048_bptt2_k7_h100 \
    --preflight-trainable-payload-bytes 1267367 \
    --full-val-final \
    --count-workers 4 &
done

wait
```

Summarize H100 rows from local artifacts, not screenshots:

```bash
find experiments/5090_architecture \
  -name run_results.json \
  -path '*h100_seq2048_bptt2_k7*' \
  -print | sort
```

The K7 preflight payload default remains `1,267,367` bytes. Preflight writes
durable evidence as `artifact_preflight.json` beside the prepared shared spec.

## Stop Rules

Keep K7 as the confirmed local candidate and stop architecture churn unless H100
portability fails. Do not spend H100 time on K8 unless it is first preflighted
and still leaves comfortable artifact headroom.

- `seq4096` does not beat K7 `seq2048` BPTT2 by at least `0.004` BPB.
- Diagnostics show no hard-token-bucket improvement after top-K/context-memory
  changes.

Do not spend remaining time on:

- more pre-trigram gate/router/EMA variants
- replaying old `core_dim=48` controller ladders
- arbitrary geometry sweeps beyond the completed `d128_l5_i512` result
- larger frozen block stacks
- attention-like machinery

## Packaging Readiness

Artifact accounting is total submission bytes:

```text
artifact_estimate_bytes = code bytes + gzip(spec.pt) + int8 trainable payload
```

Before final submission, assemble a non-record record folder with:

- `README.md` summarizing the Core/Amplifier idea and evidence table
- `submission.json` with artifact/eval metadata
- final train log(s) and exact run contract
- runnable `train_gpt.py` plus required local dependencies
- final `spec.pt` and trainable int8 payload under the 16 MB total limit

Smoke the record folder on H100 with one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt.py
```

The goal is reproducibility and honest non-record evidence, not a late DDP or
transformer-style rewrite.

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
