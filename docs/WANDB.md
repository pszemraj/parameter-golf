# W&B Logging

Last updated: 2026-03-31 03:40 EDT

This branch now has optional Weights & Biases logging in both [`train_hconv.py`](../train_hconv.py) and [`train_gpt.py`](../train_gpt.py), with the 5090 sweep harness enabling it by default.

## Default Sweep Setup

- Default W&B project for this sweep family: `pg-hconv-ablations`
- Default group from [`scripts/sweep_5090.sh`](../scripts/sweep_5090.sh): `hconv_quality_5090`
- Default watch mode: `gradients`
- Default watch frequency: `25` steps
- Default run name: the sweep target, e.g. `B1`, `C2`, `T2`

The sweep harness is CLI-first: it passes W&B settings as trainer flags, not trainer-specific env vars. The only runtime env var the harness still uses for training behavior is `TORCH_BLAS_PREFER_CUBLASLT`.

## Watch Behavior

Both trainers call `wandb.watch()` on the underlying base model after warmup/state reset, so the logged gradients reflect real training rather than compile priming.

- `--wandb-watch-log gradients` logs gradient histograms per layer and is the default.
- `--wandb-watch-log all` also logs parameter histograms. This is useful when you explicitly want parameter-distribution drift, but it increases logging volume.

Current default choice:

- Keep gradients on always for ablations.
- Leave parameter histograms off by default unless the run is specifically about optimizer/pathology debugging.

## Trainer Flags

Both trainers share the same W&B flags:

- `--wandb 0|1`
- `--wandb-project NAME`
- `--wandb-entity NAME`
- `--wandb-group NAME`
- `--wandb-run-name NAME`
- `--wandb-tags comma,separated,tags`
- `--wandb-mode online|offline|disabled`
- `--wandb-watch-log gradients|all`
- `--wandb-watch-log-freq N`

## Sweep Harness Overrides

[`scripts/sweep_5090.sh`](../scripts/sweep_5090.sh) enables W&B by default, but the launch behavior can still be overridden at the shell level:

```bash
WANDB_ENABLE=1 \
WANDB_PROJECT=pg-hconv-ablations \
WANDB_GROUP=hconv_quality_5090 \
WANDB_MODE=online \
WANDB_WATCH_LOG=gradients \
WANDB_WATCH_LOG_FREQ=25 \
bash scripts/sweep_5090.sh B1
```

Useful variants:

- Offline smoke test:

```bash
WANDB_MODE=offline COMPILE_DISABLE=1 bash scripts/sweep_5090.sh SMOKE_HCONV
```

- Turn on parameter histograms for a targeted debug run:

```bash
WANDB_WATCH_LOG=all bash scripts/sweep_5090.sh T2
```

## Logged Metrics

The logging split is intentionally compact and follows the better schema from the `feat/allama` branch:

- W&B history contains only moving runtime signals:
  - `train/*`
  - `eval/*`
  - `watch/*`
- W&B config holds static run descriptors:
  - resolved hyperparameters
  - resolved batch/eval contract
  - model identity fields such as parameter counts, layer counts, planned train tokens, and code bytes
- W&B summary holds final compliance and export fields:
  - artifact-limit status
  - headroom / over-limit warning
  - final int8+zlib artifact bytes
  - final roundtrip validation metrics
  - peak CUDA memory

That means single-point artifact and final-export scalars are not logged as time series, and the local text log remains the source of truth for verbose training details.

## Artifact Compliance

Final artifact handling now records explicit challenge-compliance summary fields:

- `artifact_limit_bytes`
- `artifact_bytes_final`
- `artifact_headroom_bytes_final`
- `artifact_over_limit_final`
- `artifact_status_final`
- `artifact_warning_final`
- `artifact/code_bytes_final`
- `artifact/int8_payload_zlib_bytes_final`

Status behavior:

- Over limit: `artifact_status_final=OVER_LIMIT`
- Under limit: `artifact_status_final=LEFT_ON_TABLE`
- Exact fit: `artifact_status_final=AT_LIMIT`

The trainers also emit a local warning line:

- `ARTIFACT_OVER_LIMIT ...` when the final export is invalid
- `ARTIFACT_LEFT_ON_TABLE ...` when bytes were left unused under the cap

This is meant to complement, not replace, the source-of-truth text logs in [`runs_hconv_quality_5090/`](../runs_hconv_quality_5090/).
