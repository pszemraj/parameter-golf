# Parameter Golf Core/Amplifier Fork

This branch is focused on a non-transformer Core/Amplifier language model:
frozen statistical structure plus a small recurrent controller. The root path
does not preserve the upstream transformer baseline, MLX baseline, or historical
`records/` archive; those can be recovered from git history if needed.

## Maintained Path

- `train_gpt.py`: root wrapper for the Core/Amplifier trainer
- `train_core_amplifier.py`: maintained trainer
- `inspect_model.py`: frozen spec builder and inspector
- `core_amplifier_lm/`: model, spec, config, scan backend, and experiment code
- `scripts/run_5090_final3day_frontier_batch.sh`: staged final-three-day batch
- `scripts/run_5090_trigram_geometry_matrix.sh`: geometry matrix launcher
- `scripts/run_5090_trigram_aligned_geometry_screen.sh`: one geometry screen
- `scripts/run_5090_adaptive_closeout.sh`: bounded adaptive closeout
- `scripts/run_5090_finalist_closeout.sh`: explicit finalist/preflight runner
- `tools/plan_5090_adaptive_closeout.py`: reads summaries and emits bounded
  next-step commands

Current planning docs:

- [docs/5090_final_week_plan.md](docs/5090_final_week_plan.md)
- [docs/5090_shape_reassessment.md](docs/5090_shape_reassessment.md)

## Local Environment

Use the local `train` conda env for Python commands:

```bash
conda run -s --name train python -m pytest -q
```

The local dataset path used by the 5090 protocol is:

```text
data/datasets/fineweb10B_sp1024
```

Check shard coverage before long cached builds:

```bash
conda run -s --name train python tools/check_dataset_shards.py \
  data/datasets/fineweb10B_sp1024 \
  --expected-train-files 195 \
  --expected-val-files 1
```

## Current Candidate

Current local finalist:

```text
blocks0_d128_l5_i512
trigram_top_k=6
seq_len=2048
bptt_chunks=2
seed=1337
```

The active finalist/preflight commands are in
[docs/5090_final_week_plan.md](docs/5090_final_week_plan.md#active-commands).

Tiny local smoke for the adaptive closeout runner:

```bash
bash scripts/run_5090_adaptive_closeout.sh \
  --smoke-test \
  --run-id smoke_check \
  --no-run-benchmark \
  --stop-after k4 \
  --count-workers 1
```

## Guardrails

Maintained 5090 runs are expected to keep:

- `SCAN_BACKEND=auto`
- `TORCH_BLAS_PREFER_CUBLASLT=1`
- `COMPILE=0`
- `GRADIENT_CHECKPOINTING=0`
- no `SPEC_MAX_TOKENS` / `DATA_MAX_TOKENS` cap
- W&B project `pg-core-amp` for real ablations
- seed `1337` unless the user explicitly asks for a stability report

Do not silently fall back to slower scan backends, approximate BPB, capped spec
builds, or transformer-like token-token mixing on maintained competition paths.

New launcher work should pass experiment protocol through typed CLI arguments
to `tools/run_core_amp_sweep.py`. Keep environment variables for process-local
controls such as `CUDA_VISIBLE_DEVICES`, `TORCH_BLAS_PREFER_CUBLASLT`,
`WANDB_MODE`, and `PYTHON`; do not add new protocol fields as ambient shell
state.

## Checks

```bash
bash -n scripts/5090_common.sh scripts/run_5090_final3day_frontier_batch.sh \
  scripts/run_5090_trigram_geometry_matrix.sh \
  scripts/run_5090_trigram_aligned_geometry_screen.sh \
  scripts/run_5090_adaptive_closeout.sh \
  scripts/run_5090_finalist_closeout.sh

conda run -s --name train python -m py_compile \
  train_gpt.py inspect_model.py train_core_amplifier.py \
  core_amplifier_lm/*.py tools/*.py tests/*.py

conda run -s --name train pytest -q
```
