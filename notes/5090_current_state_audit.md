# 5090 Current State Audit

## Scope
- Audit date: 2026-04-18
- Repo state audited from local branch rooted at commit `0c567fa`
- Goal of this audit: verify the root Core/Amplifier path on disk before broader 5090 screening

## Verified Defaults
- `core_amplifier_lm/config.py` is the current source of truth for root defaults.
- Model defaults:
  - `vocab_size=1024`
  - `core_dim=48`
  - `branch_lags=1,2,3,4,6,8,12,16,24,32,48,64`
  - `num_blocks=9`
  - `core_layers=5`
  - `core_type=mingru`
  - `core_expansion=2.0`
  - `residual_core=True`
  - `residual_core_init=-2.0`
  - `readout_rank=None`
  - `embedding_init=spectral`
  - `spectral_neighbors=64`
  - `lag_identity_base=0.15`
  - `fixed_dtype=bfloat16`
- Training defaults:
  - `seq_len=512`
  - `batch_size=256`
  - `carry_chunks=16`
  - `bptt_chunks=2`
  - `num_steps=7000`
  - `learning_rate=3e-3`
  - `lr_schedule=cosine`
  - `warmup_steps=100`
  - `lr_hold_steps=1500`
  - `min_lr=3e-4`
  - `weight_decay=1e-3`
  - `hard_loss_gamma=0.5`
  - `hard_loss_cap=5.0`
  - `grad_clip=1.0`
  - `amplifier_dtype=auto`

## Wiring Checks
- `train_gpt.py` and `train_core_amplifier.py` agree on the recommended controller path.
- Residual minGRU is the real default path end to end.
- `carry_chunks` and `bptt_chunks` are both wired into the training loop.
  - `carry_chunks` controls stream windows.
  - `bptt_chunks` controls the semi-TBPTT horizon across sequential chunks inside one optimizer step.
- `lr_hold_steps` is wired into the warmup-hold-cosine schedule.
- Delayed compile is real.
  - `--compile`, `--compile-after`, `--compile-mode`, and `--compile-base-path` all resolve and execute.
- Low-rank readout is wired.
  - `inspect_model.py init` accepts `--readout-rank`.
  - spec build stores readout type/rank metadata.
  - runtime validates requested `readout_rank` against `spec.pt`.
- Structural settings are checked at runtime.
  - `branch_lags`, `num_blocks`, and `readout_rank` are compared between config and frozen spec.

## Local Validation
- Static checks passed:
  - `bash -n scripts/sweep_controller.sh scripts/sweep_structure.sh`
  - `python -m py_compile` on root scripts, package, tools, and tests
  - `pytest -q tests/test_core_amplifier.py`
- Exact `val_bpb` works locally.
  - `sentencepiece` is installed in the local `train` conda env.
  - `inspect_model.py init` copies the tokenizer into the model dir.
  - a tiny CUDA smoke run produced non-NaN exact `val_bpb`.
- Local environment check:
  - `torch 2.11.0+cu128`
  - `sentencepiece` present
  - `pyarrow` present
  - `fastparquet` absent but not required because `pyarrow` is already available
- W&B is now wired for the root Core/Amplifier path.
  - maintained project: `pg-core-amp`
  - static descriptors go to W&B config
  - train/eval history is kept intentionally minimal
  - final runtime/artifact outcomes go to W&B summary
  - step-0 eval is no longer logged
- Representative 5090 throughput check:
  - Current recommended structure/controller shape
  - No compile
  - `seq_len=512`, `batch_size=256`, `bptt_chunks=2`
  - Short 8-step probe reached about `0.9M tok/s` steady-state on the local RTX 5090

## Drift and Issues Found
- README drift existed.
  - The top fork note already described the root Core/Amplifier path.
  - Later README sections still described upstream `train_gpt.py` behavior and old final artifact logging.
  - This has been partially corrected so the root path now points at structured run artifacts.
- Old sweep summaries were too fragile.
  - Prior sweep scripts appended hand-built TSV rows and relied on sparse `metrics.jsonl`.
  - There was no `resolved_config.json`.
  - There was no stable per-run runtime metadata file.
  - There was no Markdown summary rebuild output.

## Harness Changes Landed
- Added canonical experiment utilities in `core_amplifier_lm/experiment.py`.
- `train_core_amplifier.py` now writes:
  - `resolved_config.json`
  - `run_metadata.json`
  - `run_results.json`
  - richer `metrics.jsonl` with both train and eval rows
- Added canonical sweep runner:
  - `tools/run_core_amp_sweep.py`
- Converted bash scripts into thin wrappers:
  - `scripts/sweep_controller.sh`
  - `scripts/sweep_structure.sh`
- Replaced log-scrape summary rebuild with structured summary rebuild:
  - `tools/rebuild_summary.py`
  - outputs both `summary.tsv` and `summary.md`
- Added explicit W&B integration in `train_core_amplifier.py` and the sweep runner defaults.
  - real sweeps now default to `pg-core-amp`
  - smoke presets remain available without polluting the online project

## Current Limitations
- I did not auto-launch the broad 5090 screening or confirmation sweeps.
  - This repo’s guardrail says the user owns non-trivial sweep execution unless they explicitly request the exact run.
  - The harness is ready and the phase reports below include exact commands to launch.
- Resume accounting is improved but still assumes the current structured artifacts are the source of truth.
  - Fresh runs created by the new harness are the intended path.

## Immediate Recommendation
- Use the new Python-backed sweep runner through the existing bash wrappers.
- Start with:
  - baseline A vs baseline B under a matched token budget
  - structure ablations before wider controller searches
- Keep the first 5090 screening pass no-compile unless a later systems phase proves compile changes ranking for the chosen budget.
