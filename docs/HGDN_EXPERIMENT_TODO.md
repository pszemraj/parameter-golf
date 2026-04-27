# HGDN Experiment Follow-Ups

These are deferred from the April 26 experiment-safety pass. They are not
blockers for the local adaptive pipeline, but should be closed before
paid H100 finalist runs or official-style comparisons.

## Before Paid H100 Runs

- Tighten analyzer size eligibility for config promotion so missing final
  artifact status or missing size-screen status is explicit, not silently legal.
- Add full provenance to all manifests: `ngpu`, `grad_accum_steps`, data path,
  tokenizer path, vocab size, and artifact checksums/sizes when final artifacts
  are produced.

## Before Broad Local Sweeps

- Finish the runner migration away from user-facing environment variables.
  `scripts/run_local_hgdn_adaptive_pipeline.py`,
  `scripts/run_local_hgdn_wallclock_resolver.py`, and
  `scripts/run_local_hgdn_batch_ladder.py` are argparse-first now, but the
  trainers and older local sweep helpers still consume an internal environment
  adapter. After the new runners validate on one real local run, replace the
  trainer `Hyperparameters` environment surface with an argparse/dataclass
  config path and delete the remaining shell helper wrappers once their
  orchestration has moved into Python.
- Replace prefix-only validation sampling with either full validation or fixed
  multi-window validation samples for confirmation stages.
- Add `eligible_for_promotion` and `ineligible_reason` columns so human summaries
  cannot visually rank ineligible rows as if they were promotable.
- Factor the CUDA-active-job guard into `hgdn_shell_common.sh` and reuse it in
  the standalone local search helper.
- Default standalone local search to offline W&B or require an explicit opt-in
  before writing a 19-run batch to the official project.
