# Redundancy Audit

Last updated: 2026-04-08 01:40 EDT

This document is the interim cleanup checkpoint before returning to HGDN kernel work. The goal is to identify code and test paths that do not justify their independent existence.

## Summary

The codebase does not look bloated because of large amounts of obviously unreachable model logic. The bigger problem is parallel implementations that drifted:

- profiler-analysis workflow code forked into multiple near-identical scripts
- trainer-side data/eval/quantization utilities were copied from the baseline and then diverged
- local phase-1 shell protocol repeats the same HGDN environment contract in too many places
- the test suite has a few table-driven opportunities that are still written as hand-expanded individual tests

The cleanest immediate cleanup target is the profiler/report toolchain. The riskiest consolidation target is the baseline-vs-hybrid trainer utility layer, because those copies now differ in validation strictness and artifact semantics.

## Progress

- `scripts/hgdn_cuda_preflight.py` and `scripts/profile_hgdn_local_hotpath.py`
  now share the same CUDA module-preparation helper instead of carrying their
  own copies of the mixed-precision and conv-freeze logic.
- `scripts/hgdn_cuda_parity.py` now uses one parameterized packed-conv parity
  helper for the three backward-ownership variants instead of repeating the same
  test body three times.

## Findings

| ID | Area | Type | Recommendation | Risk |
|---|---|---|---|---|
| A1 | `scripts/compare_profiler_reports.py` | dead-ish duplicate | Delete or merge into the phase-1 comparator | low |
| A2 | profiler comparison helpers | semantic duplicates | Extract shared row/CSV/markdown helpers | low |
| A3 | `scripts/run_hgdn_local_phase1.sh` | semantic duplication | Extract shared env-contract builder | low-medium |
| A4 | `env_flag` in local GPU tools | semantic duplicate | Extract shared helper | low |
| A5 | baseline vs hybrid tokenizer/data helpers | drifted near-duplicates | Extract + parameterize | medium |
| A6 | baseline vs hybrid quantization helpers | drifted near-duplicates | Extract a shared core, keep trainer wrapper | high |
| A7 | baseline vs `model.py` transformer utilities | duplicate by design but drifting | Defer, document boundary | high |
| A8 | contiguity/parity tests in `test_model.py` | test redundancy | Parameterize and extract test helpers | low-medium |
| A9 | `scripts/export_wandb_hgdn_runs.py` | orphan utility | Defer: either document it or archive it | low |

## Detailed Findings

### A1. Generic profiler comparator is effectively superseded

Files:

- `scripts/compare_profiler_reports.py:1-206`
- `scripts/compare_hgdn_phase1.py:1-344`

Why this is a finding:

- `compare_profiler_reports.py` compares two structured reports across the same default HGDN transfer buckets.
- `compare_hgdn_phase1.py` already does that, but on the real artifact shape we now use: full phase-1 bundles plus boundary-layout deltas.
- `compare_profiler_reports.py` is not referenced from `docs/README.md`, `docs/TODO.md`, or `docs/HARDWARE_TRANSFER.md`.

Recommendation: `Delete`

- Preferred outcome: fold any genuinely useful generic comparison behavior into `compare_hgdn_phase1.py`.
- If a generic report-vs-report comparison is still needed, keep one tool and add an explicit `--generic-report` mode rather than maintaining two separate scripts.

Risk flags:

- Low risk if no one is using the generic script manually.
- Before deletion, confirm there is no external notebook or shell alias depending on it.

### A2. Profiler-analysis helpers are duplicated across three scripts

Files:

- `scripts/analyze_hgdn_phase1.py:24-42,103-128,256-268`
- `scripts/compare_hgdn_phase1.py:24-42,126-186`
- `scripts/compare_profiler_reports.py:18-36,57-103`
- `scripts/export_wandb_hgdn_runs.py:115-135`
- `profiler_report.py:17-281`

Concrete duplication:

- the same HGDN bucket list exists independently in all three profiler scripts
- `find_row(...)` is reimplemented in all three comparison/analyzer scripts
- CSV writers are reimplemented in four places
- markdown rendering logic for before/after bucket tables is split across scripts instead of sharing one renderer

Recommendation: `Extract`

- Move the following into `profiler_report.py` or a new `profile_compare_utils.py`:
  - canonical HGDN transfer bucket list
  - `find_row`
  - row-to-ms / row-to-percent helpers
  - heterogenous CSV writing
  - markdown table rendering helpers

Risk flags:

- Low risk if the extraction preserves current file outputs.
- Avoid over-generalizing too early; the shared module should reflect the current HGDN profiling workflow, not a speculative generic profiling framework.

### A3. `run_hgdn_local_phase1.sh` repeats the same HGDN env contract too many times

File:

- `scripts/run_hgdn_local_phase1.sh:98-104`
- `scripts/run_hgdn_local_phase1.sh:107-213`

Why this is a finding:

- The script writes a `commands.sh` snapshot containing long inline `GDN_*` env prefixes.
- It then repeats essentially the same env block again for the actual preflight, hotpath, and trainer invocations.
- The same family of flags is duplicated four more times inside the execution section.

Recommendation: `Extract`

- Build one shell helper that prints or exports the HGDN env contract once.
- Use it for:
  - `env_snapshot.txt`
  - `commands.sh`
  - actual sequential execution

Risk flags:

- Low-medium risk because shell quoting bugs are easy to introduce.
- Keep the launch summary explicit; do not hide the experiment contract behind opaque shell indirection.

### A4. `env_flag` is duplicated across local GPU tools

Files:

- `scripts/profile_hgdn_local_hotpath.py:69-76`
- `scripts/hgdn_cuda_preflight.py:44-51`

Why this is a finding:

- It is the same boolean env parsing logic in two files.
- These tools are part of the same local HGDN workflow and should not drift on basic flag parsing.

Recommendation: `Extract`

- Put the helper in a small local utility module, or import it from one script into the other if we want to keep the surface minimal.

Risk flags:

- Low risk.

### A5. Baseline and hybrid tokenizer/data helpers are the same family, but they are already drifting

Files:

- `train_gpt.py:210-246`
- `train_gpt.py:499-570`
- `train_gpt_hybrid.py:389-547`

Affected symbols:

- `build_sentencepiece_luts`
- `load_validation_tokens`
- `load_data_shard`
- `TokenStream`
- `DistributedTokenLoader`

Why this is a finding:

- `build_sentencepiece_luts` is semantically the same function in both trainers.
- `TokenStream` and `DistributedTokenLoader` do the same job with different front doors:
  - baseline takes a glob pattern
  - hybrid takes a pre-resolved file list
- `load_validation_tokens` and `load_data_shard` used to be baseline copies, but now differ in strictness:
  - baseline checks missing files, shard byte sizes, and short reads
  - hybrid has a cleaner interface but weaker validation

Recommendation: `Extract` + `Parameterize`

- Extract shared shard/token utility code into one module.
- Parameterize the input shape difference:
  - accept `list[Path]`
  - keep a thin pattern-resolver wrapper where needed
- Preserve strict validation instead of keeping one strict copy and one relaxed copy.

Risk flags:

- Medium risk because these helpers are experiment protocol, not just plumbing.
- Merging them blindly could change failure behavior in the baseline trainer or mask corrupt data in the hybrid trainer.

### A6. Quantization helpers are duplicated and more dangerously drifted

Files:

- `train_gpt.py:399-491`
- `train_gpt_hybrid.py:625-724`
- `train_gpt_hybrid.py:727-754`

Affected symbols:

- `quantize_state_dict_int8`
- `dequantize_state_dict_int8`
- hybrid-only wrapper: `serialize_quantized_state_dict_int8`

Why this is a finding:

- The baseline and hybrid quantization paths share the same artifact purpose and format family.
- They now differ in implementation details:
  - passthrough handling
  - stats accounting
  - dtype restoration
  - quantization helper factoring
- This is exactly the kind of drift that produces confusing artifact deltas later.

Recommendation: `Extract`

- Extract a shared quantization core that owns:
  - tensor classification
  - scale computation
  - payload assembly
  - dequantization
- Keep trainer-specific wrappers around:
  - serialization
  - byte-audit reporting
  - W&B summary emission

Risk flags:

- High risk.
- This touches submission bytes, artifact compliance, and possibly public payload format.
- Any consolidation here requires byte-for-byte regression checks, not just tensor-shape tests.

### A7. `train_gpt.py` and `model.py` duplicate transformer utilities, but not under the same invariants

Files:

- `train_gpt.py:613-740`
- `model.py:197-287`
- `model.py:756-892`

Affected symbols:

- `CastedLinear`
- `Rotary`
- `apply_rotary_emb`
- `MLP`
- `RMSNorm`
- `validate_norm_style`

Why this is a finding:

- These are obviously related copies.
- They are not safe blind merges anymore:
  - `model.py` has profiling ranges and newer dtype fast paths
  - `train_gpt.py` is the reference baseline script and intentionally simpler
  - `MLP` is not the same block anymore (`relu^2` vs squared LeakyReLU)

Recommendation: `Defer`

- Do not merge these immediately.
- Instead, document the boundary explicitly:
  - `train_gpt.py` remains a self-contained reference trainer
  - `model.py` remains the HGDN/hybrid model implementation
- Revisit only if we decide the baseline script should stop being standalone.

Risk flags:

- High risk false positive.
- This code looks similar syntactically but does not operate under the same semantics anymore.

### A8. `test_model.py` has parameterization opportunities

Files:

- `test_model.py:135-147`
- `test_model.py:177-309`
- `test_model.py:312-354`

Why this is a finding:

- Three contiguous-layout tests cover the same family of invariants with hand-written setups:
  - `test_causal_conv_output_contiguous`
  - `test_gdn_conv_output_contiguous`
  - `test_gdn_qk_only_contiguous`
- The packed parity tests duplicate a large amount of state-dict copying logic:
  - `test_gdn_packed_qkv_conv_matches_separate_path`
  - `test_gdn_packed_qkv_proj_conv_matches_separate_path`

Recommendation: `Parameterize` + `Extract`

- Convert the contiguity tests into a small table-driven family.
- Extract a helper that copies separate q/k/v weights into a packed test module, then reuse it in both parity tests.

Risk flags:

- Low-medium risk.
- Do not collapse the utility-level conv test and the HGDN integration-level contiguity test into one; they cover different semantic layers.

### A9. `export_wandb_hgdn_runs.py` is an orphan utility

File:

- `scripts/export_wandb_hgdn_runs.py:1-204`

Why this is a finding:

- It is not referenced in the current docs workflow.
- It reimplements CSV/JSON writing instead of sharing the existing reporting utilities.
- It may still be useful, but right now it reads like a one-off script that became part of the permanent surface by inertia.

Recommendation: `Defer`

- Either document it properly as the canonical W&B export tool, or move it to an archive/one-off location.

Risk flags:

- Low risk if left alone temporarily.
- Medium maintenance risk if we keep adding one-off export scripts without a tool boundary.

## Test-side non-findings

These looked redundant at first but should stay distinct:

- `test_causal_conv_output_contiguous` vs `test_gdn_conv_output_contiguous`
  - utility-level conv layout vs HGDN integration-level recurrence-input layout
- `test_gdn_packed_qkv_conv_matches_separate_path` vs `test_gdn_packed_qkv_proj_conv_matches_separate_path`
  - conv packing parity vs projection+conv packing parity

These should be parameterized or helper-ized, not deleted.

## Dead-code verdict

Strong dead-code candidates found in this pass:

- `scripts/compare_profiler_reports.py` is the only clear one.

What this audit did **not** find:

- a large amount of unreachable model logic
- permanently-off feature flag branches in the HGDN path
- obviously stale fallback branches inside `model.py`

So the cleanup priority is not "delete lots of dead branches." It is "stop carrying multiple copies of the same workflow logic."

## Recommended cleanup order

1. Remove or merge `scripts/compare_profiler_reports.py`.
2. Extract shared profiler/report helpers and default HGDN buckets.
3. Deduplicate the HGDN env-contract plumbing in `scripts/run_hgdn_local_phase1.sh`.
4. Consolidate `env_flag`.
5. Decide whether the W&B export script is canonical or archival.
6. Only then touch baseline-vs-hybrid trainer utility extraction.
7. Defer `train_gpt.py` vs `model.py` consolidation until there is an explicit decision to stop treating the baseline as a standalone reference.
