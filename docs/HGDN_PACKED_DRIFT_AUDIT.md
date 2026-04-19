# HGDN Packed Drift Audit

Last updated: 2026-04-19 18:50 CDT

Branch: `exp/hgdn-k-core`

Purpose: close the "did the packed HGDN path actually regress?" question before
spending more H100 time.

## Scope

This audit covers only the active packed HGDN path:

- packed QKV projection
- packed QKV conv custom backward
- normal packed HGDN recurrence path
- `train_gpt_hybrid.py` runtime / compile behavior

It explicitly does **not** treat the archived full-block megakernel or the
research-only core-kernel path as active packed-path evidence.

## Short conclusion

There is **no confirmed packed-path performance regression** from the current
branch state yet.

The big apparent discrepancy was mostly a **measurement-contract bug**:

- the first `h100packed_recheck` reruns at commit `5f3a755e` were not the legal
  live packed finalist shell
- they silently ran the old packed kernel-only config on a `16L x 384d`
  architecture shell
- both Colab and RunPod archives confirm that same invalid launch shape

The current packed-path branch now defaults to the correct live replay shell,
and the remaining active code changes after the core-kernel split are either:

- helper fairness fixes, or
- local packed-runtime cleanups that are intended to reduce overhead

The next H100 packed rerun should therefore be interpreted as a **fresh valid
measurement**, not as a "regression confirmation" run.

## Historical references

### Reference A: old packed 16-layer winner shell

- Run: `h100k6_fixed2k_hybrid_r1_mlp3.25_seq2048`
- Date in log: 2026-04-05
- Commit at run time: `7677396`
- Launcher:
  - `python scripts/hgdn.py h100-perf fixed2k --preset current-winner ...`
- Contract:
  - `1xH100`
  - `TRAIN_SEQ_LEN=2048`
  - `TRAIN_BATCH_TOKENS=524288`
  - `COMPILE_STRATEGY=model`
- Shape:
  - helper `single` shell at that time
  - `16L x 384d`
  - runtime overrides `GDN_RATIO=1`, `MLP_MULT=3.25`
  - packed winner flags from `current-winner`
- Logged result:
  - `step_avg = 915.10 ms`

This is a real packed-path reference, but it is **not** the same architecture
shell as the later live `14L` finalist.

### Reference B: live legal 14-layer finalist family

- Run: `h100retune_a_fixed2k_hybrid_r1_mlp3.25_seq2048`
- Date in log: 2026-04-08
- Commit hash: not recorded in the current profiling-log excerpt
- Shape:
  - `14L x 384d x mlp3.25`
- Logged result:
  - `final roundtrip = 2.4243`
  - `last step time = 897.96 ms`
  - `artifact total = 15,397,504`
  - `UNDER_LIMIT`

This is the relevant apples-to-apples live packed finalist family for the
current `live14` helper path.

### Reference C: live proxy reference entering exact bridge

- Run: `h100pack3_b_fixed2k_hybrid_r1_mlp3.25_seq2048`
- Date in log: 2026-04-12
- Commit hash: not recorded in the current profiling-log excerpt
- Shape:
  - `14L x 384d x mlp3.25`
  - `local128` proxy point
- Logged result:
  - eval `1500 = 2.3587`
  - roundtrip `= 2.3820`

This is the branch's live packed proxy quality reference, not a direct step-ms
control.

## Invalid recheck runs

These are the runs that created most of the confusion.

### Invalid rerun: Colab

- Archive:
  - `local-scratch/profiling-out-h100packed_recheck-colab-hgdn.7z`
- User-reported code commit:
  - `5f3a755e`
- Archived config:
  - `configs/hgdn/winner_20260405_19.toml`
- Logged shape:
  - `model_params:25279680`
  - `blocks:8G+8A`
- Logged result:
  - `step_avg = 884.87 ms`
  - `artifact_status:OVER_LIMIT`

### Invalid rerun: RunPod

- Archive:
  - `local-scratch/profiling-out-h100packed_recheck-runpod-hgdn.zip`
- User-reported code commit:
  - `5f3a755e`
- Logged shape:
  - `model_params:25279680`
  - `blocks:8G+8A`
- Logged result:
  - `step_avg = 912.12 ms`
  - `artifact_status:OVER_LIMIT`

### Why both are invalid as live packed controls

At that point in history:

- `scripts/run_h100_single_gpu_hgdn.sh fixed2k-hybrid` still used the generic
  `single` sweep preset
- `single` still meant the old `16L x 384d` HGDN shell
- `winner-20260405-19` still pointed to the kernel-only packed config, not the
  exact live `14L` replay shell

So these reruns are useful only for one conclusion:

- Colab vs RunPod H100 SXM are **not** the explanation for the discrepancy
- the two environments landed in the same range on the same invalid `16L`
  helper contract

## Current branch changes that affect the packed path

Active packed-path changes after the core split:

1. `12d3fbc`
   - added the exact live packed replay preset
   - contract fix, not a model/runtime regression source
2. `4b67b4a`
   - pinned host staging and rotary dtype caching
   - packed runtime cleanup
3. `fa4f9ca`
   - pinned validation staging buffers
   - packed runtime cleanup
4. `515ad92`
   - warmup-batch reuse and redundant zero-grad trimming
   - packed runtime cleanup
5. `e08be3a`
   - timing sync cleanup and compile-safe cast matching
   - packed runtime cleanup
6. `7c1676c`
   - narrower token staging and rank-local span loading
   - packed runtime cleanup
7. `0a433c9`
   - all packed H100 helper entrypoints now default to the live14 replay shell
   - fairness fix, not a trainer-speed regression source
8. `d3842b6`
   - prewarm one eval graph before the timed loop
   - packed compile cleanup
9. `e77fedd`
   - use `torch.no_grad()` on packed eval path to remove an extra dispatch-key
     recompile
   - packed compile cleanup

## Current branch changes that do not count as packed drift

These exist on the branch but are not active when running the packed mainline:

- core-kernel runtime path
- full-block megakernel path
- owned-kernel trainer guards and extension preflight for those paths

With `GDN_USE_CUDA_COREKERNEL=0` and `GDN_USE_CUDA_MEGAKERNEL=0`, they are
dormant branch surface, not active packed HGDN work.

## Current local evidence

Packed HGDN compile-audit smokes on the current branch now show:

- helper fairness fixed by `0a433c9`
- eval-graph prewarm recorded before the timed loop by `d3842b6`
- extra eval dispatch-key recompile removed by `e77fedd`

Tiny local smoke timing moved from roughly:

- `480.93 ms` at `step:1`
- `465.52 ms` step average

to roughly:

- `366.90 ms` at `step:1`
- `379.31 ms` step average

on the same packed HGDN compile-audit contract after the eval cleanup follow-up.

That is not target-hardware proof, but it is evidence that the recent packed
branch changes are moving in the intended direction locally rather than adding
obvious new trainer overhead.

## Audit result

What is closed:

- the large packed discrepancy is **not** explained by Colab vs RunPod H100
- the first `h100packed_recheck` reruns were not fair live packed controls
- the current branch no longer has that helper/config bug

What remains open:

- one fresh fair `1xH100` packed rerun from the corrected helper defaults is
  still needed before making a new absolute packed step-ms claim

What this audit says today:

- do **not** describe the current branch as already having a proven packed-path
  slowdown
- do **not** use the `5f3a755e` recheck bundles as evidence against the live
  packed `14L` finalist
- do treat the next packed H100 rerun as the first valid post-fix packed timing
  measurement

## Next action after this audit

Use the corrected packed helper path, not the old kernel-only launch surface,
for the next bounded packed H100 rerun.
