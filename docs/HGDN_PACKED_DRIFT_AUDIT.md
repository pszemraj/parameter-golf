# HGDN Packed Drift Audit

Last updated: 2026-04-21 10:35 EDT

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

That fresh measurement now exists.

- `59d0817a` (2026-04-20 02:38 UTC / 2026-04-19 21:38 CDT): `1xH100`
  live14 packed compile matrix
  closed on the corrected replay shell.
  - `hybrid`: `799.05 ms/step`
  - `model`: `799.32 ms/step`, worse sampled and final quality than `hybrid`
  - `selective`: `875.67 ms/step`, slower than `hybrid` but still alive for the
    exact-8x tiebreak because its final exact roundtrip remained stronger

This closes the "did the packed helper drift?" question for the corrected live14
launch surface. The remaining open choice is now the exact `8xH100`
submission-time tiebreak between `hybrid` and `selective`.

The follow-up `8xH100` results on `2026-04-21 06:15 UTC` clarify the next
question:

- the exact packed-HGDN tiebreak was real and fair enough to answer
  `hybrid` vs `selective`
- the exact naive-contract sanity batch was real and fair enough to show that
  packed HGDN is still behind the exact repo baseline from `train_gpt.py`

So the packed-path audit is now closed. The remaining work is not
"did the helper drift again?" It is "can the packed HGDN runtime close the
exact-baseline gap?"

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
10. local post-`2026-04-21 06:15 UTC` patchset
   - use FA3 for standard attention on Hopper when available
   - add `DISTRIBUTED_MODE=parallel_muon` to avoid the old DDP-plus-Muon stack
   - default active packed helpers back to `COMPILE_STRATEGY=hybrid`
   - hard-fail packed HGDN training if the FLA fast recurrence path is missing
   - make the naive-contract helper fair by default with
     `USE_WANDB=0` / `WANDB_WATCH=none`
   - record git/branch/host/timestamp plus attention/distributed settings in
     the exact-8x helper manifests

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
- the corrected live14 `1xH100` compile matrix says the packed path is back in
  the expected performance range on a valid contract
- the exact `8xH100` packed tiebreak says the packed path is still alive, with
  `selective` slightly stronger on final quality and `hybrid` faster on step
  time
- the exact naive-contract sanity batch says the branch-goal problem is now
  clear: packed HGDN still loses to the exact repo baseline and the hybrid
  attention-only control stays close enough to the baseline to make that
  comparison believable

What remains open:

- rerun the exact `8xH100` packed tiebreak on the patched runtime surface
  (`ATTN_USE_FLASH_ATTN3=1`, `DISTRIBUTED_MODE=parallel_muon`)
- rerun the exact naive-contract sanity batch on that same patched HGDN surface
- quantify how much of the exact-baseline gap closes before spending H100 time
  on any new HGDN ablation

What this audit says today:

- do **not** describe the current branch as already having a proven packed-path
  slowdown
- do **not** use the `5f3a755e` recheck bundles as evidence against the live
  packed `14L` finalist
- do treat the live14 compile matrix at `59d0817a` as the first valid post-fix
  packed timing measurement
- do use `hybrid` as the packed helper default unless a comparison explicitly
  says otherwise
- do keep `selective` alive only for the exact-8x tiebreak, not as the default
- do measure every new packed-path claim against the exact repo baseline from
  `train_gpt.py`, not against HGDN-internal controls alone

## Next action after this audit

Use the corrected packed helper path, not the old kernel-only launch surface,
for the exact `8xH100` packed compile tiebreak:

```bash
USE_WANDB=1 WANDB_MODE=online \
ATTN_USE_FLASH_ATTN3=1 \
DISTRIBUTED_MODE=parallel_muon \
RUN_PREFIX_BASE=h100packed_tiebreak \
bash scripts/run_h100_hgdn_compile_tiebreak_round.sh
```
