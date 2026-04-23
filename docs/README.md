# HGDN Branch Status

Last updated: 2026-04-22 15:00 EDT

Branch: `exp/hgdn-k-core`

## Current position

- Active packed finalist replay preset:
  [`winner_20260405_19_live14.toml`](../configs/hgdn/winner_20260405_19_live14.toml)
- Archived kernel-only preset:
  [`winner_20260405_19.toml`](../configs/hgdn/winner_20260405_19.toml)
- Active packed flags:
  - `GDN_CONV_OUTPUT_CONTIGUOUS=1`
  - `GDN_USE_PACKED_QKV_CONV=1`
  - `GDN_USE_PACKED_QKV_PROJ=1`
  - `GDN_CONTROL_PROJ_FP32=0`
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`
- Live architecture:
  - `14L x 384d x mlp3.25`
  - proxy reference: `h100pack3_b_fixed2k_hybrid_r1_mlp3.25_seq2048`
  - exact bridge reference: `h100bridge1_exact_hybrid_r1_mlp3.25_seq2048`
  - exact bridge result: `2.3949` stop-step eval, `2.4206` final roundtrip,
    `UNDER_LIMIT`, headroom `834,652`
- Attention-only baseline lost the exact bridge and failed the artifact cap on
  that contract.
- Full-block megakernel and core-kernel work are both archived research paths.
  Packed HGDN is the active mainline again.
- The HGDN recurrence path no longer graph-breaks on the upstream FLA
  lock/context-manager boundary.
- Real HGDN ablations go to `pg-hgdn-ablations`.
- Local Python commands on this checkout use `conda run -s --name pg ...`.

## Active decisions

- Keep HGDN as the live record-path family.
- Keep `train_gpt.py` as the absolute baseline reference; the hybrid-trainer
  attention-only control is diagnostic only.
- The next HGDN work is contract-native shell search on the exact
  baseline-shaped surface, not more blind replay of the live14 shell against
  the naive baseline contract.
- Do not reopen broad H100 architecture search unless a new exact run
  contradicts the current bridge result.
- Do not reopen the closed post-conv front-end seam or archived kernel paths
  unless new evidence says the packed path is no longer the right target.

## Current H100 work

- `59d0817a` (2026-04-20 02:38 UTC / 2026-04-19 21:38 CDT): the `1xH100`
  live14 packed compile matrix
  closed on the active replay shell.
  - `hybrid`: `799.05 ms/step`, sampled val `4.1849`, final exact roundtrip
    `4.25972394`
  - `model`: `799.32 ms/step`, sampled val `4.2709`, final exact roundtrip
    `4.43016197`
  - `selective`: `875.67 ms/step`, sampled val `4.1869`, final exact roundtrip
    `4.15098498`
- Packed-path default stays `hybrid` for speed-sensitive helpers.
- `selective` stays alive for the exact `8xH100` tiebreak.

- `2026-04-21 06:15 UTC` bundle set:
  - exact `8xH100` live14 packed tiebreak:
    - `hybrid`: `373.13 ms/step`, final exact roundtrip `2.39929889`
    - `selective`: `403.46 ms/step`, final exact roundtrip `2.39044828`
    - both `UNDER_LIMIT`
  - bounded naive-contract sanity batch:
    - exact repo baseline from `train_gpt.py`: `44.00 ms/step`, exact
      roundtrip `1.23710448`, `UNDER_LIMIT`
    - live HGDN finalist: `98.08 ms/step`, exact roundtrip `1.24735121`,
      `OVER_LIMIT`
    - hybrid-trainer attention-only control: `46.09 ms/step`, exact roundtrip
      `1.24098267`, `OVER_LIMIT`
  - the comparison surface was fair enough to show the real problem:
    packed HGDN is still behind the exact baseline, but the helper manifests
    for those two runs did not yet record git provenance.

- Post-`2026-04-21` local patchset before the next rerun:
  - standard attention blocks now use the FA3 fast path when available on
    Hopper and only fall back to SDPA otherwise
  - distributed HGDN runs can use `DISTRIBUTED_MODE=parallel_muon` to avoid the
    old DDP-plus-Muon communication stack
  - `train_gpt_hybrid.py` and `scripts/sweep.sh` now default to
    `COMPILE_STRATEGY=hybrid`
  - packed HGDN training now refuses the silent non-FLA recurrence fallback
  - the naive-contract helper defaults to `USE_WANDB=0` / `WANDB_WATCH=none`
    so timed baseline comparisons are not paying W&B overhead only on the HGDN
    legs
  - the structured launcher now accepts `--compile-strategy selective`
  - new tiebreak and naive-contract manifests include git commit, branch, host,
    timestamp, attention-backend flag, and distributed mode
  - `cb026ab` (2026-04-21 07:01 UTC / 2026-04-21 03:01 EDT): packed trainer
    distributed cleanup
    - bucket replicated non-Muon grad averaging instead of launching one
      `all_reduce` per replicated parameter in `parallel_muon` mode
    - restore baseline-style Muon bank sharding on the DDP path instead of
      computing every Muon bank on every rank
    - align the exact-8x bridge helper with the same
      `ATTN_USE_FLASH_ATTN3=1` / `DISTRIBUTED_MODE=parallel_muon` surface as the
      tiebreak and naive helpers
    - default exact-8x bridge/tiebreak helpers to `WANDB_WATCH=none`
- `2026-04-21` distributed compile follow-up:
    - multi-rank `train_gpt_hybrid.py` now suppresses top-level model compile
      and keeps only the selective submodule compile surface
    - on the exact `8xH100` `parallel_muon` surface, `COMPILE_STRATEGY=hybrid`
      therefore normalizes to the same effective compile plan as `selective`
    - the exact-8x compile helper skips duplicate strategies after that
      normalization instead of launching a broken duplicate arm

## Current local work

- New local baseline-shaped HGDN search helper:
  [`../scripts/run_local_hgdn_naive_contract_search.sh`](../scripts/run_local_hgdn_naive_contract_search.sh)
- Size-screen config:
  [`../configs/hgdn/naive_contract_search.toml`](../configs/hgdn/naive_contract_search.toml)
- First-pass artifact-size screen on `2026-04-22` says these candidates are
  all under the init-time proxy cap:
  - `l9_d512_r1_m2`
  - `l9_d512_r1_m1p75`
  - `l8_d512_r1_m2`
  - `l9_d480_r1_m2`
  - `l9_d512_r0_m2`
- The helper now defaults `PERF_SKIP_FINAL_EVAL=1` for local broad screens so
  the size screen handles artifact triage and the local loop is not dominated
  by the quantized roundtrip tail.
- Local `4070` contract-native screen on `2026-04-23` (`300` steps,
  `seq=1024`, `65536` tokens/step, `WEIGHT_DECAY=0`) says:
  - `l8_d512_r1_m2` is the current local HGDN leader:
    `560.56 ms/step`, `val_bpb=1.8709`
  - `l9_d512_r1_m1p75` is the next-best HGDN trade:
    `630.64 ms/step`, `val_bpb=1.8758`
  - `l9_d480_r1_m2` and `l9_d512_r1_m2` are slower without enough quality gain
  - same-shell `l8_d512_r0_m2` attention-only control:
    `407.30 ms/step`, `val_bpb=1.9888`
  - so the provisional winner-shell HGDN tax is about `1.38x` step time for
    about `0.118` bpb at step `300`
- The helper runs the size screen first, then a fixed-contract local batch,
  then bundles configs/logs/commands with `py7zr`.

Exact 8x packed tiebreak:

```bash
USE_WANDB=1 WANDB_MODE=online \
ATTN_USE_FLASH_ATTN3=1 \
DISTRIBUTED_MODE=parallel_muon \
WANDB_WATCH=none \
RUN_PREFIX_BASE=h100packed_tiebreak \
bash scripts/run_h100_hgdn_compile_tiebreak_round.sh
```

Naive-contract sanity batch:

```bash
USE_WANDB=0 WANDB_MODE=offline \
ATTN_USE_FLASH_ATTN3=1 \
DISTRIBUTED_MODE=parallel_muon \
WANDB_WATCH=none \
RUN_PREFIX_BASE=h100naive1 \
bash scripts/run_h100_hgdn_naive_contract_round.sh
```

The naive-contract helper now pins the direct `train_gpt.py` replay to the
recorded baseline transport/data contract:

- `NCCL_IB_DISABLE=1`
- `DATA_PATH`
- `TOKENIZER_PATH`
- `VOCAB_SIZE`

Details and open items live in [TODO.md](TODO.md).

## Launch entrypoints

Structured launcher:

- [`../scripts/hgdn.py`](../scripts/hgdn.py)

Batch helpers:

- [`../scripts/run_local_hgdn_naive_contract_search.sh`](../scripts/run_local_hgdn_naive_contract_search.sh)
- [`../scripts/run_local_hgdn_resize_round.sh`](../scripts/run_local_hgdn_resize_round.sh)
- [`../scripts/run_h100_hgdn_resize_round.sh`](../scripts/run_h100_hgdn_resize_round.sh)
- [`../scripts/run_h100_hgdn_bridge_round.sh`](../scripts/run_h100_hgdn_bridge_round.sh)
- [`../scripts/run_h100_hgdn_compile_tiebreak_round.sh`](../scripts/run_h100_hgdn_compile_tiebreak_round.sh)
- [`../scripts/run_h100_hgdn_naive_contract_round.sh`](../scripts/run_h100_hgdn_naive_contract_round.sh)
- [`../scripts/run_h100_single_gpu_hgdn.sh`](../scripts/run_h100_single_gpu_hgdn.sh)
- [`../scripts/run_h100_single_gpu_hgdn_profile.sh`](../scripts/run_h100_single_gpu_hgdn_profile.sh)

Bundling and setup:

- [`../scripts/bundle_hgdn_run.py`](../scripts/bundle_hgdn_run.py)
- [`../scripts/bootstrap_challenge_data.sh`](../scripts/bootstrap_challenge_data.sh)

Archived kernel helpers:

- [`../scripts/run_h100_single_gpu_hgdn_corekernel.sh`](../scripts/run_h100_single_gpu_hgdn_corekernel.sh)
- [`../scripts/hgdn_cuda_preflight.py`](../scripts/hgdn_cuda_preflight.py)

## Related docs

- next steps: [TODO.md](TODO.md)
- chronology and scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- packed fairness and replay audit: [HGDN_PACKED_DRIFT_AUDIT.md](HGDN_PACKED_DRIFT_AUDIT.md)
- archived core-kernel notes: [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md)
- W&B schema: [WANDB_SCHEMA.md](WANDB_SCHEMA.md)
- local vs H100 transfer notes: [HARDWARE_TRANSFER.md](HARDWARE_TRANSFER.md)
- legacy fused-CUDA notes: [HGDN_CUDA_FUSED.md](HGDN_CUDA_FUSED.md)
- cleanup backlog: [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md)
- external architecture reference: [REFERENCE.md](REFERENCE.md)
