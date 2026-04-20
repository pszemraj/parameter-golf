# HGDN Branch Status

Last updated: 2026-04-20 13:40 CDT

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

- The next packed-path H100 steps are:
  - exact `8xH100` live14 packed tiebreak: `hybrid` vs `selective`
  - bounded naive-contract sanity batch:
    - exact repo baseline from `train_gpt.py`
    - live HGDN finalist
    - hybrid-trainer attention-only control
    - direct baseline leg explicitly pins `DATA_PATH`, `TOKENIZER_PATH`, and
      `VOCAB_SIZE`

Exact 8x packed tiebreak:

```bash
USE_WANDB=0 WANDB_MODE=offline \
RUN_PREFIX_BASE=h100packed_tiebreak \
bash scripts/run_h100_hgdn_compile_tiebreak_round.sh
```

Naive-contract sanity batch:

```bash
USE_WANDB=0 WANDB_MODE=offline \
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
