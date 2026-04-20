# HGDN Next Steps

Last updated: 2026-04-19 22:30 CDT

## 1. Recover the packed-path speed floor on H100

- Run the live packed finalist replay under the dedicated compile-strategy
  matrix helper:

```bash
USE_WANDB=0 WANDB_MODE=offline \
RUN_PREFIX=h100packed_compilematrix \
bash scripts/run_h100_single_gpu_hgdn.sh fixed2k-hybrid-compile-matrix
```

- Compare `COMPILE_STRATEGY=model`, `hybrid`, and `selective` on the same live
  replay shell.
- Use the live14 replay shell, not the archived 16-layer kernel-only shell.
- Keep the packed path configured as:
  - `GDN_USE_CUDA_COREKERNEL=0`
  - `GDN_USE_CUDA_MEGAKERNEL=0`
  - `GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1`

## 2. Run the naive-contract sanity batch

- Use [`../scripts/run_h100_hgdn_naive_contract_round.sh`](../scripts/run_h100_hgdn_naive_contract_round.sh).
- Include exactly three legs:
  - exact repo baseline from `train_gpt.py`
  - live HGDN finalist
  - hybrid-trainer attention-only control
- Pin the hybrid-trainer legs to `WEIGHT_DECAY=0`.
- Treat the repo baseline as the calibration anchor. The hybrid attention-only
  control remains diagnostic only.

## 3. Keep HGDN-only finalist work tightly bounded

- After the packed compile matrix and naive-contract batch, keep HGDN-only work
  inside the live `14L x 384d x mlp3.25` family.
- First learning-dynamics screen:
  - `NORM_STYLE=pre`
  - `NORM_STYLE=post`
  - `NORM_STYLE=keel`
- Keep architecture, batch contract, and artifact accounting fixed while
  screening those changes.

## 4. Finish packed-path compile/runtime cleanup only where it still matters

- Treat the packed FLA recurrence path as compile-eligible; do not force it
  into eager-only islands again.
- Keep `TORCH_LOGS=recompiles,graph_breaks` opt-in only.
- Only spend more compile/backend work where the packed H100 matrix or packed
  H100 profiles still show meaningful payoff:
  - `compiled_autograd`
  - regional compilation
  - train/eval grad-mode handling
  - backend swaps

## 5. Leave archived kernel paths archived

- Full-block megakernel: research-only.
- Core-kernel pivot: research-only.
- Do not spend new H100 time on either archived path unless a local rewrite
  produces a materially different result first.

## Recent checkpoints

- `7df6f8d1` (2026-04-18 branch point): core-kernel fork split from `exp/hgdn`.
- `c96ff08` (2026-04-18 19:03 CDT / 2026-04-19 00:03 UTC): clean `1xH100`
  core compare; packed `1191.52 ms/step`, core `6369.37 ms/step`.
- `0a433c9` (2026-04-19 12:47 CDT): packed helper defaults switched to the live
  `single-live14` replay shell.
- `d3842b6` (2026-04-19 12:50 CDT): packed eval path moved to `torch.no_grad()`
  after compile prewarm.
- `f3d8c2f` (2026-04-19 19:08 CDT): packed FLA-backed GDN blocks became
  compile-eligible in `selective` / `hybrid`.

## References

- branch status and commands: [README.md](README.md)
- chronology and exact scoreboards: [PROFILING_LOG.md](PROFILING_LOG.md)
- packed replay audit: [HGDN_PACKED_DRIFT_AUDIT.md](HGDN_PACKED_DRIFT_AUDIT.md)
- archived core-kernel notes: [HGDN_CORE_KERNEL_PLAN.md](HGDN_CORE_KERNEL_PLAN.md)
- cleanup backlog: [REDUNDANCY_AUDIT.md](REDUNDANCY_AUDIT.md)
