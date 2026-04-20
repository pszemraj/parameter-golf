# HGDN Megakernel Extracted Model Contract

## Live forward math

The live HGDN block contract comes from
[`../model.py`](../model.py).

1. Input `x` is shaped `(B, T, D)` and is normally `bf16` on CUDA.
2. Packed qkv projection is dense:
   `w_qkv.weight` has shape `(H * (2 * Dk + Dv), D)`.
   The packed output is `(B, T, H * (2 * Dk + Dv))`.
3. Packed causal depthwise conv uses PyTorch `Conv1d(total_dim, total_dim, K, groups=total_dim, bias=False)` with weight layout `(channels, 1, K)`.
   The active packed path flattens that to `(channels, K)` before passing weights to CUDA.
4. Conv ordering matches eager `Conv1d(..., padding=K-1)[..., :T]`.
   Tap `0` multiplies the oldest visible token and tap `K-1` multiplies the current token.
5. Post-conv activation is `SiLU` on all q/k/v channels before any split-specific logic.
6. Q and K split into `(B, T, H, Dk)`, then L2-normalize over the last dimension with `eps=1e-6`.
   Eager code uses `F.normalize(..., p=2, dim=-1, eps=1e-6)` and casts the result back to the activation dtype.
7. V splits into `(B, T, H, Dv)` with no normalization.
8. Control projections are dense over `D`:
   `w_a.weight` shape `(H, D)`,
   `w_b.weight` shape `(H, D)`,
   `w_g.weight` shape `(H * Dv, D)`.
9. Decay gate math is:
   `g_pre = w_a(x)`,
   `g = -exp(A_log) * softplus(g_pre + dt_bias)`,
   with the pointwise math in fp32 when `gates_fp32=True`,
   then cast back to the activation dtype.
10. Beta path is:
    `beta_pre = w_b(x)`,
    `beta = sigmoid(beta_pre)`,
    `beta *= 2.0` when `allow_neg_eigval=True`,
    then cast to the activation dtype.
11. Recurrence inputs are:
    `q, k` in activation dtype,
    `v` in activation dtype,
    `alpha = exp(g)` interpreted in fp32,
    `beta` interpreted in fp32.
12. Reference recurrence is
    [`gdn_recurrent_naive`](../model.py):
    `S_t = alpha_t * (S_{t-1} - beta_t * k_t (k_t^T S_{t-1})) + k_t v_t^T`
    and
    `o_t = S_t^T q_t`.
    State shape is `(B, H, Dk, Dv)`.
13. Output gate path is:
    `g_out = w_g(x).view(B, T, H, Dv)`.
14. Output normalization is parameter-free RMSNorm over the last dimension with `eps=1e-6`.
    When `output_norm_fp32=True`, eager code does `rms_norm(o.float()).to(x.dtype)`.
15. Final gated output is:
    `z = rms_norm(o) * SiLU(g_out)`.
16. Dense output projection is cross-head:
    `w_out.weight` shape `(D, H * Dv)`,
    output is `w_out(z.reshape(B, T, H * Dv))`.

## Shapes and dtype policy

- `x`: `(B, T, D)`, usually `bf16`
- `w_qkv`: `(H * (2 * Dk + Dv), D)`, usually `bf16`
- `w_a`, `w_b`: `(H, D)`, usually `bf16` in the current winner config
- `w_g`: `(H * Dv, D)`, usually `bf16`
- `w_out`: `(D, H * Dv)`, usually `bf16`
- `conv_w`: eager storage `(channels, 1, K)`, megakernel storage `(channels, K)`, usually `bf16`
- `A_log`, `dt_bias`: `(H,)`, fp32
- Gate math, norm reductions, and recurrence state: fp32
- Saved forward activations and parameter grads for feature maps: bf16

## Runtime megakernel preconditions

The repo-backed megakernel path now treats these as hard runtime contract
checks rather than hidden wrapper fixes:

- when `use_cuda_megakernel=True` and `x` is CUDA `bf16`, missing extension
  availability is a hard error, not a silent eager fallback
- `x`, `w_qkv`, `w_a`, `w_b`, `w_g`, `w_out`, `conv_w`, and backward `grad_y`
  must already be CUDA `bf16` contiguous tensors
- `A_log` and `dt_bias` must already be CUDA `fp32` contiguous tensors
- `w_a`, `w_b`, and `w_g` must remain `bf16`, so
  `GDN_CONTROL_PROJ_FP32=0` is required in megakernel mode
- correctness/parity builds omit `--use_fast_math`; any fast-math variant is a
  separate opt-in performance experiment, not the default validation target

## Winner-style contract used for the first pass

From
[`../configs/hgdn/current_winner_retune.toml`](../configs/hgdn/current_winner_retune.toml):

- `D = 384`
- `H = 8`
- `Dk = 48`
- `expand_v = 1.0`, so `Dv = 48`
- `K = 4`
- `use_packed_qkv_conv = true`
- `use_packed_qkv_proj = true`
- `conv_output_contiguous = true`
- `gdn_control_proj_fp32 = false`
- `gates_fp32 = true`
- `output_norm_fp32 = true`

## Real mismatches versus the draft `local-scratch/hgdn_megakernel_v3`

1. The draft hard-requires H100 at runtime. The repo contract needs local `sm_89` compile/parity support with correctness-only labeling.
2. The draft chooses a fixed cooperative grid size (`2 * SMs`). The repo path needs an occupancy-based cooperative launch size because register pressure can invalidate that assumption.
3. The draft test harness uses synthetic standalone tensors. The repo path needs real `GatedDeltaNet` weight/layout extraction and eager reference gradients from the live module.
4. The draft does not live behind a repo runtime flag. The repo path needs a dedicated `use_cuda_megakernel` flag and trainer/env plumbing.
5. The draft reports `W_out` as dense and cross-head, which is correct, but its validation harness does not exercise the live module’s actual `CastedLinear` weights and conv storage.
6. The draft assumes only an isolated extension workflow. The repo path needs both an in-place build script and an optional JIT build path, with explicit `TORCH_CUDA_ARCH_LIST` handling.
