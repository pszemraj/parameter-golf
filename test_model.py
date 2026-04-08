"""test_model.py — v3 tests. Run: python test_model.py"""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from hgdn_cuda import (
    packed_qkv_conv_reference,
    packed_qkv_frontend_reference,
    packed_qkv_split_l2norm_reference,
    preact_silu_split_l2norm_nct_backward_reference,
    preact_silu_split_l2norm_nct_reference,
    rmsnorm_silu_gate_reference,
)

from model import (
    SCALAR_PARAM_PATTERNS,
    CausalConv1d,
    GatedDeltaNet,
    HybridGPT,
    PackedCausalConv1d,
    gdn_recurrent_naive,
    l2_norm,
    rms_norm,
    make_baseline_fill,
    make_attention_only_baseline,
    make_hybrid_tight,
    make_hybrid_wide,
    validate_norm_style,
)
from profiler_report import (
    ProfileRow,
    build_profile_report,
    build_profile_rows,
    load_profile_report,
    write_profile_report,
)
from train_gpt_hybrid import (
    normalize_wandb_watch_mode,
    restore_low_dim_params_to_fp32,
    serialize_quantized_state_dict_int8,
)


def _make_bf16_gdn(**kwargs: object) -> GatedDeltaNet:
    """Build the standard bf16 GDN test fixture with fp32 control params restored.

    :param dict kwargs: Extra `GatedDeltaNet` keyword arguments.
    :return GatedDeltaNet: Prepared bf16 test layer.
    """
    layer = GatedDeltaNet(
        64,
        n_heads=4,
        head_k_dim=8,
        expand_v=2.0,
        use_fla=False,
        **kwargs,
    ).bfloat16()
    layer.A_log.data = layer.A_log.data.float()
    layer.dt_bias.data = layer.dt_bias.data.float()
    return layer


def _load_packed_qkv_state(
    separate: GatedDeltaNet,
    packed: GatedDeltaNet,
    *,
    copy_proj: bool,
) -> None:
    """Copy split q/k/v projection and conv weights into a packed HGDN module.

    :param GatedDeltaNet separate: Reference module with split q/k/v path.
    :param GatedDeltaNet packed: Packed module to receive copied weights.
    :param bool copy_proj: Whether to also copy `w_q/w_k/w_v` into `w_qkv`.
    """
    packed_state = packed.state_dict()
    skip_prefixes = ["q_conv.", "k_conv.", "v_conv."]
    if copy_proj:
        skip_prefixes.extend(["w_q.", "w_k.", "w_v."])
    for key, value in separate.state_dict().items():
        if key.startswith(tuple(skip_prefixes)):
            continue
        packed_state[key] = value.clone()
    q_dim = separate.q_conv.conv.weight.shape[0]
    k_dim = separate.k_conv.conv.weight.shape[0]
    packed_state["qkv_conv.conv.weight"][:q_dim] = separate.q_conv.conv.weight.clone()
    packed_state["qkv_conv.conv.weight"][q_dim : q_dim + k_dim] = (
        separate.k_conv.conv.weight.clone()
    )
    packed_state["qkv_conv.conv.weight"][q_dim + k_dim :] = (
        separate.v_conv.conv.weight.clone()
    )
    if copy_proj:
        q_dim = separate.w_q.weight.shape[0]
        k_dim = separate.w_k.weight.shape[0]
        packed_state["w_qkv.linear.weight"][:q_dim] = separate.w_q.weight.clone()
        packed_state["w_qkv.linear.weight"][q_dim : q_dim + k_dim] = (
            separate.w_k.weight.clone()
        )
        packed_state["w_qkv.linear.weight"][q_dim + k_dim :] = (
            separate.w_v.weight.clone()
        )
    packed.load_state_dict(packed_state, strict=True)


def _make_small_gdn(**kwargs: object) -> GatedDeltaNet:
    """Build the standard tiny HGDN config used by validation tests.

    :param dict kwargs: Extra `GatedDeltaNet` keyword arguments.
    :return GatedDeltaNet: Tiny HGDN fixture.
    """
    return GatedDeltaNet(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        **kwargs,
    )


def _make_standard_packed_kwargs(**overrides: object) -> dict[str, object]:
    """Build the default packed-path kwargs used by HGDN CPU fallback tests.

    :param dict overrides: Keyword overrides applied to the shared packed-path config.
    :return dict[str, object]: Packed-path kwargs for `GatedDeltaNet`.
    """
    kwargs: dict[str, object] = dict(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        allow_neg_eigval=True,
        conv_size=4,
        use_fla=False,
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        conv_output_contiguous=True,
    )
    kwargs.update(overrides)
    return kwargs


def _assert_gdn_init_rejected(message: str, **kwargs: object) -> None:
    """Assert that a tiny HGDN fixture rejects an invalid configuration.

    :param str message: Assertion message raised if the config is accepted.
    :param dict kwargs: `GatedDeltaNet` keyword arguments expected to fail validation.
    """
    try:
        _make_small_gdn(**kwargs)
    except ValueError:
        return
    raise AssertionError(message)


def _assert_gdn_cpu_fallback_matches_reference(
    *,
    base_kwargs: dict[str, object],
    candidate_overrides: dict[str, object],
    reference_overrides: dict[str, object] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    grad_atol: float = 1e-5,
    grad_rtol: float = 1e-5,
    success_message: str,
) -> None:
    """Check that a CPU fallback path matches the packed HGDN reference path.

    :param dict[str, object] base_kwargs: Shared kwargs for both reference and candidate layers.
    :param dict[str, object] candidate_overrides: Candidate-only kwargs.
    :param dict[str, object] | None reference_overrides: Optional reference-only kwargs.
    :param float atol: Forward absolute tolerance, defaults to 1e-5.
    :param float rtol: Forward relative tolerance, defaults to 1e-5.
    :param float grad_atol: Backward absolute tolerance, defaults to 1e-5.
    :param float grad_rtol: Backward relative tolerance, defaults to 1e-5.
    :param str success_message: Message printed after the check passes.
    """
    torch.manual_seed(42)
    reference_kwargs = dict(base_kwargs)
    if reference_overrides is not None:
        reference_kwargs.update(reference_overrides)
    candidate_kwargs = dict(base_kwargs)
    candidate_kwargs.update(candidate_overrides)

    reference = GatedDeltaNet(**reference_kwargs)
    candidate = GatedDeltaNet(**candidate_kwargs)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    x_reference = torch.randn(2, 32, 64, requires_grad=True)
    x_candidate = x_reference.detach().clone().requires_grad_(True)
    y_reference = reference(x_reference)
    y_candidate = candidate(x_candidate)
    torch.testing.assert_close(y_candidate, y_reference, atol=atol, rtol=rtol)

    grad = torch.randn_like(y_reference)
    y_reference.backward(grad, retain_graph=True)
    y_candidate.backward(grad)
    torch.testing.assert_close(
        x_candidate.grad,
        x_reference.grad,
        atol=grad_atol,
        rtol=grad_rtol,
    )
    print(success_message)


def _assert_bf16_packed_recurrence_inputs_contiguous(
    success_message: str, **kwargs: object
) -> None:
    """Assert that a bf16 packed-path test fixture materializes contiguous q/k/v outputs.

    :param str success_message: Message printed after the check passes.
    :param dict kwargs: Extra `_make_bf16_gdn` keyword arguments.
    """
    layer = _make_bf16_gdn(**kwargs)
    q, k, v, _, _ = layer._project_recurrence_inputs(
        torch.randn(2, 16, 64, dtype=torch.bfloat16)
    )
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    print(success_message)


def test_recurrence():
    B, T, H, Dk, Dv = 2, 16, 4, 8, 16
    torch.manual_seed(42)
    q = l2_norm(torch.randn(B, T, H, Dk))
    k = l2_norm(torch.randn(B, T, H, Dk))
    v = torch.randn(B, T, H, Dv)
    alpha = torch.sigmoid(torch.randn(B, T, H))
    beta = torch.sigmoid(torch.randn(B, T, H)) * 2.0
    o, s = gdn_recurrent_naive(q, k, v, alpha, beta)
    assert o.shape == (B, T, H, Dv) and s.shape == (B, H, Dk, Dv)
    assert not torch.isnan(o).any()
    print("  ✓ recurrence OK")


def test_gdn_split_projections():
    """Verify projections are separate (not fused) and correctly typed."""
    B, T, D = 2, 16, 64
    torch.manual_seed(42)
    layer = GatedDeltaNet(D, n_heads=4, head_k_dim=8, expand_v=2.0, use_fla=False)
    # Check separate projection attributes exist
    assert hasattr(layer, "w_q") and hasattr(layer, "w_k") and hasattr(layer, "w_v")
    assert hasattr(layer, "w_a") and hasattr(layer, "w_b") and hasattr(layer, "w_g")
    assert not hasattr(layer, "w_fused"), "Fused projection should not exist"
    # Check shapes
    assert layer.w_q.weight.shape == (4 * 8, D)  # (total_qk, d_model)
    assert layer.w_v.weight.shape == (4 * 16, D)  # (total_v, d_model)
    assert layer.w_a.weight.shape == (4, D)  # (n_heads, d_model)
    assert layer.w_b.weight.shape == (4, D)
    assert layer.w_g.weight.shape == (4 * 16, D)  # (total_v, d_model)
    # Check forward/backward
    x = torch.randn(B, T, D, requires_grad=True)
    o = layer(x)
    assert o.shape == (B, T, D)
    o.sum().backward()
    assert x.grad is not None
    no_grad = [n for n, p in layer.named_parameters() if p.grad is None]
    assert not no_grad, f"Missing grads: {no_grad}"
    print(
        f"  ✓ split projections OK (q:{layer.w_q.weight.shape}, a:{layer.w_a.weight.shape})"
    )


def test_gdn_recurrence_input_dtypes():
    """Keep recurrence inputs aligned with activation dtype for FLA kernels."""
    layer = _make_bf16_gdn()
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    q, k, v, g, beta = layer._project_recurrence_inputs(x)
    assert q.dtype == x.dtype
    assert k.dtype == x.dtype
    assert v.dtype == x.dtype
    assert g.dtype == x.dtype
    assert beta.dtype == x.dtype
    print(f"  ✓ recurrence input dtypes OK ({q.dtype})")


def test_muon_routing():
    """Verify SCALAR_PARAM_PATTERNS correctly tags w_a/w_b/w_g for Adam."""
    model = HybridGPT(
        vocab_size=16,
        num_layers=4,
        d_model=64,
        attn_heads=4,
        attn_kv_heads=2,
        gdn_n_heads=4,
        gdn_head_k_dim=8,
        gdn_expand_v=2.0,
        gdn_ratio=3,
        mlp_mult=2,
    )
    muon_params, adam_params = [], []
    for name, p in model.blocks.named_parameters():
        is_scalar = any(pat in name for pat in SCALAR_PARAM_PATTERNS)
        if p.ndim == 2 and not is_scalar:
            muon_params.append(name)
        else:
            adam_params.append(name)
    # w_q, w_k, w_v, w_out should be Muon
    muon_set = set(muon_params)
    for expected in ["w_q.weight", "w_k.weight", "w_v.weight", "w_out.weight"]:
        found = any(expected in n for n in muon_set)
        assert found, f"Expected {expected} in Muon params"
    # w_a, w_b, w_g should be Adam
    adam_set = set(adam_params)
    for expected in ["w_a.weight", "w_b.weight", "w_g.weight", "A_log", "dt_bias"]:
        found = any(expected in n for n in adam_set)
        assert found, f"Expected {expected} in Adam params, got: {adam_set}"
    print(f"  ✓ Muon routing OK ({len(muon_params)} Muon, {len(adam_params)} Adam)")


def test_causal_conv():
    conv = CausalConv1d(32, 4)
    x1 = torch.randn(2, 16, 32)
    x2 = x1.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])
    torch.testing.assert_close(conv(x1)[:, :5], conv(x2)[:, :5], atol=1e-6, rtol=1e-6)
    print("  ✓ causal conv OK")


def test_causal_conv_output_contiguous():
    """Optional contiguous conv materialization should preserve values and improve layout."""
    torch.manual_seed(42)
    conv = CausalConv1d(32, 4)
    conv_contig = CausalConv1d(32, 4, output_contiguous=True)
    conv_contig.load_state_dict(conv.state_dict())
    x = torch.randn(2, 16, 32)
    y = conv(x)
    y_contig = conv_contig(x)
    torch.testing.assert_close(y_contig, y, atol=1e-6, rtol=1e-6)
    assert not y.is_contiguous()
    assert y_contig.is_contiguous()
    print("  ✓ causal conv contiguous output OK")


def test_causal_conv_disable():
    """Disabling the causal conv should become an exact identity."""
    conv = CausalConv1d(32, 4, enabled=False)
    x = torch.randn(2, 16, 32)
    torch.testing.assert_close(conv(x), x, atol=0.0, rtol=0.0)
    print("  ✓ causal conv disable OK")


def test_gdn_conv_toggles():
    """Allow v-path ablations without breaking the HGDN forward path."""
    layer = GatedDeltaNet(
        64,
        n_heads=4,
        head_k_dim=8,
        expand_v=2.0,
        use_fla=False,
        use_v_conv=False,
    )
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert not layer.v_conv.enabled
    print("  ✓ GDN conv toggles OK")


def test_gdn_packed_qkv_conv_matches_separate_path():
    """Packed q/k/v conv should match the separate conv path with copied weights."""
    torch.manual_seed(42)
    kwargs = dict(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        allow_neg_eigval=True,
        conv_size=4,
        use_fla=False,
        use_q_conv=True,
        use_k_conv=True,
        use_v_conv=True,
        conv_output_contiguous=True,
    )
    separate = GatedDeltaNet(**kwargs)
    packed = GatedDeltaNet(**kwargs, use_packed_qkv_conv=True)
    _load_packed_qkv_state(separate, packed, copy_proj=False)
    x = torch.randn(2, 32, 64)
    torch.testing.assert_close(separate(x), packed(x), atol=1e-5, rtol=1e-5)
    print("  ✓ packed qkv conv matches separate path")


def test_gdn_packed_qkv_conv_requires_aligned_settings() -> None:
    """Packed q/k/v conv should reject mixed enablement or layout settings."""
    for message, kwargs in [
        (
            "Expected packed qkv conv to reject disabled v conv",
            dict(use_packed_qkv_conv=True, use_v_conv=False),
        ),
        (
            "Expected packed qkv conv to reject mixed output_contiguous flags",
            dict(
                use_packed_qkv_conv=True,
                q_conv_output_contiguous=True,
                k_conv_output_contiguous=False,
                v_conv_output_contiguous=True,
            ),
        ),
        (
            "Expected packed qkv projection to require packed qkv conv",
            dict(use_packed_qkv_proj=True),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ packed qkv conv rejects incompatible settings")


def test_gdn_packed_qkv_proj_conv_matches_separate_path():
    """Packed qkv projection+conv should match the separate path with copied weights."""
    torch.manual_seed(42)
    kwargs = dict(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        allow_neg_eigval=True,
        conv_size=4,
        use_fla=False,
        use_q_conv=True,
        use_k_conv=True,
        use_v_conv=True,
        conv_output_contiguous=True,
    )
    separate = GatedDeltaNet(**kwargs)
    packed = GatedDeltaNet(
        **kwargs,
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
    )
    _load_packed_qkv_state(separate, packed, copy_proj=True)
    x = torch.randn(2, 32, 64)
    torch.testing.assert_close(separate(x), packed(x), atol=1e-5, rtol=1e-5)
    print("  ✓ packed qkv projection+conv matches separate path")


def test_gdn_packed_qkv_custom_backward_matches_default_path():
    """Packed qkv custom backward should preserve forward and gradient parity."""
    torch.manual_seed(42)
    kwargs = dict(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        allow_neg_eigval=True,
        conv_size=4,
        use_fla=False,
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        conv_output_contiguous=True,
    )
    eager = GatedDeltaNet(**kwargs)
    custom = GatedDeltaNet(**kwargs, use_packed_qkv_conv_custom_backward=True)
    custom.load_state_dict(eager.state_dict(), strict=True)

    x_eager = torch.randn(2, 32, 64, requires_grad=True)
    x_custom = x_eager.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_eager)

    out_eager = eager(x_eager)
    out_custom = custom(x_custom)
    torch.testing.assert_close(out_custom, out_eager, atol=1e-6, rtol=1e-6)

    out_eager.backward(grad, retain_graph=True)
    out_custom.backward(grad)
    torch.testing.assert_close(x_custom.grad, x_eager.grad, atol=1e-5, rtol=1e-5)

    eager_grads = dict(eager.named_parameters())
    custom_grads = dict(custom.named_parameters())
    for name in [
        "w_qkv.linear.weight",
        "qkv_conv.conv.weight",
        "w_a.weight",
        "w_out.weight",
    ]:
        torch.testing.assert_close(
            custom_grads[name].grad,
            eager_grads[name].grad,
            atol=1e-5,
            rtol=1e-5,
        )
    print("  ✓ packed qkv custom backward matches default path")


def test_gdn_packed_qkv_custom_backward_validation() -> None:
    """Packed qkv custom backward should only run on the non-fused packed path."""
    for message, kwargs in [
        (
            "Expected packed qkv custom backward to require packed qkv conv",
            dict(use_packed_qkv_conv_custom_backward=True),
        ),
        (
            "Expected packed qkv custom backward to reject the CUDA fused frontend",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                use_packed_qkv_conv_custom_backward=True,
                use_cuda_fused_frontend=True,
            ),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ packed qkv custom backward validates requirements")


def test_gdn_packed_qkv_single_contig_matches_default_path() -> None:
    """Single packed materialization should preserve packed qkv outputs and grads."""
    torch.manual_seed(42)
    kwargs = dict(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        allow_neg_eigval=True,
        conv_size=4,
        use_fla=False,
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        use_packed_qkv_conv_custom_backward=True,
        conv_output_contiguous=True,
    )
    eager = GatedDeltaNet(**kwargs)
    single = GatedDeltaNet(**kwargs, use_packed_qkv_single_contig=True)
    single.load_state_dict(eager.state_dict(), strict=True)

    x_eager = torch.randn(2, 32, 64, requires_grad=True)
    x_single = x_eager.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_eager)

    out_eager = eager(x_eager)
    out_single = single(x_single)
    torch.testing.assert_close(out_single, out_eager, atol=1e-6, rtol=1e-6)

    out_eager.backward(grad, retain_graph=True)
    out_single.backward(grad)
    torch.testing.assert_close(x_single.grad, x_eager.grad, atol=1e-5, rtol=1e-5)

    eager_grads = dict(eager.named_parameters())
    single_grads = dict(single.named_parameters())
    for name in [
        "w_qkv.linear.weight",
        "qkv_conv.conv.weight",
        "w_a.weight",
        "w_out.weight",
    ]:
        torch.testing.assert_close(
            single_grads[name].grad,
            eager_grads[name].grad,
            atol=1e-5,
            rtol=1e-5,
        )
    _assert_bf16_packed_recurrence_inputs_contiguous(
        "  ✓ packed qkv single-contig matches default path",
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        use_packed_qkv_conv_custom_backward=True,
        use_packed_qkv_single_contig=True,
        conv_output_contiguous=True,
    )


def test_gdn_packed_qkv_single_contig_validation() -> None:
    """Single packed materialization should only run on the non-fused packed path."""
    for message, kwargs in [
        (
            "Expected packed qkv single-contig to require packed qkv conv",
            dict(use_packed_qkv_single_contig=True),
        ),
        (
            "Expected packed qkv single-contig to require packed qkv output_contiguous",
            dict(use_packed_qkv_conv=True, use_packed_qkv_single_contig=True),
        ),
        (
            "Expected packed qkv single-contig to reject the CUDA fused frontend",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_packed_qkv_single_contig=True,
                use_cuda_fused_frontend=True,
            ),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ packed qkv single-contig validates requirements")


def test_gdn_packed_qkv_split_copy_matches_default_path() -> None:
    """Generated split-copy should preserve packed qkv outputs, grads, and contiguity."""
    if not hasattr(torch.ops.aten, "split_with_sizes_copy"):
        print(
            "  - skipping packed qkv split-copy parity (aten.split_with_sizes_copy unavailable)"
        )
        return
    torch.manual_seed(42)
    kwargs = dict(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        use_fla=False,
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        use_packed_qkv_conv_custom_backward=True,
        conv_output_contiguous=True,
    )
    eager = GatedDeltaNet(**kwargs)
    split_copy = GatedDeltaNet(**kwargs, use_packed_qkv_split_copy=True)
    split_copy.load_state_dict(eager.state_dict())

    x_eager = torch.randn(2, 32, 64, requires_grad=True)
    x_split = x_eager.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x_eager)

    out_eager = eager(x_eager)
    out_split = split_copy(x_split)
    torch.testing.assert_close(out_split, out_eager, atol=1e-6, rtol=1e-6)

    out_eager.backward(grad, retain_graph=True)
    out_split.backward(grad)
    torch.testing.assert_close(x_split.grad, x_eager.grad, atol=1e-5, rtol=1e-5)

    eager_grads = dict(eager.named_parameters())
    split_grads = dict(split_copy.named_parameters())
    for name in [
        "w_qkv.linear.weight",
        "qkv_conv.conv.weight",
        "w_a.weight",
        "w_out.weight",
    ]:
        torch.testing.assert_close(
            split_grads[name].grad,
            eager_grads[name].grad,
            atol=1e-5,
            rtol=1e-5,
        )
    _assert_bf16_packed_recurrence_inputs_contiguous(
        "  ✓ packed qkv split-copy matches default path",
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        use_packed_qkv_conv_custom_backward=True,
        use_packed_qkv_split_copy=True,
        conv_output_contiguous=True,
    )


def test_gdn_packed_qkv_split_copy_validation() -> None:
    """Generated split-copy should only run on the non-fused packed path."""
    if not hasattr(torch.ops.aten, "split_with_sizes_copy"):
        print(
            "  - skipping packed qkv split-copy validation (aten.split_with_sizes_copy unavailable)"
        )
        return
    for message, kwargs in [
        (
            "Expected packed qkv split-copy to require packed qkv conv",
            dict(use_packed_qkv_split_copy=True),
        ),
        (
            "Expected packed qkv split-copy to require packed qkv output_contiguous",
            dict(use_packed_qkv_conv=True, use_packed_qkv_split_copy=True),
        ),
        (
            "Expected packed qkv split-copy to reject the CUDA fused frontend",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_packed_qkv_split_copy=True,
                use_cuda_fused_frontend=True,
            ),
        ),
        (
            "Expected packed qkv split-copy to reject packed qkv single-contig",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_packed_qkv_single_contig=True,
                use_packed_qkv_split_copy=True,
            ),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ packed qkv split-copy validates requirements")


def test_packed_qkv_split_l2norm_reference_matches_eager_ops():
    """Reference packed split+l2norm should match eager split plus q/k normalization."""
    torch.manual_seed(42)
    packed = torch.randn(2, 32, 96, requires_grad=True)
    packed_ref = packed.detach().clone().requires_grad_(True)
    q_ref, k_ref, v_ref = packed_qkv_split_l2norm_reference(
        packed_ref,
        n_heads=4,
        head_k_dim=8,
        head_v_dim=8,
    )
    q_eager, k_eager, v_eager = packed.split((32, 32, 32), dim=-1)
    q_eager = l2_norm(q_eager.view(2, 32, 4, 8))
    k_eager = l2_norm(k_eager.view(2, 32, 4, 8))
    v_eager = v_eager.view(2, 32, 4, 8)
    torch.testing.assert_close(q_ref, q_eager, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_ref, k_eager, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(v_ref, v_eager, atol=1e-6, rtol=1e-6)

    grad_q = torch.randn_like(q_eager)
    grad_k = torch.randn_like(k_eager)
    (q_ref * grad_q).sum().backward(retain_graph=True)
    (k_ref * grad_k).sum().backward()
    ref_grad = packed_ref.grad.detach().clone()

    packed_grad = None
    packed_eager = packed.detach().clone().requires_grad_(True)
    q_eager, k_eager, _ = packed_eager.split((32, 32, 32), dim=-1)
    q_eager = l2_norm(q_eager.view(2, 32, 4, 8))
    k_eager = l2_norm(k_eager.view(2, 32, 4, 8))
    (q_eager * grad_q).sum().backward(retain_graph=True)
    (k_eager * grad_k).sum().backward()
    packed_grad = packed_eager.grad.detach().clone()
    torch.testing.assert_close(ref_grad, packed_grad, atol=1e-5, rtol=1e-5)
    print("  ✓ packed split+l2norm reference matches eager ops")


def test_preact_silu_split_l2norm_nct_reference_matches_eager_ops():
    """Reference NCT frontend op should match eager SiLU+split+q/k normalization."""
    torch.manual_seed(42)
    preact = torch.randn(2, 96, 32, requires_grad=True)
    preact_ref = preact.detach().clone().requires_grad_(True)
    q_ref, k_ref, v_ref, inv_q, inv_k = preact_silu_split_l2norm_nct_reference(
        preact_ref,
        n_heads=4,
        head_k_dim=8,
        head_v_dim=8,
    )

    packed = F.silu(preact).transpose(1, 2).contiguous()
    q_eager, k_eager, v_eager = packed.split((32, 32, 32), dim=-1)
    q_eager = q_eager.view(2, 32, 4, 8)
    k_eager = k_eager.view(2, 32, 4, 8)
    v_eager = v_eager.view(2, 32, 4, 8)
    q_norm = l2_norm(q_eager)
    k_norm = l2_norm(k_eager)

    torch.testing.assert_close(q_ref, q_norm, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_ref, k_norm, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(v_ref, v_eager, atol=1e-6, rtol=1e-6)
    assert inv_q.shape == (2, 32, 4)
    assert inv_k.shape == (2, 32, 4)

    grad_q = torch.randn_like(q_ref)
    grad_k = torch.randn_like(k_ref)
    grad_v = torch.randn_like(v_ref)
    grad_ref = preact_silu_split_l2norm_nct_backward_reference(
        grad_q,
        grad_k,
        grad_v,
        preact_ref.detach(),
        q_ref.detach(),
        k_ref.detach(),
        inv_q.detach(),
        inv_k.detach(),
    )

    preact_eager = preact.detach().clone().requires_grad_(True)
    packed = F.silu(preact_eager).transpose(1, 2).contiguous()
    q_eager, k_eager, v_eager = packed.split((32, 32, 32), dim=-1)
    q_eager = l2_norm(q_eager.view(2, 32, 4, 8))
    k_eager = l2_norm(k_eager.view(2, 32, 4, 8))
    v_eager = v_eager.view(2, 32, 4, 8)
    (q_eager * grad_q).sum().backward(retain_graph=True)
    (k_eager * grad_k).sum().backward(retain_graph=True)
    (v_eager * grad_v).sum().backward()
    torch.testing.assert_close(preact_eager.grad, grad_ref, atol=1e-5, rtol=1e-5)
    print("  ✓ NCT preact frontend reference matches eager ops")


def test_packed_qkv_conv_reference_matches_eager_ops():
    """Reference packed causal depthwise conv should match eager packed conv ops."""
    torch.manual_seed(42)
    layer = PackedCausalConv1d((32, 32, 32), kernel_size=4, output_contiguous=True)
    qkv_ref = torch.randn(2, 32, 96, requires_grad=True)
    weight_ref = layer.conv.weight.view(layer.conv.weight.shape[0], -1).detach().clone()
    weight_ref.requires_grad_(True)

    packed_ref = packed_qkv_conv_reference(qkv_ref, weight_ref)
    layer.conv.weight.data.copy_(weight_ref.view_as(layer.conv.weight).detach())
    qkv_eager = qkv_ref.detach().clone().requires_grad_(True)
    packed_eager = layer.forward_packed_tensor(qkv_eager)
    torch.testing.assert_close(packed_ref, packed_eager, atol=1e-6, rtol=1e-6)

    grad = torch.randn_like(packed_ref)
    packed_ref.backward(grad, retain_graph=True)
    grad_qkv_ref = qkv_ref.grad.detach().clone()
    grad_weight_ref = weight_ref.grad.detach().clone()

    layer.zero_grad(set_to_none=True)
    packed_eager.backward(grad)
    torch.testing.assert_close(qkv_eager.grad, grad_qkv_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        layer.conv.weight.grad.view_as(weight_ref),
        grad_weight_ref,
        atol=1e-5,
        rtol=1e-5,
    )
    print("  ✓ packed conv reference matches eager ops")


def test_gdn_cuda_split_norm_cpu_fallback_matches_packed_path() -> None:
    """CPU fallback for CUDA split+l2norm should preserve packed-path outputs and gradients."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(
            use_packed_qkv_conv_custom_backward=True
        ),
        candidate_overrides=dict(use_cuda_split_norm=True),
        success_message="  ✓ CUDA split+l2norm CPU fallback matches packed path",
    )


def test_gdn_cuda_frontend_nct_cpu_fallback_matches_packed_path() -> None:
    """CPU fallback for the NCT frontend op should preserve packed-path outputs and gradients."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(),
        candidate_overrides=dict(use_cuda_frontend_nct=True),
        grad_atol=3e-4,
        grad_rtol=5e-3,
        success_message="  ✓ CUDA frontend NCT CPU fallback matches packed path",
    )


def test_gdn_cuda_frontend_nct_custom_backward_cpu_fallback_matches_custombwd_path() -> (
    None
):
    """CPU fallback for the NCT frontend should compose with the packed custom backward path."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(
            use_packed_qkv_conv_custom_backward=True
        ),
        candidate_overrides=dict(use_cuda_frontend_nct=True),
        grad_atol=3e-4,
        grad_rtol=5e-3,
        success_message="  ✓ CUDA frontend NCT composes with packed custom backward",
    )


def test_gdn_cuda_packed_conv_cpu_fallback_matches_packed_path() -> None:
    """CPU fallback for CUDA packed conv should preserve packed-path outputs and gradients."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(),
        candidate_overrides=dict(use_cuda_packed_conv=True),
        success_message="  ✓ CUDA packed conv CPU fallback matches packed path",
    )


def test_gdn_cuda_packed_conv_aten_bwd_cpu_fallback_matches_packed_path() -> None:
    """CPU fallback for CUDA packed-conv ATen-backward should preserve packed-path outputs and gradients."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(),
        candidate_overrides=dict(use_cuda_packed_conv_aten_backward=True),
        success_message="  ✓ CUDA packed conv ATen-backward CPU fallback matches packed path",
    )


def test_gdn_cuda_packed_conv_aten_weight_bwd_cpu_fallback_matches_packed_path() -> (
    None
):
    """CPU fallback for CUDA packed-conv ATen-weight-backward should preserve the packed path."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(),
        candidate_overrides=dict(use_cuda_packed_conv_aten_weight_backward=True),
        success_message="  ✓ CUDA packed conv ATen-weight-backward CPU fallback matches packed path",
    )


def test_gdn_cuda_packed_conv_validation() -> None:
    """CUDA packed conv should only run on the packed non-fused front-end path."""
    for message, kwargs in [
        (
            "Expected CUDA packed conv to require packed qkv proj+conv",
            dict(use_cuda_packed_conv=True),
        ),
        (
            "Expected CUDA packed conv ATen-weight-backward to reject ATen-backward",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_cuda_packed_conv_aten_weight_backward=True,
                use_cuda_packed_conv_aten_backward=True,
            ),
        ),
        (
            "Expected CUDA packed conv ATen-weight-backward to reject packed qkv custom backward",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_packed_qkv_conv_custom_backward=True,
                use_cuda_packed_conv_aten_weight_backward=True,
            ),
        ),
        (
            "Expected CUDA packed conv ATen-backward to reject packed qkv custom backward",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_packed_qkv_conv_custom_backward=True,
                use_cuda_packed_conv_aten_backward=True,
            ),
        ),
        (
            "Expected CUDA packed conv ATen-backward to reject full CUDA packed conv",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_cuda_packed_conv=True,
                use_cuda_packed_conv_aten_backward=True,
            ),
        ),
        (
            "Expected CUDA packed conv to reject packed qkv custom backward",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_packed_qkv_conv_custom_backward=True,
                use_cuda_packed_conv=True,
            ),
        ),
        (
            "Expected CUDA packed conv to reject CUDA split+l2norm",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_cuda_packed_conv=True,
                use_cuda_split_norm=True,
            ),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ CUDA packed conv validates requirements")


def test_gdn_cuda_frontend_nct_validation() -> None:
    """NCT frontend op should only run on the packed non-fused front-end path."""
    _assert_gdn_init_rejected(
        "Expected CUDA frontend NCT to require packed qkv proj+conv",
        use_cuda_frontend_nct=True,
    )
    layer = _make_small_gdn(
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        conv_output_contiguous=True,
        use_packed_qkv_conv_custom_backward=True,
        use_cuda_frontend_nct=True,
    )
    assert layer.use_packed_qkv_conv_custom_backward
    assert layer.use_cuda_frontend_nct
    _assert_gdn_init_rejected(
        "Expected CUDA frontend NCT to reject CUDA split+l2norm",
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        conv_output_contiguous=True,
        use_cuda_frontend_nct=True,
        use_cuda_split_norm=True,
    )
    print("  ✓ CUDA frontend NCT validates requirements")


def test_gdn_cuda_split_norm_validation() -> None:
    """CUDA split+l2norm should only run on the non-fused packed-projection path."""
    for message, kwargs in [
        (
            "Expected CUDA split+l2norm to require packed qkv proj+conv",
            dict(use_cuda_split_norm=True),
        ),
        (
            "Expected CUDA split+l2norm to reject the CUDA fused frontend",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_cuda_split_norm=True,
                use_cuda_fused_frontend=True,
            ),
        ),
        (
            "Expected CUDA split+l2norm to reject split-copy materialization",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_cuda_split_norm=True,
                use_packed_qkv_split_copy=True,
            ),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ CUDA split+l2norm validates requirements")


def test_packed_qkv_frontend_reference_matches_eager_ops():
    """Reference packed front-end should match the eager packed conv + q/k norm path."""
    torch.manual_seed(42)
    layer = GatedDeltaNet(
        d_model=64,
        n_heads=4,
        head_k_dim=8,
        expand_v=1.0,
        allow_neg_eigval=True,
        conv_size=4,
        use_fla=False,
        use_packed_qkv_conv=True,
        use_packed_qkv_proj=True,
        conv_output_contiguous=True,
    )
    x = torch.randn(2, 32, 64)
    qkv = layer.w_qkv.forward_packed(x)
    q_ref, k_ref, v_ref = packed_qkv_frontend_reference(
        qkv,
        layer.qkv_conv.conv.weight.view(layer.qkv_conv.conv.weight.shape[0], -1),
        n_heads=layer.n_heads,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
    )
    q_eager, k_eager, v_eager = layer.qkv_conv.forward_packed(qkv)
    q_eager = l2_norm(
        q_eager.view(x.size(0), x.size(1), layer.n_heads, layer.head_k_dim)
    )
    k_eager = l2_norm(
        k_eager.view(x.size(0), x.size(1), layer.n_heads, layer.head_k_dim)
    )
    v_eager = v_eager.view(x.size(0), x.size(1), layer.n_heads, layer.head_v_dim)
    torch.testing.assert_close(q_ref, q_eager, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(k_ref, k_eager, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(v_ref, v_eager, atol=1e-6, rtol=1e-6)
    print("  ✓ packed qkv frontend reference matches eager ops")


def test_rmsnorm_silu_gate_reference_matches_eager_ops():
    """Reference output fuse should match RMSNorm(o) * SiLU(gate)."""
    torch.manual_seed(42)
    o = torch.randn(2, 16, 4, 8, requires_grad=True)
    gate = torch.randn(2, 16, 4, 8, requires_grad=True)
    ref = rmsnorm_silu_gate_reference(o, gate, fp32_accum=True)
    eager = rms_norm(o.float()).to(o.dtype) * F.silu(gate)
    torch.testing.assert_close(ref, eager, atol=1e-6, rtol=1e-6)

    grad = torch.randn_like(ref)
    ref.backward(grad, retain_graph=True)
    ref_go, ref_gg = o.grad.clone(), gate.grad.clone()
    o.grad = None
    gate.grad = None
    eager.backward(grad)
    torch.testing.assert_close(ref_go, o.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ref_gg, gate.grad, atol=1e-6, rtol=1e-6)
    print("  ✓ rmsnorm*silu reference matches eager ops")


def test_gdn_cuda_fused_flags_validate_requirements() -> None:
    """Reject fused HGDN settings that the CUDA kernels do not implement."""
    for message, kwargs in [
        (
            "Expected fused frontend to require packed qkv proj+conv",
            dict(use_cuda_fused_frontend=True),
        ),
        (
            "Expected fused output to require output_norm_fp32",
            dict(use_cuda_fused_output=True, output_norm_fp32=False),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ fused HGDN flag validation OK")


def test_gdn_cuda_fused_cpu_fallback_matches_packed_path() -> None:
    """CPU fallback for fused HGDN paths should preserve outputs and gradients."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(output_norm_fp32=True),
        candidate_overrides=dict(
            use_cuda_fused_frontend=True,
            use_cuda_fused_output=True,
        ),
        grad_atol=3e-4,
        grad_rtol=5e-3,
        success_message="  ✓ fused HGDN CPU fallback matches packed path",
    )


def test_gdn_cuda_fused_frontend_lib_validation() -> None:
    """Compile-visible fused frontend should validate the intended family contract."""
    for message, kwargs in [
        (
            "Expected compile-visible fused frontend to require packed qkv proj+conv",
            dict(use_cuda_fused_frontend_lib=True),
        ),
        (
            "Expected compile-visible fused frontend to reject packed qkv custom backward",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_packed_qkv_conv_custom_backward=True,
                use_cuda_fused_frontend_lib=True,
            ),
        ),
        (
            "Expected compile-visible fused frontend to reject the old fused frontend path",
            dict(
                use_packed_qkv_conv=True,
                use_packed_qkv_proj=True,
                conv_output_contiguous=True,
                use_cuda_fused_frontend=True,
                use_cuda_fused_frontend_lib=True,
            ),
        ),
    ]:
        _assert_gdn_init_rejected(message, **kwargs)
    print("  ✓ compile-visible fused frontend validates requirements")


def test_gdn_cuda_fused_frontend_lib_cpu_fallback_matches_packed_path() -> None:
    """Compile-visible fused frontend CPU fallback should preserve packed-path outputs and gradients."""
    _assert_gdn_cpu_fallback_matches_reference(
        base_kwargs=_make_standard_packed_kwargs(output_norm_fp32=True),
        candidate_overrides=dict(use_cuda_fused_frontend_lib=True),
        grad_atol=3e-4,
        grad_rtol=5e-3,
        success_message="  ✓ compile-visible fused frontend CPU fallback matches packed path",
    )


def test_gdn_conv_output_contiguous():
    """Contiguous conv outputs should produce contiguous recurrence inputs."""
    layer = _make_bf16_gdn(conv_output_contiguous=True)
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    q, k, v, g, beta = layer._project_recurrence_inputs(x)
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert g.is_contiguous()
    assert beta.is_contiguous()
    print("  ✓ GDN contiguous recurrence inputs OK")


def test_gdn_qk_only_contiguous():
    """Allow selecting contiguous conv outputs per HGDN path."""
    layer = _make_bf16_gdn(
        conv_output_contiguous=False,
        q_conv_output_contiguous=True,
        k_conv_output_contiguous=True,
        v_conv_output_contiguous=False,
    )
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    q, k, v, _, _ = layer._project_recurrence_inputs(x)
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert not v.is_contiguous()
    print("  ✓ GDN q/k-only contiguous outputs OK")


def test_gdn_gate_and_output_precision_toggles():
    """Precision toggles should preserve HGDN forward/backward viability."""
    layer = _make_bf16_gdn(
        gates_fp32=False,
        output_norm_fp32=False,
    )
    x = torch.randn(2, 16, 64, dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    assert out.dtype == x.dtype
    out.float().sum().backward()
    assert x.grad is not None
    print("  ✓ GDN precision toggles OK")


def test_restore_low_dim_params_to_fp32_gdn_control_override():
    """Allow HGDN control projections to remain bf16 while scalars stay fp32."""
    layer = GatedDeltaNet(
        64, n_heads=4, head_k_dim=8, expand_v=2.0, use_fla=False
    ).bfloat16()
    restore_low_dim_params_to_fp32(layer, gdn_control_proj_fp32=False)
    assert layer.w_a.weight.dtype == torch.bfloat16
    assert layer.w_b.weight.dtype == torch.bfloat16
    assert layer.w_g.weight.dtype == torch.bfloat16
    assert layer.A_log.dtype == torch.float32
    assert layer.dt_bias.dtype == torch.float32
    print("  ✓ GDN control projection fp32 override OK")


def test_restore_low_dim_params_to_fp32_default_restores_gdn_control():
    """Keep the default fp32 restore behavior for HGDN control projections."""
    layer = GatedDeltaNet(
        64, n_heads=4, head_k_dim=8, expand_v=2.0, use_fla=False
    ).bfloat16()
    restore_low_dim_params_to_fp32(layer)
    assert layer.w_a.weight.dtype == torch.float32
    assert layer.w_b.weight.dtype == torch.float32
    assert layer.w_g.weight.dtype == torch.float32
    print("  ✓ default GDN control projection fp32 restore OK")


def test_hybrid_fwd_bwd():
    torch.manual_seed(42)
    m = HybridGPT(
        vocab_size=32,
        num_layers=4,
        d_model=64,
        attn_heads=4,
        attn_kv_heads=2,
        gdn_n_heads=4,
        gdn_head_k_dim=8,
        gdn_expand_v=2.0,
        gdn_ratio=3,
        mlp_mult=2,
    )
    loss = m(torch.randint(0, 32, (2, 24)), torch.randint(0, 32, (2, 24)))
    assert not torch.isnan(loss)
    loss.backward()
    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    nan_grads = [
        n
        for n, p in m.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not no_grad, f"Missing grads: {no_grad}"
    assert not nan_grads, f"NaN grads: {nan_grads}"
    print(f"  ✓ fwd/bwd OK  loss={loss.item():.4f}")


def test_norm_styles():
    """Exercise pre/post/KEEL block variants on a tiny hybrid stack."""
    torch.manual_seed(42)
    ids = torch.randint(0, 32, (2, 24))
    tgt = torch.randint(0, 32, (2, 24))
    for style in ["pre", "post", "keel"]:
        model = HybridGPT(
            vocab_size=32,
            num_layers=4,
            d_model=64,
            attn_heads=4,
            attn_kv_heads=2,
            gdn_n_heads=4,
            gdn_head_k_dim=8,
            gdn_expand_v=1.0,
            gdn_ratio=1,
            mlp_mult=2,
            norm_style=style,
        )
        loss = model(ids, tgt)
        assert not torch.isnan(loss), f"{style} loss is NaN"
        loss.backward()
        assert model.norm_style == style
        if style == "keel":
            assert model.residual_alpha == 8.0
        no_grad = [
            name for name, param in model.named_parameters() if param.grad is None
        ]
        assert not no_grad, f"{style} missing grads: {no_grad}"
    print("  ✓ norm styles OK (pre/post/keel)")


def test_invalid_norm_style():
    """Reject unsupported normalization styles."""
    try:
        validate_norm_style("weird")
    except ValueError:
        print("  ✓ invalid norm style rejected")
        return
    raise AssertionError("validate_norm_style should reject unknown styles")


def test_wandb_watch_mode():
    """Normalize supported W&B watch modes and reject unknown values."""
    assert normalize_wandb_watch_mode("none") == "none"
    assert normalize_wandb_watch_mode("false") == "none"
    assert normalize_wandb_watch_mode("gradients") == "gradients"
    assert normalize_wandb_watch_mode("all") == "all"
    try:
        normalize_wandb_watch_mode("weird")
    except ValueError:
        print("  ✓ wandb watch mode rejected invalid value")
        return
    raise AssertionError("normalize_wandb_watch_mode should reject unknown values")


def test_block_types():
    for ratio, expected in [
        (0, ["attn"] * 4),
        (1, ["gdn", "attn"] * 2),
        (3, ["gdn", "gdn", "gdn", "attn"]),
        (100, ["gdn"] * 4),
    ]:
        m = HybridGPT(
            vocab_size=16,
            num_layers=4,
            d_model=32,
            attn_heads=2,
            attn_kv_heads=1,
            gdn_n_heads=2,
            gdn_head_k_dim=8,
            gdn_expand_v=1.0,
            gdn_ratio=ratio,
            mlp_mult=2,
        )
        assert m.block_types == expected, f"ratio={ratio}: {m.block_types}"
    print("  ✓ block types OK")


def test_artifact_audit():
    """Verify the real quantized artifact audit behaves monotonically."""
    for label, m in [
        ("hybrid_tight  (8h Dk48 Dv48 mlp3.0)", make_hybrid_tight()),
        ("hybrid_wide   (4h Dk48 Dv96 mlp3.25)", make_hybrid_wide()),
        ("baseline_fill (11L×512d mlp2.75)", make_baseline_fill()),
        ("attention_only_baseline (16L×384d mlp3.75)", make_attention_only_baseline()),
    ]:
        total = sum(p.numel() for p in m.parameters())
        _, _, audit = serialize_quantized_state_dict_int8(m.state_dict())
        ng = sum(1 for t in m.block_types if t == "gdn")
        na = sum(1 for t in m.block_types if t == "attn")
        assert audit["baseline_tensor_bytes"] >= audit["int8_payload_bytes"]
        assert audit["quant_raw_torch_bytes"] >= audit["int8_payload_bytes"]
        assert audit["quant_raw_torch_bytes"] >= audit["quant_zlib_bytes"]
        print(
            f"  ✓ {label}: {total:,}p ({ng}G+{na}A) "
            f"payload={audit['int8_payload_bytes'] / 1e6:.2f}MB "
            f"raw_torch={audit['quant_raw_torch_bytes'] / 1e6:.2f}MB "
            f"int8_zlib_init={audit['quant_zlib_bytes'] / 1e6:.2f}MB"
        )
    print("  ✓ artifact audit OK")


def test_state_tracking():
    V, B, T = 32, 4, 48

    def make_data(B, T, V, seed):
        torch.manual_seed(seed)
        ids_l, tgt_l = [], []
        for _ in range(B):
            regs = [1, 2, 3]
            seq = list(regs)
            tgts = [0, 0, 0]
            for _ in range(T - 4):
                op = torch.randint(4, 7, (1,)).item()
                seq.append(op)
                if op == 4:
                    regs[0], regs[1] = regs[1], regs[0]
                elif op == 5:
                    regs[1], regs[2] = regs[2], regs[1]
                else:
                    regs[0], regs[2] = regs[2], regs[0]
                tgts.append(0)
            seq.append(7)
            tgts.append(regs[0])
            while len(seq) < T:
                seq.append(0)
                tgts.append(0)
            ids_l.append(seq[:T])
            tgt_l.append(tgts[:T])
        return torch.tensor(ids_l), torch.tensor(tgt_l)

    base = dict(
        vocab_size=V,
        num_layers=4,
        d_model=64,
        attn_heads=4,
        attn_kv_heads=2,
        gdn_n_heads=4,
        gdn_head_k_dim=8,
        gdn_expand_v=2.0,
        mlp_mult=2,
    )
    results = {}
    for name, ratio in [("hybrid", 3), ("attn", 0)]:
        seed_losses = []
        for seed in range(3):
            torch.manual_seed(seed * 1000 + 42)
            m = HybridGPT(**base, gdn_ratio=ratio)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for step in range(50):
                ids, tgt = make_data(B, T, V, seed=step + seed * 50)
                loss = m(ids, tgt)
                loss.backward()
                opt.step()
                opt.zero_grad()
            seed_losses.append(loss.item())
        results[name] = sum(seed_losses) / len(seed_losses)
    print(f"  hybrid={results['hybrid']:.4f}  attn={results['attn']:.4f}")
    assert results["hybrid"] < results["attn"], "Hybrid should beat attn"
    print("  ✓ hybrid beats attn on state tracking")


def test_convergence():
    torch.manual_seed(42)
    m = HybridGPT(
        vocab_size=32,
        num_layers=4,
        d_model=64,
        attn_heads=4,
        attn_kv_heads=2,
        gdn_n_heads=4,
        gdn_head_k_dim=8,
        gdn_expand_v=2.0,
        gdn_ratio=3,
        mlp_mult=2,
    )
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    ids = (torch.arange(32) % 32).unsqueeze(0).expand(4, -1)
    tgt = torch.cat([ids[:, 1:], ids[:, :1]], dim=1)
    losses = []
    for _ in range(30):
        loss = m(ids, tgt)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"{losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"  ✓ converges: {losses[0]:.3f} → {losses[-1]:.3f}")


def test_profile_report_json_roundtrip():
    """Write and reload a structured profiler report without txt artifacts."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        x = torch.randn(8, 8)
        _ = x @ x
    rows = build_profile_rows(prof.key_averages())
    report = build_profile_report(rows, sort_by="self_cpu_time_total")
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)
        write_profile_report(
            outpath,
            report=report,
            stem="sample",
        )
        loaded = load_profile_report(outpath / "sample.json")
        assert loaded["metadata"]["row_count"] > 0
        assert (outpath / "sample.json").is_file()
        assert (outpath / "sample.csv").is_file()
        assert not (outpath / "sample.txt").exists()
    print("  ✓ profiler report json/csv roundtrip OK")


def test_profile_report_coalesces_duplicate_names():
    """Merge duplicate profiler names instead of keeping zero-time duplicates."""
    rows = [
        ProfileRow(
            name="gdn.g_pointwise",
            count=8,
            self_cpu_time_us=0.0,
            cpu_time_total_us=0.0,
            self_device_time_us=27_000.0,
            device_time_total_us=27_000.0,
            cpu_memory_bytes=0,
            self_cpu_memory_bytes=0,
            device_memory_bytes=0,
            self_device_memory_bytes=0,
        ),
        ProfileRow(
            name="gdn.g_pointwise",
            count=8,
            self_cpu_time_us=0.0,
            cpu_time_total_us=0.0,
            self_device_time_us=0.0,
            device_time_total_us=0.0,
            cpu_memory_bytes=0,
            self_cpu_memory_bytes=0,
            device_memory_bytes=0,
            self_device_memory_bytes=0,
        ),
    ]
    report = build_profile_report(rows, sort_by="self_device_time_total")
    assert report["metadata"]["row_count"] == 1
    assert report["rows"][0]["name"] == "gdn.g_pointwise"
    assert report["rows"][0]["count"] == 8
    assert report["rows"][0]["self_device_time_us"] == 27_000.0
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = Path(tmpdir)
        write_profile_report(outpath, report=report, stem="sample")
        loaded = load_profile_report(outpath / "sample.json")
        assert loaded["metadata"]["row_count"] == 1
        assert loaded["rows"][0]["self_device_time_us"] == 27_000.0
    print("  ✓ profiler report coalesces duplicate names")


if __name__ == "__main__":
    print("=" * 60)
    print(f"GDN Hybrid v3 Tests — PyTorch {torch.__version__}")
    print("=" * 60)
    tests = [
        ("Recurrence", test_recurrence),
        ("Split projections", test_gdn_split_projections),
        ("Recurrence input dtypes", test_gdn_recurrence_input_dtypes),
        ("Muon routing", test_muon_routing),
        ("Causal conv", test_causal_conv),
        ("Causal conv disable", test_causal_conv_disable),
        ("GDN conv toggles", test_gdn_conv_toggles),
        ("Packed qkv conv parity", test_gdn_packed_qkv_conv_matches_separate_path),
        (
            "Packed qkv conv validation",
            test_gdn_packed_qkv_conv_requires_aligned_settings,
        ),
        (
            "Packed qkv projection+conv parity",
            test_gdn_packed_qkv_proj_conv_matches_separate_path,
        ),
        (
            "Packed qkv custom backward parity",
            test_gdn_packed_qkv_custom_backward_matches_default_path,
        ),
        (
            "Packed qkv custom backward validation",
            test_gdn_packed_qkv_custom_backward_validation,
        ),
        (
            "Packed qkv single-contig parity",
            test_gdn_packed_qkv_single_contig_matches_default_path,
        ),
        (
            "Packed qkv single-contig validation",
            test_gdn_packed_qkv_single_contig_validation,
        ),
        (
            "Packed qkv split-copy parity",
            test_gdn_packed_qkv_split_copy_matches_default_path,
        ),
        (
            "Packed qkv split-copy validation",
            test_gdn_packed_qkv_split_copy_validation,
        ),
        (
            "Packed split+l2norm reference parity",
            test_packed_qkv_split_l2norm_reference_matches_eager_ops,
        ),
        (
            "NCT preact frontend reference parity",
            test_preact_silu_split_l2norm_nct_reference_matches_eager_ops,
        ),
        (
            "Packed conv reference parity",
            test_packed_qkv_conv_reference_matches_eager_ops,
        ),
        (
            "Packed qkv frontend reference parity",
            test_packed_qkv_frontend_reference_matches_eager_ops,
        ),
        (
            "RMSNorm*SiLU reference parity",
            test_rmsnorm_silu_gate_reference_matches_eager_ops,
        ),
        (
            "Fused HGDN flag validation",
            test_gdn_cuda_fused_flags_validate_requirements,
        ),
        (
            "Fused HGDN CPU fallback parity",
            test_gdn_cuda_fused_cpu_fallback_matches_packed_path,
        ),
        (
            "Compile-visible fused frontend validation",
            test_gdn_cuda_fused_frontend_lib_validation,
        ),
        (
            "Compile-visible fused frontend CPU fallback parity",
            test_gdn_cuda_fused_frontend_lib_cpu_fallback_matches_packed_path,
        ),
        (
            "CUDA split+l2norm CPU fallback parity",
            test_gdn_cuda_split_norm_cpu_fallback_matches_packed_path,
        ),
        (
            "CUDA frontend NCT CPU fallback parity",
            test_gdn_cuda_frontend_nct_cpu_fallback_matches_packed_path,
        ),
        (
            "CUDA split+l2norm validation",
            test_gdn_cuda_split_norm_validation,
        ),
        (
            "CUDA frontend NCT validation",
            test_gdn_cuda_frontend_nct_validation,
        ),
        (
            "CUDA packed conv CPU fallback parity",
            test_gdn_cuda_packed_conv_cpu_fallback_matches_packed_path,
        ),
        (
            "CUDA packed conv ATen-backward CPU fallback parity",
            test_gdn_cuda_packed_conv_aten_bwd_cpu_fallback_matches_packed_path,
        ),
        (
            "CUDA packed conv ATen-weight-backward CPU fallback parity",
            test_gdn_cuda_packed_conv_aten_weight_bwd_cpu_fallback_matches_packed_path,
        ),
        (
            "CUDA packed conv validation",
            test_gdn_cuda_packed_conv_validation,
        ),
        ("Hybrid fwd/bwd", test_hybrid_fwd_bwd),
        ("Norm styles", test_norm_styles),
        ("Invalid norm style", test_invalid_norm_style),
        ("W&B watch mode", test_wandb_watch_mode),
        ("Block types", test_block_types),
        ("Artifact audit (all presets)", test_artifact_audit),
        ("State tracking", test_state_tracking),
        ("Convergence", test_convergence),
        ("Profiler report json/csv roundtrip", test_profile_report_json_roundtrip),
        (
            "Profiler report duplicate coalescing",
            test_profile_report_coalesces_duplicate_names,
        ),
    ]
    for i, (name, fn) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {name}")
        fn()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
