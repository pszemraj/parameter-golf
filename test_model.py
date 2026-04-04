"""test_model.py — v3 tests. Run: python test_model.py"""

import tempfile
from pathlib import Path

import torch

from model import (
    SCALAR_PARAM_PATTERNS,
    CausalConv1d,
    GatedDeltaNet,
    HybridGPT,
    gdn_recurrent_naive,
    l2_norm,
    make_baseline_fill,
    make_depth_control,
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


def _make_bf16_gdn(**kwargs) -> GatedDeltaNet:
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


def test_gdn_packed_qkv_conv_requires_aligned_settings():
    """Packed q/k/v conv should reject mixed enablement or layout settings."""
    try:
        GatedDeltaNet(
            d_model=64,
            n_heads=4,
            head_k_dim=8,
            expand_v=1.0,
            use_packed_qkv_conv=True,
            use_v_conv=False,
        )
        raise AssertionError("Expected packed qkv conv to reject disabled v conv")
    except ValueError:
        pass
    try:
        GatedDeltaNet(
            d_model=64,
            n_heads=4,
            head_k_dim=8,
            expand_v=1.0,
            use_packed_qkv_conv=True,
            q_conv_output_contiguous=True,
            k_conv_output_contiguous=False,
            v_conv_output_contiguous=True,
        )
        raise AssertionError(
            "Expected packed qkv conv to reject mixed output_contiguous flags"
        )
    except ValueError:
        pass
    try:
        GatedDeltaNet(
            d_model=64,
            n_heads=4,
            head_k_dim=8,
            expand_v=1.0,
            use_packed_qkv_proj=True,
        )
        raise AssertionError(
            "Expected packed qkv projection to require packed qkv conv"
        )
    except ValueError:
        pass
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
        ("depth_control (16L×384d mlp3.75)", make_depth_control()),
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
