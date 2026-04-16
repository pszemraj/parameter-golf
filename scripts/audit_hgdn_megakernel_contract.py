#!/usr/bin/env python3
"""Static contract audit for the HGDN megakernel path.

Run from the repository root:

    python scripts/audit_hgdn_megakernel_contract.py

This is intentionally conservative. It does not prove numerical correctness; it
checks for the failure modes that make a claimed megakernel experiment
misleading: silent fallback, hidden copy/cast kernels, missing sidecar exclusion,
and unsafe CUDA build flags for recurrent parity.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()
MODEL = ROOT / "model.py"
HGDN_UTILS = ROOT / "hgdn_runtime_utils.py"
BINDING = ROOT / "hgdn_megakernel" / "hgdn_megakernel_binding.py"
CUDA = ROOT / "hgdn_megakernel" / "hgdn_megakernel.cu"
SETUP = ROOT / "setup_hgdn_megakernel.py"
TRAINER = ROOT / "train_gpt_hybrid.py"
TEST_HARNESS = ROOT / "hgdn_megakernel" / "test_megakernel.py"


def read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def check(name: str, ok: bool, detail: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def main() -> None:
    model = read(MODEL)
    hgdn_utils = read(HGDN_UTILS)
    binding = read(BINDING)
    cuda = read(CUDA)
    setup = read(SETUP)
    trainer = read(TRAINER)
    test_harness = read(TEST_HARNESS)

    failures = 0

    failures += not check(
        "megakernel flag exists",
        "use_cuda_megakernel" in model and "GDN_USE_CUDA_MEGAKERNEL" in trainer,
        "model/trainer expose the megakernel runtime flag",
    )

    sidecar_guard = (
        "use_cuda_megakernel is incompatible with the existing HGDN sidecar CUDA paths"
        in model
    )
    failures += not check(
        "sidecar exclusion",
        sidecar_guard,
        "megakernel mode should reject old frontend/output/conv sidecar modes",
    )

    silent_fallback_pattern = re.compile(
        r"self\.use_cuda_megakernel[\s\S]{0,250}?megakernel_extension_loaded\(\)[\s\S]{0,80}?return run_from_gated_delta_net",
        re.MULTILINE,
    )
    fallback_present = bool(silent_fallback_pattern.search(model))
    explicit_raise_near_forward = (
        "HGDN megakernel" in model
        and "extension" in model
        and "RuntimeError"
        in model[
            model.find("def forward(self, x: Tensor)") : model.find(
                "# ── Standard attention"
            )
        ]
    )
    failures += not check(
        "no silent fallback",
        not fallback_present or explicit_raise_near_forward,
        "if GDN_USE_CUDA_MEGAKERNEL=1 and extension is unavailable, fail before benchmarking",
    )

    dtype_guard = bool(
        re.search(
            r"self\.use_cuda_megakernel\s+and\s+x\.is_cuda[\s\S]{0,200}x\.dtype\s*!=\s*torch\.bfloat16[\s\S]{0,160}RuntimeError",
            model,
        )
    )
    failures += not check(
        "cuda bf16 megakernel dtype guard",
        dtype_guard,
        "megakernel mode should hard-fail on CUDA activations that are not bf16",
    )

    hidden_forward_copy = any(
        token in binding
        for token in (
            "x.contiguous()",
            "w_qkv.contiguous()",
            "w_a.contiguous()",
            "w_b.contiguous()",
            "w_g.contiguous()",
            "w_out.contiguous()",
            "conv_w.contiguous()",
        )
    )
    hidden_backward_copy = "grad_y.contiguous().to(dtype=torch.bfloat16)" in binding
    failures += not check(
        "no hidden copy/cast in autograd wrapper",
        not hidden_forward_copy and not hidden_backward_copy,
        "wrapper should validate contiguity/dtype instead of calling .contiguous()/.to() inside measured path",
    )

    failures += not check(
        "no fast-math in parity build",
        "--use_fast_math" not in setup and "--use_fast_math" not in binding,
        "remove --use_fast_math until long recurrent parity is clean",
    )

    failures += not check(
        "cooperative one-launch kernels present",
        "cudaLaunchCooperativeKernel" in cuda
        and "hgdn_forward_bf16_kernel" in cuda
        and "hgdn_backward_bf16_kernel" in cuda,
        "forward/backward should be one cooperative CUDA kernel each",
    )

    failures += not check(
        "dense W_out inside kernel",
        "phase_gemm_abt_store_bf16(z, w_out, y" in cuda
        or "w_out" in cuda
        and "dense cross-head W_out" in cuda,
        "full-block path must handle dense cross-head output projection",
    )

    control_guard = (
        "gdn_use_cuda_megakernel" in trainer
        and "gdn_control_proj_fp32" in trainer
        and re.search(
            r"gdn_use_cuda_megakernel[\s\S]{0,300}gdn_control_proj_fp32[\s\S]{0,120}(raise|parser\.error|SystemExit)",
            trainer,
        )
    )
    failures += not check(
        "control projection dtype preflight",
        bool(control_guard),
        "trainer should reject GDN_USE_CUDA_MEGAKERNEL=1 with GDN_CONTROL_PROJ_FP32=1",
    )

    status_guard = (
        "hgdn_megakernel_preflight" in trainer
        and "hgdn_megakernel_extension_status" in trainer
        and re.search(
            r"gdn_use_cuda_megakernel[\s\S]{0,600}hgdn_megakernel_preflight[\s\S]{0,600}loaded",
            trainer,
        )
    )
    failures += not check(
        "trainer extension preflight",
        bool(status_guard),
        "trainer should log extension status and fail before training when megakernel mode is unavailable",
    )

    compile_guard = bool(
        re.search(
            r"use_megakernel\s*=\s*bool\([\s\S]{0,200}use_cuda_megakernel[\s\S]{0,400}if not use_megakernel:[\s\S]{0,300}maybe_disable_compile[\s\S]{0,300}else:[\s\S]{0,200}gdn_megakernel_left_enabled",
            hgdn_utils,
        )
    )
    failures += not check(
        "megakernel compile integration",
        compile_guard,
        "prepare_hybrid_compile should leave megakernel GDN blocks compile-eligible instead of forcing eager disable",
    )

    medium_parity = all(
        token in test_harness
        for token in (
            "B1_T128",
            "B1_T512",
            'f"{reference_name}/forward_y"',
            'f"{reference_name}/{name}"',
        )
    )
    failures += not check(
        "medium parity coverage",
        medium_parity,
        "test harness should cover B=1,T=128 and B=1,T=512 against named references",
    )

    optional_b2 = "--include-b2-t512" in test_harness and "B2_T512" in test_harness
    failures += not check(
        "optional B2,T512 coverage",
        optional_b2,
        "test harness should expose an optional B=2,T=512 parity case",
    )

    t2048_flag = "--include-b1-t2048" in test_harness and "B1_T2048" in test_harness
    failures += not check(
        "optional B1,T2048 coverage",
        t2048_flag,
        "test harness should expose an optional B=1,T=2048 parity case for the H100 gate",
    )

    print()
    if failures:
        print(f"HGDN megakernel audit failed with {failures} issue(s).")
        raise SystemExit(1)
    print("HGDN megakernel static audit passed.")


if __name__ == "__main__":
    main()
