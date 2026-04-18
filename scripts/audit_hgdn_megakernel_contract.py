#!/usr/bin/env python3
"""Static contract audit for the HGDN megakernel path.

Run from the repository root:

    python scripts/audit_hgdn_megakernel_contract.py

This is intentionally conservative. It does not prove numerical correctness; it
checks for the failure modes that make a claimed megakernel experiment
misleading: silent fallback, hidden copy/cast kernels, missing sidecar exclusion,
compile-visible regressions, and unsafe CUDA build flags for recurrent parity.
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
TEST_CORE_HARNESS = ROOT / "hgdn_megakernel" / "test_corekernel.py"
CORE_HELPER = ROOT / "scripts" / "run_h100_single_gpu_hgdn_corekernel.sh"


def read(path: Path) -> str:
    """Read one repository file as UTF-8 text.

    :param Path path: File path to inspect.
    :return str: File contents, or an empty string when the file is absent.
    """
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def check(name: str, ok: bool, detail: str) -> bool:
    """Print one audit check result and return the boolean outcome.

    :param str name: Human-readable check label.
    :param bool ok: Whether the check passed.
    :param str detail: Short detail string describing the expectation.
    :return bool: The original `ok` value.
    """
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def main() -> None:
    """Run the static HGDN kernel-contract audit from the repo root."""
    model = read(MODEL)
    hgdn_utils = read(HGDN_UTILS)
    binding = read(BINDING)
    cuda = read(CUDA)
    setup = read(SETUP)
    trainer = read(TRAINER)
    test_harness = read(TEST_HARNESS)
    test_core_harness = read(TEST_CORE_HARNESS)
    core_helper = read(CORE_HELPER)

    failures = 0

    failures += not check(
        "megakernel flag exists",
        "use_cuda_megakernel" in model and "GDN_USE_CUDA_MEGAKERNEL" in trainer,
        "model/trainer expose the megakernel runtime flag",
    )

    failures += not check(
        "corekernel flag exists",
        "use_cuda_corekernel" in model and "GDN_USE_CUDA_COREKERNEL" in trainer,
        "model/trainer expose the core-kernel runtime flag",
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

    core_dtype_guard = bool(
        re.search(
            r"self\.use_cuda_corekernel\s+and\s+x\.is_cuda[\s\S]{0,200}x\.dtype\s*!=\s*torch\.bfloat16[\s\S]{0,160}RuntimeError",
            model,
        )
    )
    failures += not check(
        "cuda bf16 corekernel dtype guard",
        core_dtype_guard,
        "core-kernel mode should hard-fail on CUDA activations that are not bf16",
    )

    core_forward_start = model.find("if self.use_cuda_corekernel and x.is_cuda:")
    core_forward_end = model.find("if self.use_cuda_megakernel and x.is_cuda:")
    core_forward_block = (
        model[core_forward_start:core_forward_end]
        if core_forward_start != -1 and core_forward_end != -1
        else ""
    )
    core_boundary = all(
        token in core_forward_block
        for token in (
            "qkv = self.w_qkv.forward_packed(x)",
            "g_pre = self.w_a(x)",
            "beta_pre = self.w_b(x)",
            "g_out = self.w_g(x).view(B, T, H, Dv)",
            "conv_w = self.qkv_conv.conv.weight.flatten(1)",
            "torch.ops.hgdn_corekernel_v1.run(",
            "int(self.conv_size)",
            "int(self.corekernel_rec_chunk_t)",
            "return self.w_out(z.reshape(B, T, -1))",
        )
    )
    failures += not check(
        "corekernel dense-outside boundary",
        core_boundary,
        "core-kernel mode should keep W_qkv/W_a/W_b/W_g/W_out outside the owned CUDA op",
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

    custom_op_contract = (
        all(
            token in binding
            for token in (
                'torch.library.Library("hgdn_megakernel_v1", "DEF")',
                '@torch.library.register_fake("hgdn_megakernel_v1::run")',
                "torch.library.register_autograd(",
                "ctx.set_materialize_grads(False)",
            )
        )
        and "torch.autograd.Function" not in binding
    )
    failures += not check(
        "compile-visible custom-op boundary",
        custom_op_contract,
        "binding should expose the megakernel through a torch.library op instead of an eager-only autograd.Function island",
    )

    core_custom_op_contract = all(
        token in binding
        for token in (
            'torch.library.Library("hgdn_corekernel_v1", "DEF")',
            '@torch.library.register_fake("hgdn_corekernel_v1::run")',
            'torch.library.register_autograd(\n    "hgdn_corekernel_v1::run"',
        )
    )
    failures += not check(
        "compile-visible core custom-op boundary",
        core_custom_op_contract,
        "binding should expose the HGDN core kernel through a torch.library op",
    )

    core_binding_contract = (
        "def hgdn_corekernel(" in binding
        and "qkv: Tensor," in binding
        and "g_pre: Tensor," in binding
        and "beta_pre: Tensor," in binding
        and "g_out: Tensor," in binding
        and "conv_w: Tensor," in binding
        and "A_log: Tensor," in binding
        and "dt_bias: Tensor," in binding
        and "w_qkv"
        not in binding[
            binding.find("def hgdn_corekernel(") : binding.find(
                "def run_core_from_gated_delta_net("
            )
        ]
        and "w_out"
        not in binding[
            binding.find("def hgdn_corekernel(") : binding.find(
                "def run_core_from_gated_delta_net("
            )
        ]
    )
    failures += not check(
        "corekernel forward input contract",
        core_binding_contract,
        "core-kernel binding should consume projected tensors plus conv/gate state, not dense weights",
    )

    binding_output_contract = (
        "Tensor y, Tensor qkv, Tensor g_pre, Tensor beta_pre, Tensor g_log," in binding
        and "Tensor beta, Tensor g_out, Tensor o_raw, Tensor state_ckpt)" in binding
        and "Tensor qkv, Tensor pre," not in binding
        and "Tensor q_norm" not in binding
        and "Tensor k_norm" not in binding
        and "Tensor v_post" not in binding
        and "Tensor inv_q" not in binding
        and "Tensor inv_k" not in binding
        and "Tensor o_norm" not in binding
        and "Tensor z, Tensor state_ckpt" not in binding
    )
    failures += not check(
        "binding forward saved-tensor contract",
        binding_output_contract,
        "forward binding should save g_out/o_raw/state_ckpt and should not still expose saved o_norm/z tensors",
    )

    binding_backward_contract = (
        '"Tensor g_pre, Tensor beta_pre, Tensor g_log, Tensor beta, Tensor g_out, "'
        in binding
        and '"Tensor o_raw, Tensor state_ckpt, int n_heads, "' in binding
        and "Tensor qkv, Tensor pre," not in binding
        and "Tensor q_norm" not in binding
        and "Tensor k_norm" not in binding
        and "Tensor v_post" not in binding
        and "Tensor inv_q" not in binding
        and "Tensor inv_k" not in binding
        and "ctx.save_for_backward(" in binding
        and "o_norm" not in binding
    )
    failures += not check(
        "binding backward saved-tensor contract",
        binding_backward_contract,
        "backward binding should consume the reduced saved-tensor set without saved o_norm/z inputs",
    )

    failures += not check(
        "no fast-math in parity build",
        "--use_fast_math" not in setup and "--use_fast_math" not in binding,
        "remove --use_fast_math until long recurrent parity is clean",
    )

    tile_knobs = all(
        token in setup
        for token in (
            'os.environ.get("HGDN_THREADS")',
            'os.environ.get("HGDN_GEMM_ATB_SPLIT_M_THRESHOLD")',
            'os.environ.get("HGDN_REC_V_TILE")',
            'os.environ.get("HGDN_REC_CHUNK_T")',
            "-DHGDN_THREADS=",
            "-DHGDN_GEMM_ATB_SPLIT_M_THRESHOLD=",
            "-DHGDN_REC_V_TILE=",
            "-DHGDN_REC_CHUNK_T=",
        )
    ) and all(
        token in cuda
        for token in (
            "#ifndef HGDN_THREADS",
            "#ifndef HGDN_GEMM_ATB_SPLIT_M_THRESHOLD",
            "#ifndef HGDN_REC_V_TILE",
            "#ifndef HGDN_REC_CHUNK_T",
            "constexpr int THREADS = HGDN_THREADS;",
            "HGDN_GEMM_ATB_SPLIT_M_THRESHOLD",
            "constexpr int REC_V_TILE = HGDN_REC_V_TILE;",
            "constexpr int REC_CHUNK_T_MAX = HGDN_REC_CHUNK_T;",
        )
    )
    failures += not check(
        "recurrence tile build knobs",
        tile_knobs,
        "setup and CUDA sources should expose THREADS/GEMM split threshold/REC_V_TILE/REC_CHUNK_T as explicit compile-time tuning knobs",
    )

    runtime_chunk_knob = (
        "int rec_chunk_t" in binding
        and "GDN_MEGAKERNEL_REC_CHUNK_T" in binding
        and "rec_chunk_t_max" in binding
        and "int rec_chunk_t" in cuda
        and "REC_CHUNK_T_MAX" in cuda
        and "GDN_MEGAKERNEL_REC_CHUNK_T" in trainer
    )
    failures += not check(
        "runtime chunk cadence control",
        runtime_chunk_knob,
        "megakernel path should accept a runtime rec_chunk_t bounded by the compiled REC_CHUNK_T max and preflight it in the trainer",
    )

    core_chunk_alias = (
        "def resolve_runtime_rec_chunk_t(" in binding
        and "prefer_corekernel=True" in binding
        and "GDN_COREKERNEL_REC_CHUNK_T" in binding
        and "hgdn_resolve_runtime_rec_chunk_t" in trainer
        and "GDN_COREKERNEL_REC_CHUNK_T" in trainer
        and "GDN_COREKERNEL_REC_CHUNK_T" in core_helper
    )
    failures += not check(
        "corekernel runtime chunk alias",
        core_chunk_alias,
        "core-kernel path should prefer GDN_COREKERNEL_REC_CHUNK_T while accepting the legacy megakernel fallback",
    )

    tiled_dot_helper = all(
        token in cuda
        for token in (
            "block_dot_cols_slice8",
            "block_dot_cols_tiled",
            "REC_DOT_COL_SLICE",
        )
    )
    failures += not check(
        "recurrence dot helper supports tiled columns",
        tiled_dot_helper,
        "the recurrence dot helper should support wider REC_V_TILE sweeps without changing model math",
    )

    failures += not check(
        "cooperative one-launch kernels present",
        "cudaLaunchCooperativeKernel" in cuda
        and "hgdn_forward_bf16_kernel" in cuda
        and "hgdn_backward_bf16_kernel" in cuda,
        "forward/backward should be one cooperative CUDA kernel each",
    )

    dense_w_out = (
        bool(re.search(r"phase_gemm_abt_store_bf16\(\s*z_tmp,\s*w_out,\s*y", cuda))
        and bool(
            re.search(
                r"phase_gemm_aT_b_store_bf16\(\s*grad_y,\s*grad_z,\s*grad_w_out",
                cuda,
            )
        )
        and bool(
            re.search(r"phase_gemm_ab_store_bf16\(\s*grad_y,\s*w_out,\s*grad_z", cuda)
        )
    )
    failures += not check(
        "dense W_out inside kernel",
        dense_w_out,
        "full-block path must handle dense cross-head output projection and its gradients inside the owned CUDA path",
    )

    output_recompute = (
        "auto z_tmp = torch::empty({B, T, P}, bf16_options);" in cuda
        and "phase_gemm_abt_store_bf16(" in cuda
        and "grad_z[idx3(b, t, p, T, P)] = f2b(o_value * gate_value);" in cuda
        and "sum_sq = block_sum(sum_sq, shmem);" in cuda
    )
    failures += not check(
        "output gate recompute path",
        output_recompute,
        "current checkpoint should recompute z/output-norm state in backward instead of saving o_norm/z from forward",
    )

    pre_recompute = (
        "auto pre_tmp = torch::empty({B, T, C}, bf16_options);" in cuda
        and "float q_preact = conv_at(qkv, conv_w, b, t, q_channel, T, C, K);" in cuda
        and "float v_preact = conv_at(qkv, conv_w, b, t, v_channel, T, C, K);" in cuda
    )
    failures += not check(
        "pre activation recompute path",
        pre_recompute,
        "current checkpoint should keep pre temporary in forward and recompute conv preactivations in backward instead of saving pre",
    )

    qkv_post_recompute = (
        "shared.q_hist[local_t * Dk + i] = q_value;" in cuda
        and "shared.v_hist[local_t * dv_local + j] = v_value;" in cuda
        and "float* q_preact_s = shmem;" in cuda
        and "float* k_preact_s = shmem + Dk;" in cuda
        and "const bf16* __restrict__ q_norm" not in cuda
        and "const bf16* __restrict__ k_norm" not in cuda
        and "const bf16* __restrict__ v_post" not in cuda
        and "const float* __restrict__ inv_q" not in cuda
        and "const float* __restrict__ inv_k" not in cuda
    )
    failures += not check(
        "qkv post/norm recompute path",
        qkv_post_recompute,
        "backward should recompute q/k/v post-conv state from qkv and conv_w instead of saving q_norm/k_norm/v_post/inv_q/inv_k",
    )

    control_tail_parallel = (
        "atomicAdd(&grad_A_log[h]," in cuda and "atomicAdd(&grad_dt_bias[h]," in cuda
    )
    failures += not check(
        "control-tail reduction parallelized",
        control_tail_parallel,
        "current checkpoint should accumulate grad_A_log/grad_dt_bias across the BT*H control loop instead of leaving a serialized H-job tail",
    )

    control_guard = (
        "gdn_use_cuda_megakernel" in trainer
        and "gdn_use_cuda_corekernel" in trainer
        and "gdn_control_proj_fp32" in trainer
        and "requires GDN_CONTROL_PROJ_FP32=0" in trainer
        and re.search(
            r"if args\.gdn_use_cuda_corekernel or args\.gdn_use_cuda_megakernel:[\s\S]{0,1200}if args\.gdn_control_proj_fp32:[\s\S]{0,320}raise RuntimeError",
            trainer,
        )
    )
    failures += not check(
        "control projection dtype preflight",
        bool(control_guard),
        "trainer should reject an owned HGDN CUDA kernel path with GDN_CONTROL_PROJ_FP32=1",
    )

    status_guard = (
        'f"hgdn_{mode_name}_preflight:' in trainer
        and "hgdn_megakernel_extension_status" in trainer
        and re.search(
            r"if args\.gdn_use_cuda_corekernel or args\.gdn_use_cuda_megakernel:[\s\S]{0,1400}mode_name\s*=\s*[\"']corekernel[\"'] if args\.gdn_use_cuda_corekernel else [\"']megakernel[\"'][\s\S]{0,1400}loaded",
            trainer,
        )
    )
    failures += not check(
        "trainer extension preflight",
        bool(status_guard),
        "trainer should log extension status and fail before training when an owned HGDN CUDA kernel path is unavailable",
    )

    compile_guard = bool(
        "compile_top_level: bool = True" in hgdn_utils
        and '"gdn_blocks_compiled": 0' in hgdn_utils
        and "block.gdn = maybe_compile(" in hgdn_utils
        and "gdn_blocks_compiled" in trainer
    )
    failures += not check(
        "megakernel compile integration",
        compile_guard,
        "prepare_hybrid_compile should compile owned HGDN CUDA-kernel blocks separately and support top-level compile ordering",
    )

    honest_core_compare_helper = bool(
        re.search(
            r"default_compare100_candidate_specs=.*packed_control:GDN_USE_CUDA_COREKERNEL=0,GDN_USE_CUDA_MEGAKERNEL=0,GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=1;core_rc8:GDN_USE_CUDA_COREKERNEL=1,GDN_USE_CUDA_MEGAKERNEL=0,GDN_USE_PACKED_QKV_CONV_CUSTOM_BACKWARD=0,GDN_COREKERNEL_REC_CHUNK_T=8",
            core_helper,
        )
    )
    failures += not check(
        "corekernel compare100 control contract",
        honest_core_compare_helper,
        "core-kernel compare helper should default to the live packed control and the preferred core chunk knob",
    )

    core_helper_launcher = (
        'local launcher_mode="${HK_TRAINER_LAUNCHER_MODE:-plain}"' in core_helper
        and 'COMPILE_STRATEGY="${compile_strategy}"' in core_helper
        and "-u RANK -u WORLD_SIZE -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT"
        in core_helper
        and '"${python_cmd[@]}" train_gpt_hybrid.py' in core_helper
    )
    failures += not check(
        "corekernel 1x helper launcher contract",
        core_helper_launcher,
        "the 1x core-kernel helper should default to a plain single-process launcher and selective compile strategy",
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

    core_medium_parity = all(
        token in test_core_harness
        for token in (
            "B1_T128",
            "B1_T512",
            "core_forward_ms",
            "eager_forward_ms",
            'f"{reference_name}/{label}"',
        )
    )
    failures += not check(
        "corekernel medium parity coverage",
        core_medium_parity,
        "core-kernel harness should cover B=1,T=128 and B=1,T=512 with eager-control timing",
    )

    core_optional_b2 = (
        "--include-b2-t512" in test_core_harness and "B2_T512" in test_core_harness
    )
    failures += not check(
        "corekernel optional B2,T512 coverage",
        core_optional_b2,
        "core-kernel harness should expose an optional B=2,T=512 parity case",
    )

    core_t2048_flag = (
        "--include-b1-t2048" in test_core_harness and "B1_T2048" in test_core_harness
    )
    failures += not check(
        "corekernel optional B1,T2048 coverage",
        core_t2048_flag,
        "core-kernel harness should expose an optional B=1,T=2048 parity case for the H100 gate",
    )

    core_launch_check = all(
        token in test_core_harness
        for token in (
            "hgdn_core_forward_bf16_kernel",
            "hgdn_core_backward_bf16_kernel",
            "core_forward_launch_count",
            "core_backward_launch_count",
        )
    )
    failures += not check(
        "corekernel owned-launch check",
        core_launch_check,
        "core-kernel harness should prove one owned forward + one owned backward launch",
    )

    print()
    if failures:
        print(f"HGDN megakernel audit failed with {failures} issue(s).")
        raise SystemExit(1)
    print("HGDN megakernel static audit passed.")


if __name__ == "__main__":
    main()
