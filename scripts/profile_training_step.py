#!/usr/bin/env python3
"""Benchmark and profile one steady-state training step for the active models.

This harness exists to keep ALlama kernel work honest. It runs the frozen
quality anchor and the GPT reference under the same single-GPU accumulation
contract used in local sweeps, with optional `torch.profiler` capture and
optional CUDA-profiler-range markers for Nsight.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from allama_shared import (  # noqa: E402
    HyperSharedALlama,
    HyperSharedConfig,
    OptimizerBundle,
    build_allama_optimizers,
    compile_model_for_train,
    configure_cuda_fastpath,
)
from train_gpt import (  # noqa: E402
    CONTROL_TENSOR_NAME_PATTERNS,
    CastedLinear,
    GPT,
    Muon,
    restore_low_dim_params_to_fp32,
)


@dataclass(frozen=True)
class ModelSpec:
    """Resolved benchmark model specification.

    :param str model: Canonical benchmark model name.
    :param int vocab_size: Token vocabulary size.
    :param int train_seq_len: Per-microbatch sequence length.
    :param int train_batch_tokens: Tokens per global step.
    :param int microbatches: Gradient-accumulation microbatches per global step.
    :param str sdpa_backend: Requested SDPA backend.
    """

    model: str
    vocab_size: int
    train_seq_len: int
    train_batch_tokens: int
    microbatches: int
    sdpa_backend: str

    @property
    def local_batch_size(self) -> int:
        """Return the per-microbatch batch size implied by the token contract.

        :return int: Batch size per microbatch.
        """
        denom = self.train_seq_len * self.microbatches
        if self.train_batch_tokens % denom != 0:
            raise ValueError(
                f"train_batch_tokens={self.train_batch_tokens} is not divisible by "
                f"train_seq_len * microbatches = {denom}"
            )
        batch_size = self.train_batch_tokens // denom
        if batch_size <= 0:
            raise ValueError("local batch size must be positive")
        return batch_size


@dataclass(frozen=True)
class HarnessSummary:
    """Serializable benchmark summary.

    :param str model: Canonical benchmark model name.
    :param int stored_params: Stored parameter count.
    :param int local_batch_size: Per-microbatch batch size.
    :param int train_seq_len: Per-microbatch sequence length.
    :param int microbatches: Microbatches per global step.
    :param int train_batch_tokens: Tokens per global step.
    :param int warmup_steps: Warmup global steps run before timing or capture.
    :param int measured_steps: Timed or profiled global steps.
    :param bool compile: Whether `torch.compile` was enabled.
    :param bool fullgraph: Whether fullgraph capture was requested.
    :param str sdpa_backend: Requested SDPA backend.
    :param float mean_step_s: Mean wall-clock time per global step.
    :param float tokens_per_s: Effective tokens per second for measured steps.
    :param int peak_cuda_mem_bytes: Peak CUDA memory during measured steps.
    """

    model: str
    stored_params: int
    local_batch_size: int
    train_seq_len: int
    microbatches: int
    train_batch_tokens: int
    warmup_steps: int
    measured_steps: int
    compile: bool
    fullgraph: bool
    sdpa_backend: str
    mean_step_s: float
    tokens_per_s: float
    peak_cuda_mem_bytes: int


class OptimizerAdapter:
    """Small adapter that hides the optimizer-stack shape.

    :param list[torch.optim.Optimizer] optimizers: Optimizers stepped together.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients across the wrapped optimizer stack.

        :param bool set_to_none: Whether to clear grads with `None`.
        :return None: Clears gradients in place.
        """
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Apply one optimizer step across the wrapped stack.

        :return None: Steps each optimizer once.
        """
        for opt in self.optimizers:
            opt.step()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the profiling harness.

    :return argparse.Namespace: Parsed command-line configuration.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model",
        choices=("allama_anchor", "gpt_reference"),
        required=True,
        help="Model contract to benchmark or profile.",
    )
    parser.add_argument(
        "--mode",
        choices=("benchmark", "torch-profiler"),
        default="benchmark",
        help="Whether to run wall-clock timing or torch.profiler capture.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./runs_allama_validation/perf_harness"),
        help="Directory for JSON summaries and optional profiler traces.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="Global steps to run before timing or capture.",
    )
    parser.add_argument(
        "--measured-steps",
        type=int,
        default=5,
        help="Global steps to time or profile after warmup.",
    )
    parser.add_argument(
        "--train-seq-len",
        type=int,
        default=1024,
        help="Sequence length per microbatch.",
    )
    parser.add_argument(
        "--train-batch-tokens",
        type=int,
        default=262_144,
        help="Tokens per global step.",
    )
    parser.add_argument(
        "--microbatches",
        type=int,
        default=64,
        help="Gradient-accumulation microbatches per global step.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="Vocabulary size used for synthetic tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for synthetic token generation.",
    )
    parser.add_argument(
        "--sdpa-backend",
        default="auto",
        help="SDPA backend passed through to CUDA fastpath setup.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket the measured region with cudaProfilerStart/Stop for Nsight.",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="Enable torch.compile.",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="compile_model",
        action="store_false",
        help="Disable torch.compile.",
    )
    parser.set_defaults(compile_model=True)
    fullgraph_group = parser.add_mutually_exclusive_group()
    fullgraph_group.add_argument(
        "--fullgraph",
        dest="fullgraph",
        action="store_true",
        help="Require fullgraph capture when compiling.",
    )
    fullgraph_group.add_argument(
        "--no-fullgraph",
        dest="fullgraph",
        action="store_false",
        help="Allow graph breaks when compiling.",
    )
    parser.set_defaults(fullgraph=True)
    return parser.parse_args()


def set_cuda_fast_math(sdpa_backend: str) -> dict[str, bool | str]:
    """Enable the same CUDA fast-math defaults used in training.

    :param str sdpa_backend: Requested SDPA backend.
    :return dict[str, bool | str]: Effective SDPA backend summary.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training-step harness")
    fastpath = configure_cuda_fastpath(sdpa_backend=sdpa_backend)
    if hasattr(torch.backends, "fp32_precision"):
        torch.backends.fp32_precision = "tf32"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if (
        hasattr(torch.backends, "cudnn")
        and hasattr(torch.backends.cudnn, "conv")
        and hasattr(torch.backends.cudnn.conv, "fp32_precision")
    ):
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    return fastpath


def build_model_spec(args: argparse.Namespace) -> ModelSpec:
    """Resolve the benchmark contract from CLI arguments.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return ModelSpec: Resolved model contract.
    """
    return ModelSpec(
        model=str(args.model),
        vocab_size=int(args.vocab_size),
        train_seq_len=int(args.train_seq_len),
        train_batch_tokens=int(args.train_batch_tokens),
        microbatches=int(args.microbatches),
        sdpa_backend=str(args.sdpa_backend),
    )


def build_allama_anchor(device: torch.device) -> tuple[nn.Module, OptimizerBundle]:
    """Construct the frozen ALlama quality anchor and optimizer bundle.

    :param torch.device device: CUDA device for model and optimizer state.
    :return tuple[nn.Module, OptimizerBundle]: Model and optimizer bundle.
    """
    cfg = HyperSharedConfig(
        vocab_size=1024,
        model_dim=896,
        embed_dim=896,
        num_layers=20,
        num_shared_blocks=4,
        share_pattern="cycle",
        num_heads=14,
        num_kv_heads=2,
        mlp_mult=1.5,
        mlp_multiple_of=128,
        rope_base=10000.0,
        norm_eps=1e-5,
        norm_kind="rmsnorm",
        norm_layout="prenorm",
        qk_norm=True,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        q_gain_init=1.5,
        x0_gate_init=-2.9444389792,
        use_x0_shortcut=True,
        use_final_norm=True,
        zero_init_residual=True,
        attn_dropout=0.0,
        resid_dropout=0.0,
        use_bias=False,
        cast_linears=True,
    )
    model = HyperSharedALlama(cfg).to(device).bfloat16()
    from allama_shared import restore_low_dim_params_to_fp32

    restore_low_dim_params_to_fp32(model)
    optimizers = build_allama_optimizers(
        model,
        tied_embed_lr=0.03,
        head_lr=0.01,
        matrix_lr=0.02,
        scalar_lr=0.04,
        beta1=0.9,
        beta2=0.95,
        adam_eps=1e-8,
        muon_momentum=0.95,
        muon_backend_steps=5,
    )
    return model, optimizers


def build_gpt_reference(device: torch.device) -> tuple[nn.Module, OptimizerAdapter]:
    """Construct the reference GPT baseline and optimizer stack.

    :param torch.device device: CUDA device for model and optimizer state.
    :return tuple[nn.Module, OptimizerAdapter]: Model and optimizer adapter.
    """
    model = (
        GPT(
            vocab_size=1024,
            num_layers=9,
            model_dim=512,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
        )
        .to(device)
        .bfloat16()
    )
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    block_named_params = list(model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2
        or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    optimizer_tok = torch.optim.Adam(
        [{"params": [model.tok_emb.weight], "lr": 0.05, "base_lr": 0.05}],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=0.04,
        momentum=0.95,
        backend_steps=5,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = 0.04
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": 0.04, "base_lr": 0.04}],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    return model, OptimizerAdapter(optimizers)


def build_model_and_optimizer(
    spec: ModelSpec,
    device: torch.device,
    compile_enabled: bool,
    fullgraph: bool,
) -> tuple[nn.Module, Any]:
    """Construct the selected benchmark model and optimizer stack.

    :param ModelSpec spec: Resolved benchmark contract.
    :param torch.device device: CUDA device for model and optimizer state.
    :param bool compile_enabled: Whether to compile the model.
    :param bool fullgraph: Whether to require fullgraph capture.
    :return tuple[nn.Module, Any]: Model and optimizer bundle or adapter.
    """
    if spec.model == "allama_anchor":
        model, optimizers = build_allama_anchor(device)
        model = compile_model_for_train(
            model, enabled=compile_enabled, fullgraph=fullgraph
        )
        return model, optimizers
    if spec.model == "gpt_reference":
        model, optimizers = build_gpt_reference(device)
        if compile_enabled and hasattr(torch, "compile"):
            model = torch.compile(model, dynamic=False, fullgraph=fullgraph)
        return model, optimizers
    raise ValueError(f"unknown model spec {spec.model!r}")


def make_static_batch(
    spec: ModelSpec,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create one reusable synthetic batch for the benchmark contract.

    :param ModelSpec spec: Resolved benchmark contract.
    :param torch.device device: CUDA device for the tokens.
    :param int seed: RNG seed for reproducibility.
    :return tuple[Tensor, Tensor]: Input ids and target ids shaped `(B, T)`.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (spec.local_batch_size, spec.train_seq_len)
    input_ids = torch.randint(
        low=0,
        high=spec.vocab_size,
        size=shape,
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    target_ids = torch.randint(
        low=0,
        high=spec.vocab_size,
        size=shape,
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    return input_ids, target_ids


def cuda_profiler_toggle(enabled: bool, start: bool) -> None:
    """Start or stop the CUDA profiler API when requested.

    :param bool enabled: Whether profiler-range control is active.
    :param bool start: Whether to start or stop the CUDA profiler.
    :return None: Issues CUDA profiler API calls when requested.
    """
    if not enabled:
        return
    cudart = torch.cuda.cudart()
    if start:
        cudart.cudaProfilerStart()
    else:
        cudart.cudaProfilerStop()


def run_global_step(
    model: nn.Module,
    optimizers: Any,
    input_ids: Tensor,
    target_ids: Tensor,
    microbatches: int,
    *,
    grad_scale: float,
    cuda_profiler_range: bool,
    step_label: str,
) -> None:
    """Run one optimizer step with fixed microbatch accumulation.

    :param nn.Module model: Model under test.
    :param Any optimizers: Optimizer bundle or adapter.
    :param Tensor input_ids: Static synthetic inputs.
    :param Tensor target_ids: Static synthetic targets.
    :param int microbatches: Gradient-accumulation microbatches per step.
    :param float grad_scale: Backward scaling factor `1 / microbatches`.
    :param bool cuda_profiler_range: Whether to bracket the measured region.
    :param str step_label: NVTX label for the global step.
    :return None: Performs one training step in place.
    """
    optimizers.zero_grad(set_to_none=True)
    torch.cuda.nvtx.range_push(step_label)
    cuda_profiler_toggle(cuda_profiler_range, start=True)
    try:
        for microbatch_idx in range(microbatches):
            torch.cuda.nvtx.range_push(f"microbatch_{microbatch_idx:03d}")
            loss = model(input_ids, target_ids)
            (loss * grad_scale).backward()
            torch.cuda.nvtx.range_pop()
        optimizers.step()
    finally:
        cuda_profiler_toggle(cuda_profiler_range, start=False)
        torch.cuda.nvtx.range_pop()


def benchmark(
    spec: ModelSpec,
    model: nn.Module,
    optimizers: Any,
    input_ids: Tensor,
    target_ids: Tensor,
    args: argparse.Namespace,
) -> HarnessSummary:
    """Measure mean steady-state global-step time for the resolved model.

    :param ModelSpec spec: Resolved benchmark contract.
    :param nn.Module model: Model under test.
    :param Any optimizers: Optimizer bundle or adapter.
    :param Tensor input_ids: Static synthetic inputs.
    :param Tensor target_ids: Static synthetic targets.
    :param argparse.Namespace args: Parsed CLI arguments.
    :return HarnessSummary: Serializable benchmark summary.
    """
    model.train()
    grad_scale = 1.0 / spec.microbatches
    for warmup_idx in range(args.warmup_steps):
        run_global_step(
            model,
            optimizers,
            input_ids,
            target_ids,
            spec.microbatches,
            grad_scale=grad_scale,
            cuda_profiler_range=False,
            step_label=f"warmup_step_{warmup_idx:02d}",
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for step_idx in range(args.measured_steps):
        run_global_step(
            model,
            optimizers,
            input_ids,
            target_ids,
            spec.microbatches,
            grad_scale=grad_scale,
            cuda_profiler_range=args.cuda_profiler_range and step_idx == 0,
            step_label=f"measured_step_{step_idx:02d}",
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mean_step = elapsed / args.measured_steps
    stored_params = int(sum(param.numel() for param in model.parameters()))
    return HarnessSummary(
        model=spec.model,
        stored_params=stored_params,
        local_batch_size=spec.local_batch_size,
        train_seq_len=spec.train_seq_len,
        microbatches=spec.microbatches,
        train_batch_tokens=spec.train_batch_tokens,
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(args.measured_steps),
        compile=bool(args.compile_model),
        fullgraph=bool(args.fullgraph),
        sdpa_backend=spec.sdpa_backend,
        mean_step_s=float(mean_step),
        tokens_per_s=float(spec.train_batch_tokens * args.measured_steps / elapsed),
        peak_cuda_mem_bytes=int(torch.cuda.max_memory_allocated()),
    )


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text to disk, creating parent directories first.

    :param Path path: File path to write.
    :param str text: Text payload.
    :return None: Writes the file to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_torch_profiler(
    spec: ModelSpec,
    model: nn.Module,
    optimizers: Any,
    input_ids: Tensor,
    target_ids: Tensor,
    args: argparse.Namespace,
    run_dir: Path,
) -> HarnessSummary:
    """Capture a `torch.profiler` trace for steady-state training steps.

    :param ModelSpec spec: Resolved benchmark contract.
    :param nn.Module model: Model under test.
    :param Any optimizers: Optimizer bundle or adapter.
    :param Tensor input_ids: Static synthetic inputs.
    :param Tensor target_ids: Static synthetic targets.
    :param argparse.Namespace args: Parsed CLI arguments.
    :param Path run_dir: Output directory for trace artifacts.
    :return HarnessSummary: Serializable profiler summary.
    """
    model.train()
    grad_scale = 1.0 / spec.microbatches
    for warmup_idx in range(args.warmup_steps):
        run_global_step(
            model,
            optimizers,
            input_ids,
            target_ids,
            spec.microbatches,
            grad_scale=grad_scale,
            cuda_profiler_range=False,
            step_label=f"warmup_step_{warmup_idx:02d}",
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    profiler_ctx = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    t0 = time.perf_counter()
    with profiler_ctx as prof:
        for step_idx in range(args.measured_steps):
            run_global_step(
                model,
                optimizers,
                input_ids,
                target_ids,
                spec.microbatches,
                grad_scale=grad_scale,
                cuda_profiler_range=args.cuda_profiler_range and step_idx == 0,
                step_label=f"profiled_step_{step_idx:02d}",
            )
            prof.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    events = prof.key_averages(group_by_input_shape=True)
    events_no_shapes = prof.key_averages()
    write_text(
        run_dir / "key_averages_cuda_self.txt",
        events_no_shapes.table(sort_by="self_cuda_time_total", row_limit=200),
    )
    write_text(
        run_dir / "key_averages_cuda_total.txt",
        events_no_shapes.table(sort_by="cuda_time_total", row_limit=200),
    )
    write_text(
        run_dir / "key_averages_shapes.txt",
        events.table(sort_by="self_cuda_time_total", row_limit=200),
    )
    prof.export_chrome_trace(str(run_dir / "trace.json"))
    stored_params = int(sum(param.numel() for param in model.parameters()))
    return HarnessSummary(
        model=spec.model,
        stored_params=stored_params,
        local_batch_size=spec.local_batch_size,
        train_seq_len=spec.train_seq_len,
        microbatches=spec.microbatches,
        train_batch_tokens=spec.train_batch_tokens,
        warmup_steps=int(args.warmup_steps),
        measured_steps=int(args.measured_steps),
        compile=bool(args.compile_model),
        fullgraph=bool(args.fullgraph),
        sdpa_backend=spec.sdpa_backend,
        mean_step_s=float(elapsed / args.measured_steps),
        tokens_per_s=float(spec.train_batch_tokens * args.measured_steps / elapsed),
        peak_cuda_mem_bytes=int(torch.cuda.max_memory_allocated()),
    )


def main() -> None:
    """Run the requested benchmark or profiler mode.

    :return None: Writes run artifacts under the requested output directory.
    """
    args = parse_args()
    spec = build_model_spec(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / spec.model
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    fastpath = set_cuda_fast_math(spec.sdpa_backend)
    model, optimizers = build_model_and_optimizer(
        spec,
        device=device,
        compile_enabled=bool(args.compile_model),
        fullgraph=bool(args.fullgraph),
    )
    input_ids, target_ids = make_static_batch(spec, device=device, seed=int(args.seed))

    if args.mode == "benchmark":
        summary = benchmark(spec, model, optimizers, input_ids, target_ids, args)
    else:
        summary = run_torch_profiler(
            spec, model, optimizers, input_ids, target_ids, args, run_dir
        )

    payload = {
        "summary": asdict(summary),
        "fastpath": fastpath,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
    }
    write_text(run_dir / "run_summary.json", json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
