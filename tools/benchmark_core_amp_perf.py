#!/usr/bin/env python3
"""Benchmark Core/Amplifier runtime geometry without launching a sweep."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_amplifier_lm.model import AmplifierSpec, CoreAmplifierLM, MinGRUCore


DEFAULT_SHAPES = (
    "current_d48_l12_i480:48:12:10",
    "d64_l10_i512:64:10:8",
    "d96_l6_i512:96:6:5.333333333333333",
    "d96_l8_i512:96:8:5.333333333333333",
    "d128_l4_i512:128:4:4",
    "d128_l5_i512:128:5:4",
    "d128_l6_i384:128:6:3",
    "d160_l4_i512:160:4:3.2",
)


@dataclass(frozen=True)
class ShapeSpec:
    """Controller geometry to benchmark."""

    name: str
    dim: int
    layers: int
    expansion: float

    @property
    def dim_inner(self) -> int:
        """Return the minGRU inner recurrent width.

        :return int: Integer inner width used by ``MinGRUCell``.
        """
        return int(self.dim * self.expansion)

    @property
    def recurrent_cells(self) -> int:
        """Return total stacked recurrent cells.

        :return int: ``layers * dim_inner``.
        """
        return int(self.layers * self.dim_inner)


@dataclass(frozen=True)
class PerfResult:
    """One benchmark result row."""

    mode: str
    name: str
    dim: int
    layers: int
    expansion: float
    dim_inner: int
    recurrent_cells: int
    branch_count: int
    num_blocks: int
    batch_size: int
    seq_len: int
    tokens_per_step: int
    trainable_params: int
    fixed_nbytes: int
    mean_ms: float
    tokens_per_sec: float
    peak_mem_mib: float
    backend: str
    compiled: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return argparse.Namespace: Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["core", "full"], default="core")
    ap.add_argument("--shape", action="append", default=None, help="name:dim:layers:exp")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--branch-lags", default="1,2,3,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--num-blocks", type=int, default=0)
    ap.add_argument("--trigram-top-k", type=int, default=2)
    ap.add_argument("--compile", action="store_true", help="Benchmark torch.compile as requested")
    ap.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return ap.parse_args()


def parse_shape(raw: str) -> ShapeSpec:
    """Parse a shape specification.

    :param str raw: Shape string in ``name:dim:layers:exp`` format.
    :return ShapeSpec: Parsed shape.
    """
    fields = raw.split(":")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(f"invalid shape {raw!r}; expected name:dim:layers:exp")
    return ShapeSpec(
        name=fields[0],
        dim=int(fields[1]),
        layers=int(fields[2]),
        expansion=float(fields[3]),
    )


def iter_shapes(raw_shapes: Iterable[str] | None) -> list[ShapeSpec]:
    """Return parsed shape specs.

    :param Iterable[str] | None raw_shapes: Raw shape strings.
    :return list[ShapeSpec]: Parsed shape specs.
    """
    return [parse_shape(raw) for raw in (raw_shapes or DEFAULT_SHAPES)]


def synchronize() -> None:
    """Synchronize CUDA when available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def setup_cuda() -> torch.device:
    """Return the CUDA device for benchmarks.

    :return torch.device: CUDA device.
    """
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda")


def maybe_compile(model: torch.nn.Module, *, enabled: bool) -> torch.nn.Module:
    """Compile a model only when explicitly requested.

    :param torch.nn.Module model: Model to compile.
    :param bool enabled: Whether to call ``torch.compile``.
    :return torch.nn.Module: Original or compiled model.
    """
    if not enabled:
        return model
    return torch.compile(model, mode="reduce-overhead")


def benchmark_core(
    shape: ShapeSpec,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    compile_enabled: bool,
) -> PerfResult:
    """Benchmark only the trainable minGRU controller.

    :param ShapeSpec shape: Controller geometry.
    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :param int warmup: Untimed warmup steps.
    :param int steps: Timed steps.
    :param bool compile_enabled: Whether to use ``torch.compile``.
    :return PerfResult: Benchmark row.
    """
    device = setup_cuda()
    gc.collect()
    torch.cuda.empty_cache()
    model = MinGRUCore(
        shape.dim,
        shape.dim,
        num_layers=shape.layers,
        expansion_factor=shape.expansion,
        residual_blocks=True,
        residual_init=-3.0,
        scan_backend="auto",
    ).to(device)
    model = maybe_compile(model, enabled=compile_enabled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, seq_len, shape.dim, device=device)
    target = torch.randn(batch_size, seq_len, shape.dim, device=device)
    times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + steps):
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = model(x)
            loss = (out - target).float().square().mean()
        loss.backward()
        optimizer.step()
        synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - started)
    mean_sec = sum(times) / max(1, len(times))
    backend = "unknown"
    if hasattr(model, "active_scan_backend"):
        backend = str(model.active_scan_backend())
    params = sum(p.numel() for p in model.parameters())
    return PerfResult(
        mode="core",
        name=shape.name,
        dim=shape.dim,
        layers=shape.layers,
        expansion=shape.expansion,
        dim_inner=shape.dim_inner,
        recurrent_cells=shape.recurrent_cells,
        branch_count=0,
        num_blocks=0,
        batch_size=batch_size,
        seq_len=seq_len,
        tokens_per_step=batch_size * seq_len,
        trainable_params=params,
        fixed_nbytes=0,
        mean_ms=mean_sec * 1000.0,
        tokens_per_sec=(batch_size * seq_len) / mean_sec,
        peak_mem_mib=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        backend=backend,
        compiled=compile_enabled,
    )


def make_synthetic_spec(
    *,
    vocab_size: int,
    core_dim: int,
    branch_lags: tuple[int, ...],
    num_blocks: int,
    trigram_top_k: int,
) -> AmplifierSpec:
    """Build a synthetic spec for runtime benchmarking.

    :param int vocab_size: Vocabulary size.
    :param int core_dim: Core width.
    :param tuple[int, ...] branch_lags: Branch lags.
    :param int num_blocks: Synthetic frozen amplifier block count.
    :param int trigram_top_k: Synthetic trigram-memory top-K, or ``0`` for none.
    :return AmplifierSpec: Synthetic spec.
    """
    generator = torch.Generator(device="cpu").manual_seed(1234 + core_dim + trigram_top_k)
    branch_count = len(branch_lags)
    amp_dim = branch_count * core_dim
    trigram_top_tokens = None
    trigram_values = None
    trigram_confidence = None
    if trigram_top_k > 0:
        context_count = vocab_size * vocab_size
        trigram_top_tokens = torch.randint(
            0,
            vocab_size,
            (context_count, trigram_top_k),
            dtype=torch.int16,
            generator=generator,
        )
        trigram_values = torch.randint(
            -32,
            33,
            (context_count, trigram_top_k),
            dtype=torch.int8,
            generator=generator,
        )
        trigram_confidence = torch.randint(
            0,
            255,
            (context_count,),
            dtype=torch.uint8,
            generator=generator,
        )
    amp_w1 = torch.empty(0, amp_dim, amp_dim, dtype=torch.bfloat16)
    amp_w2 = torch.empty(0, amp_dim, amp_dim, dtype=torch.bfloat16)
    if num_blocks > 0:
        amp_w1 = (torch.randn(num_blocks, amp_dim, amp_dim, generator=generator) * 0.02).to(
            torch.bfloat16
        )
        amp_w2 = (torch.randn(num_blocks, amp_dim, amp_dim, generator=generator) * 0.02).to(
            torch.bfloat16
        )
    return AmplifierSpec(
        vocab_size=vocab_size,
        core_dim=core_dim,
        branch_lags=branch_lags,
        num_blocks=num_blocks,
        token_embed=(torch.randn(vocab_size, core_dim, generator=generator) * 0.02).to(
            torch.bfloat16
        ),
        base_bigram_logits=(torch.randn(vocab_size, vocab_size, generator=generator) * 0.02).to(
            torch.bfloat16
        ),
        lag_ops=(torch.randn(branch_count, core_dim, core_dim, generator=generator) * 0.02).to(
            torch.bfloat16
        ),
        amp_w1=amp_w1,
        amp_w2=amp_w2,
        readout_weight=(torch.randn(vocab_size, amp_dim, generator=generator) * 0.02).to(
            torch.bfloat16
        ),
        trigram_top_tokens=trigram_top_tokens,
        trigram_residual_values=trigram_values,
        trigram_context_confidence=trigram_confidence,
        metadata={
            "requested_core_dim": core_dim,
            "trigram_top_k": trigram_top_k,
            "synthetic_num_blocks": num_blocks,
        },
    )


def benchmark_full(
    shape: ShapeSpec,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    vocab_size: int,
    branch_lags: tuple[int, ...],
    num_blocks: int,
    trigram_top_k: int,
    compile_enabled: bool,
) -> PerfResult:
    """Benchmark a synthetic full Core/Amplifier training step.

    :param ShapeSpec shape: Controller geometry.
    :param int batch_size: Batch size.
    :param int seq_len: Sequence length.
    :param int warmup: Untimed warmup steps.
    :param int steps: Timed steps.
    :param int vocab_size: Vocabulary size.
    :param tuple[int, ...] branch_lags: Branch lags.
    :param int num_blocks: Frozen amplifier block count.
    :param int trigram_top_k: Trigram-memory top-K, or ``0`` for none.
    :param bool compile_enabled: Whether to use ``torch.compile``.
    :return PerfResult: Benchmark row.
    """
    device = setup_cuda()
    gc.collect()
    torch.cuda.empty_cache()
    spec = make_synthetic_spec(
        vocab_size=vocab_size,
        core_dim=shape.dim,
        branch_lags=branch_lags,
        num_blocks=num_blocks,
        trigram_top_k=trigram_top_k,
    )
    model = CoreAmplifierLM(
        spec,
        core_layers=shape.layers,
        core_expansion=shape.expansion,
        residual_core=True,
        residual_core_init=-3.0,
        branch_temporal_mode="current",
        residual_token_gate_mode="none",
        branch_router_mode="none",
        trigram_memory="frozen" if trigram_top_k > 0 else "none",
        trigram_log_scale_init=0.0,
        scan_backend="auto",
    ).to(device)
    model = maybe_compile(model, enabled=compile_enabled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    times: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + steps):
        optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(batch_size, device=device)
        synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs, state=state, return_state=True)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - started)
    mean_sec = sum(times) / max(1, len(times))
    return PerfResult(
        mode="full",
        name=shape.name,
        dim=shape.dim,
        layers=shape.layers,
        expansion=shape.expansion,
        dim_inner=shape.dim_inner,
        recurrent_cells=shape.recurrent_cells,
        branch_count=len(branch_lags),
        num_blocks=num_blocks,
        batch_size=batch_size,
        seq_len=seq_len,
        tokens_per_step=batch_size * seq_len,
        trainable_params=model.trainable_parameters,
        fixed_nbytes=spec.fixed_nbytes,
        mean_ms=mean_sec * 1000.0,
        tokens_per_sec=(batch_size * seq_len) / mean_sec,
        peak_mem_mib=torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        backend=model.active_scan_backend_name(),
        compiled=compile_enabled,
    )


def print_result(row: PerfResult) -> None:
    """Print one benchmark row.

    :param PerfResult row: Result row.
    """
    print(
        f"{row.mode}\t{row.name}\tdim={row.dim}\tlayers={row.layers}\t"
        f"exp={row.expansion:g}\tinner={row.dim_inner}\tcells={row.recurrent_cells}\t"
        f"branches={row.branch_count}\tblocks={row.num_blocks}\tparams={row.trainable_params}\t"
        f"fixed={row.fixed_nbytes}\tbackend={row.backend}\t"
        f"compiled={int(row.compiled)}\tmean_ms={row.mean_ms:.3f}\t"
        f"tok_s={row.tokens_per_sec:,.0f}\tpeak_mib={row.peak_mem_mib:.1f}",
        flush=True,
    )


def main() -> None:
    """Run the requested benchmark."""
    args = parse_args()
    shapes = iter_shapes(args.shape)
    branch_lags = tuple(int(x) for x in args.branch_lags.split(",") if x)
    print(
        f"device={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'} "
        f"mode={args.mode} batch={args.batch_size} seq={args.seq_len} "
        f"steps={args.steps} warmup={args.warmup} blocks={args.num_blocks} "
        f"compile={int(args.compile)}",
        flush=True,
    )
    rows: list[PerfResult] = []
    for shape in shapes:
        if args.mode == "core":
            row = benchmark_core(
                shape,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                steps=args.steps,
                compile_enabled=args.compile,
            )
        else:
            row = benchmark_full(
                shape,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                warmup=args.warmup,
                steps=args.steps,
                vocab_size=args.vocab_size,
                branch_lags=branch_lags,
                num_blocks=args.num_blocks,
                trigram_top_k=args.trigram_top_k,
                compile_enabled=args.compile,
            )
        print_result(row)
        rows.append(row)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
