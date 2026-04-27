#!/usr/bin/env python3
"""Run the local HGDN recurrence implementation matrix through argparse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _repo_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

from hgdn_helper_cli import parse_bool_flag  # noqa: E402
from hgdn_local_experiments import (  # noqa: E402
    LocalTrainContract,
    build_recurrence_matrix_plan,
    run_recurrence_matrix,
)
from hgdn_local_runner import (  # noqa: E402
    REPO_ROOT,
    enable_line_buffering,
    ensure_py7zr_available,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--use-wandb", type=parse_bool_flag, default=False)
    parser.add_argument(
        "--wandb-mode", default="offline", choices=["online", "offline"]
    )
    parser.add_argument("--wandb-project", default="pg-hgdn-ablations")
    parser.add_argument("--wandb-watch", default="none")
    parser.add_argument("--wandb-watch-log-freq", type=int, default=25)
    parser.add_argument("--run-prefix-base", default="localrecurmatrix1")
    parser.add_argument("--bundle-stage-dir", type=Path, default=None)
    parser.add_argument("--archive-output", type=Path, default=None)
    parser.add_argument("--command-log", type=Path, default=None)
    parser.add_argument("--torchinductor-max-autotune", type=int, default=0)
    parser.add_argument("--torchinductor-max-autotune-gemm", type=int, default=0)
    parser.add_argument("--torch-logs", default="")
    parser.add_argument("--torch-trace", default="")
    parser.add_argument("--allow-existing-logs", type=parse_bool_flag, default=False)
    parser.add_argument("--check-cuda-idle", type=parse_bool_flag, default=True)
    parser.add_argument("--allow-active-cuda-jobs", type=parse_bool_flag, default=False)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--train-batch-tokens", type=int, default=65_536)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--val-loss-every", type=int, default=100)
    parser.add_argument("--train-log-every", type=int, default=25)
    parser.add_argument("--min-val-seqs", type=int, default=512)
    parser.add_argument("--val-max-seqs", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-batch-seqs", type=int, default=None)
    parser.add_argument("--max-wallclock-seconds", type=float, default=0.0)
    parser.add_argument("--compile", type=parse_bool_flag, default=True)
    parser.add_argument("--compile-strategy", default="hybrid")
    parser.add_argument("--distributed-mode", default="parallel_muon")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--perf-skip-final-eval", type=parse_bool_flag, default=False)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data/datasets/fineweb10B_sp1024",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model",
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--hgdn-config",
        default="configs/hgdn/naive_contract_l8_d512_mid2_dk48_v2_m1p5.toml",
    )
    parser.add_argument(
        "--attn-config",
        default="configs/hgdn/naive_contract_l8_d512_r0_m1p5.toml",
    )
    return parser.parse_args(argv)


def build_contract(args: argparse.Namespace) -> LocalTrainContract:
    """Build the resolved local trainer contract."""
    return LocalTrainContract.resolve(
        python_bin=args.python_bin,
        use_wandb=args.use_wandb,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_watch=args.wandb_watch,
        wandb_watch_log_freq=args.wandb_watch_log_freq,
        ngpu=args.ngpu,
        grad_accum_steps=args.grad_accum_steps,
        iterations=args.iterations,
        train_batch_tokens=args.train_batch_tokens,
        train_seq_len=args.train_seq_len,
        val_loss_every=args.val_loss_every,
        train_log_every=args.train_log_every,
        min_val_seqs=args.min_val_seqs,
        val_max_seqs=args.val_max_seqs,
        val_batch_size=args.val_batch_size,
        val_batch_seqs=args.val_batch_seqs,
        max_wallclock_seconds=args.max_wallclock_seconds,
        compile=args.compile,
        compile_strategy=args.compile_strategy,
        distributed_mode=args.distributed_mode,
        weight_decay=args.weight_decay,
        perf_skip_final_eval=args.perf_skip_final_eval,
        torchinductor_max_autotune=args.torchinductor_max_autotune,
        torchinductor_max_autotune_gemm=args.torchinductor_max_autotune_gemm,
        torch_logs=args.torch_logs,
        torch_trace=args.torch_trace,
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        allow_existing_logs=args.allow_existing_logs,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the recurrence matrix."""
    enable_line_buffering()
    args = parse_args(argv)
    ensure_py7zr_available()
    contract = build_contract(args)
    plan = build_recurrence_matrix_plan(
        run_prefix_base=args.run_prefix_base,
        contract=contract,
        hgdn_config=args.hgdn_config,
        attn_config=args.attn_config,
        bundle_stage_dir=args.bundle_stage_dir,
        archive_output=args.archive_output,
        command_log=args.command_log,
        check_cuda_idle=args.check_cuda_idle,
        allow_active_cuda_jobs=args.allow_active_cuda_jobs,
    )
    run_recurrence_matrix(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
