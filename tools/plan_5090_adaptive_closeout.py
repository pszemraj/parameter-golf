#!/usr/bin/env python3
"""Plan bounded adaptive 5090 closeout experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_5090_geometry_frontier import (  # noqa: E402
    DEFAULT_BASELINE_BPB,
    DEFAULT_BASELINE_TOK_S,
    DEFAULT_LABELS,
    SCREEN_EFFECTIVE_STEP_TOKENS,
    AnalyzedRow,
    Geometry,
    as_float,
    as_int,
    decision,
    estimated_time_matched_steps,
    load_benchmark,
    load_summary_row,
    parse_geometry,
    screen_bpb,
    speed_ratio,
)


CONFIRM_STEPS = 8192
CONFIRM_HOLD_STEPS = 7000
BPTT_STEPS = 4096
BPTT_HOLD_STEPS = 3500
DEFAULT_BPTT_BATCH_SIZE = 128
DEFAULT_BPTT_CHUNKS = 2
SCREEN_PLANNED_STEPS = 4096
SCREEN_TRIGRAM_TOP_K = 2


@dataclass(frozen=True)
class PlannedCommand:
    """One command selected by the adaptive planner."""

    stage: str
    label: str
    reason: str
    command: list[str]

    @property
    def shell(self) -> str:
        """Return a shell-quoted command line.

        :return str: Command suitable for a generated shell script.
        """
        return shlex.join(self.command)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param Optional[list[str]] argv: Optional argument list for tests.
    :return argparse.Namespace: Parsed command-line arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--stage", choices=("confirmations", "bptt", "k4"), required=True)
    ap.add_argument("--run-version", default="geom1")
    ap.add_argument("--seed", default="1337")
    ap.add_argument("--benchmark", type=Path, default=None)
    ap.add_argument("--baseline-bpb", type=float, default=DEFAULT_BASELINE_BPB)
    ap.add_argument("--baseline-tok-s", type=float, default=DEFAULT_BASELINE_TOK_S)
    ap.add_argument("--label", action="append", default=None, help="Geometry label to consider")
    ap.add_argument("--max-confirmations", type=int, default=2)
    ap.add_argument("--confirmation-run-version", default=None)
    ap.add_argument("--bptt-run-version", default=None)
    ap.add_argument("--k4-run-version", default=None)
    ap.add_argument("--k4-bptt-run-version", default=None)
    ap.add_argument("--bptt-improvement-bpb", type=float, default=0.0)
    ap.add_argument("--screen-steps", type=int, default=SCREEN_PLANNED_STEPS)
    ap.add_argument("--effective-step-tokens", type=int, default=SCREEN_EFFECTIVE_STEP_TOKENS)
    ap.add_argument("--confirm-steps", type=int, default=CONFIRM_STEPS)
    ap.add_argument("--confirm-hold-steps", type=int, default=CONFIRM_HOLD_STEPS)
    ap.add_argument(
        "--confirm-full-val-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require and emit a full final validation for confirmation rows.",
    )
    ap.add_argument("--variant-steps", type=int, default=BPTT_STEPS)
    ap.add_argument("--variant-hold-steps", type=int, default=BPTT_HOLD_STEPS)
    ap.add_argument("--screen-batch-size", type=int, default=None)
    ap.add_argument("--bptt-batch-size", type=int, default=DEFAULT_BPTT_BATCH_SIZE)
    ap.add_argument("--bptt-chunks", type=int, default=DEFAULT_BPTT_CHUNKS)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--val-steps", type=int, default=None)
    ap.add_argument("--trigram-max-tokens", type=int, default=None)
    ap.add_argument("--data-max-tokens", type=int, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--write-script", type=Path, default=None)
    ap.add_argument("--emit", choices=("markdown", "json", "shell"), default="markdown")
    return ap.parse_args(argv)


def confirmation_run_version(args: argparse.Namespace) -> str:
    """Return the confirmation run-version suffix.

    :param argparse.Namespace args: Parsed arguments.
    :return str: Confirmation run-version.
    """
    return str(args.confirmation_run_version or f"{args.run_version}_confirm")


def bptt_run_version(args: argparse.Namespace) -> str:
    """Return the BPTT screen run-version suffix.

    :param argparse.Namespace args: Parsed arguments.
    :return str: BPTT run-version.
    """
    return str(args.bptt_run_version or f"{args.run_version}_bptt2")


def k4_run_version(args: argparse.Namespace, *, use_bptt: bool) -> str:
    """Return the K4 screen run-version suffix.

    :param argparse.Namespace args: Parsed arguments.
    :param bool use_bptt: Whether the K4 run uses BPTT2 settings.
    :return str: K4 run-version.
    """
    if use_bptt:
        return str(args.k4_bptt_run_version or f"{args.run_version}_k4_bptt2")
    return str(args.k4_run_version or f"{args.run_version}_k4")


def analyze_geometry_rows(args: argparse.Namespace) -> list[AnalyzedRow]:
    """Analyze geometry screen rows using the frontier analyzer rules.

    :param argparse.Namespace args: Parsed arguments.
    :return list[AnalyzedRow]: Analyzed geometry rows.
    """
    repo_root = args.repo_root.resolve()
    benchmark = load_benchmark(args.benchmark)
    baseline_bench = as_float(benchmark.get("current_d48_l12_i480", {}).get("tokens_per_sec"))
    rows: list[AnalyzedRow] = []
    for label in tuple(args.label or DEFAULT_LABELS):
        geometry = parse_geometry(label)
        summary = load_summary_row(
            repo_root,
            geometry,
            run_version=args.run_version,
            seed=str(args.seed),
        )
        errors = screen_eligibility_errors(geometry, summary, args)
        bpb = screen_bpb(summary)
        delta = None if bpb is None else bpb - float(args.baseline_bpb)
        ratio = speed_ratio(
            summary,
            benchmark.get(geometry.benchmark_name, {}),
            baseline_tok_s=float(args.baseline_tok_s),
            baseline_benchmark_tok_s=baseline_bench,
        )
        valid = not errors
        rows.append(
            AnalyzedRow(
                geometry=geometry,
                summary=summary,
                delta_bpb=delta,
                speed_ratio=ratio,
                verdict=decision(delta, ratio, valid_screen_row=valid),
                eligibility_errors=errors,
                estimated_time_matched_steps=estimated_time_matched_steps(
                    summary,
                    ratio,
                    valid_screen_row=valid,
                ),
            )
        )
    return rows


def screen_eligibility_errors(
    geometry: Geometry,
    row: dict[str, str],
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Return protocol violations for the active screen contract.

    :param Geometry geometry: Parsed geometry.
    :param dict[str, str] row: Summary row.
    :param argparse.Namespace args: Parsed arguments.
    :return tuple[str, ...]: Human-readable validation errors.
    """
    if not row:
        return ("missing_summary",)

    errors: list[str] = []
    if row.get("status") != "completed":
        errors.append("not_completed")
    expected = {
        "planned_steps": int(args.screen_steps),
        "effective_step_tokens": int(args.effective_step_tokens),
        "num_blocks": 0,
        "trigram_top_k": SCREEN_TRIGRAM_TOP_K,
        "core_dim": geometry.core_dim,
        "core_layers": geometry.layers,
        "core_inner_dim": geometry.inner_dim,
        "recurrent_cells": geometry.recurrent_cells,
    }
    if args.screen_batch_size is not None:
        expected["batch_size"] = int(args.screen_batch_size)
    for field, expected_value in expected.items():
        actual = as_int(row.get(field))
        if actual != expected_value:
            errors.append(f"{field}!={expected_value}")
    return tuple(errors)


def row_is_completed(row: dict[str, str]) -> bool:
    """Return whether a summary row is completed.

    :param dict[str, str] row: Summary row.
    :return bool: ``True`` for completed rows.
    """
    return bool(row) and row.get("status") == "completed"


def row_bool(row: dict[str, str], field: str) -> bool:
    """Parse a boolean summary field.

    :param dict[str, str] row: Summary row.
    :param str field: Field name.
    :return bool: Parsed boolean.
    """
    return str(row.get(field, "")).strip().lower() in {"1", "true", "yes", "on"}


def valid_confirmation_row(row: dict[str, str], args: argparse.Namespace) -> bool:
    """Return whether a row is a completed 1B K2 geometry confirmation.

    :param dict[str, str] row: Summary row.
    :param argparse.Namespace args: Parsed arguments.
    :return bool: ``True`` when the confirmation contract is satisfied.
    """
    valid = (
        row_is_completed(row)
        and as_int(row.get("planned_steps")) == int(args.confirm_steps)
        and as_int(row.get("effective_step_tokens")) == int(args.effective_step_tokens)
        and as_int(row.get("num_blocks")) == 0
        and as_int(row.get("trigram_top_k")) == 2
        and row_bool(row, "exact_val_bpb")
    )
    if not valid:
        return False
    if bool(args.confirm_full_val_final) and not row_bool(row, "last_eval_full_coverage"):
        return False
    return True


def valid_screen_variant_row(
    row: dict[str, str],
    args: argparse.Namespace,
    *,
    top_k: int,
    batch_size: Optional[int] = None,
    bptt_chunks: Optional[int] = None,
) -> bool:
    """Return whether a row is a completed 512M screen variant.

    :param dict[str, str] row: Summary row.
    :param argparse.Namespace args: Parsed arguments.
    :param int top_k: Expected trigram top-K.
    :param Optional[int] batch_size: Optional expected batch size.
    :param Optional[int] bptt_chunks: Optional expected BPTT chunks.
    :return bool: ``True`` when the row matches the requested screen variant.
    """
    if not row_is_completed(row):
        return False
    if as_int(row.get("planned_steps")) != int(args.variant_steps):
        return False
    if as_int(row.get("effective_step_tokens")) != int(args.effective_step_tokens):
        return False
    if as_int(row.get("num_blocks")) != 0 or as_int(row.get("trigram_top_k")) != int(top_k):
        return False
    if batch_size is not None and as_int(row.get("batch_size")) != int(batch_size):
        return False
    if bptt_chunks is not None and as_int(row.get("bptt_chunks")) != int(bptt_chunks):
        return False
    return True


def geometry_command(
    label: str,
    *,
    run_version: str,
    seed: str,
    num_steps: int,
    hold_steps: int,
    trigram_top_k: int,
    full_val_final: bool,
    val_every: int,
    log_every: int,
    log_state_every: int,
    save_every: int,
    batch_size: Optional[int] = None,
    bptt_chunks: Optional[int] = None,
    train_label: Optional[str] = None,
    seq_len: Optional[int] = None,
    target_effective_step_tokens: Optional[int] = None,
    val_steps: Optional[int] = None,
    trigram_max_tokens: Optional[int] = None,
    data_max_tokens: Optional[int] = None,
    no_wandb: bool = False,
    smoke_test: bool = False,
) -> list[str]:
    """Build an aligned-geometry launcher command.

    :param str label: Geometry label.
    :param str run_version: Run-version suffix.
    :param str seed: Seed string.
    :param int num_steps: Planned optimizer steps.
    :param int hold_steps: LR hold steps.
    :param int trigram_top_k: Trigram top-K setting.
    :param bool full_val_final: Whether to run full final validation.
    :param int val_every: Eval cadence.
    :param int log_every: Train log cadence.
    :param int log_state_every: State log cadence.
    :param int save_every: Checkpoint cadence.
    :param Optional[int] batch_size: Optional batch-size override.
    :param Optional[int] bptt_chunks: Optional BPTT chunk override.
    :param Optional[str] train_label: Optional train-label override for W&B/run names.
    :param Optional[int] seq_len: Optional sequence-length override.
    :param Optional[int] target_effective_step_tokens: Optional effective token contract.
    :param Optional[int] val_steps: Optional sampled validation steps.
    :param Optional[int] trigram_max_tokens: Optional trigram count cap.
    :param Optional[int] data_max_tokens: Optional train-data token cap.
    :param bool no_wandb: Whether to disable W&B.
    :param bool smoke_test: Whether to mark the launcher as smoke/debug.
    :return list[str]: Command argv.
    """
    geometry = parse_geometry(label)
    cmd = [
        "bash",
        "scripts/run_5090_trigram_aligned_geometry_screen.sh",
        "--run-version",
        str(run_version),
        "--seeds",
        str(seed),
        "--geometry-label",
        geometry.label,
        "--geometry-core-dim",
        str(geometry.core_dim),
        "--geometry-core-layers",
        str(geometry.layers),
        "--geometry-core-inner-dim",
        str(geometry.inner_dim),
        "--num-steps",
        str(num_steps),
        "--lr-hold-steps",
        str(hold_steps),
        "--trigram-top-k",
        str(trigram_top_k),
        "--val-every",
        str(val_every),
        "--log-every",
        str(log_every),
        "--log-state-every",
        str(log_state_every),
        "--save-every",
        str(save_every),
    ]
    if full_val_final:
        cmd.append("--full-val-final")
    else:
        cmd.append("--no-full-val-final")
    if batch_size is not None:
        cmd.extend(["--geometry-batch-size", str(batch_size)])
    if bptt_chunks is not None:
        cmd.extend(["--geometry-bptt-chunks", str(bptt_chunks)])
    if train_label is not None:
        cmd.extend(["--geometry-train-label", train_label])
    if seq_len is not None:
        cmd.extend(["--geometry-seq-len", str(seq_len)])
    if target_effective_step_tokens is not None:
        cmd.extend(["--target-effective-step-tokens", str(target_effective_step_tokens)])
    if val_steps is not None:
        cmd.extend(["--val-steps", str(val_steps)])
    if trigram_max_tokens is not None:
        cmd.extend(["--trigram-max-tokens", str(trigram_max_tokens)])
    if data_max_tokens is not None:
        cmd.extend(["--data-max-tokens", str(data_max_tokens)])
    if no_wandb:
        cmd.append("--no-wandb")
    if smoke_test:
        cmd.append("--smoke-test")
    return cmd


def shared_command_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Return launcher kwargs shared by all adaptive stages.

    :param argparse.Namespace args: Parsed arguments.
    :return dict[str, object]: Keyword arguments for ``geometry_command``.
    """
    return {
        "seq_len": args.seq_len,
        "target_effective_step_tokens": int(args.effective_step_tokens),
        "val_steps": args.val_steps,
        "trigram_max_tokens": args.trigram_max_tokens,
        "data_max_tokens": args.data_max_tokens,
        "no_wandb": bool(args.no_wandb),
        "smoke_test": bool(args.smoke_test),
    }


def plan_confirmations(args: argparse.Namespace) -> list[PlannedCommand]:
    """Plan 1B confirmations for promoted geometry screen rows.

    :param argparse.Namespace args: Parsed arguments.
    :return list[PlannedCommand]: Planned confirmation commands.
    """
    candidates = [
        row
        for row in analyze_geometry_rows(args)
        if row.is_valid_screen_row and row.verdict.startswith("promote")
    ]
    candidates.sort(
        key=lambda row: (
            0 if row.verdict == "promote_1b" else 1,
            screen_bpb(row.summary) if screen_bpb(row.summary) is not None else float("inf"),
            -(row.speed_ratio or 0.0),
        )
    )

    out: list[PlannedCommand] = []
    for row in candidates:
        geometry = row.geometry
        existing = load_summary_row(
            args.repo_root.resolve(),
            geometry,
            run_version=confirmation_run_version(args),
            seed=str(args.seed),
        )
        if valid_confirmation_row(existing, args):
            continue
        bpb = screen_bpb(row.summary)
        reason = (
            f"{row.verdict}; screen_bpb={bpb:.10f}; "
            f"delta={row.delta_bpb:.6f}; speed_ratio={(row.speed_ratio or 0.0):.3f}"
        )
        out.append(
            PlannedCommand(
                stage="confirmations",
                label=geometry.label,
                reason=reason,
                command=geometry_command(
                    geometry.label,
                    run_version=confirmation_run_version(args),
                    seed=str(args.seed),
                    num_steps=int(args.confirm_steps),
                    hold_steps=int(args.confirm_hold_steps),
                    trigram_top_k=2,
                    full_val_final=bool(args.confirm_full_val_final),
                    val_every=512,
                    log_every=128,
                    log_state_every=512,
                    save_every=4096,
                    batch_size=args.screen_batch_size,
                    train_label="smoke_confirm" if bool(args.smoke_test) else None,
                    **shared_command_kwargs(args),
                ),
            )
        )
        if len(out) >= max(0, int(args.max_confirmations)):
            break
    return out


def completed_confirmations(args: argparse.Namespace) -> list[tuple[str, dict[str, str]]]:
    """Return completed K2 geometry confirmations.

    :param argparse.Namespace args: Parsed arguments.
    :return list[tuple[str, dict[str, str]]]: ``(label, summary_row)`` pairs.
    """
    out: list[tuple[str, dict[str, str]]] = []
    for label in tuple(args.label or DEFAULT_LABELS):
        geometry = parse_geometry(label)
        row = load_summary_row(
            args.repo_root.resolve(),
            geometry,
            run_version=confirmation_run_version(args),
            seed=str(args.seed),
        )
        if valid_confirmation_row(row, args):
            out.append((label, row))
    out.sort(key=lambda item: screen_bpb(item[1]) or float("inf"))
    return out


def plan_bptt(args: argparse.Namespace) -> list[PlannedCommand]:
    """Plan a BPTT2 screen for the best completed confirmation.

    :param argparse.Namespace args: Parsed arguments.
    :return list[PlannedCommand]: Planned BPTT command, or empty list.
    """
    confirmations = completed_confirmations(args)
    if not confirmations:
        return []
    label, confirm_row = confirmations[0]
    existing = load_summary_row(
        args.repo_root.resolve(),
        parse_geometry(label),
        run_version=bptt_run_version(args),
        seed=str(args.seed),
    )
    if valid_screen_variant_row(
        existing,
        args,
        top_k=2,
        batch_size=int(args.bptt_batch_size),
        bptt_chunks=int(args.bptt_chunks),
    ):
        return []
    reason = (
        "best completed geometry confirmation; "
        f"confirm_bpb={(screen_bpb(confirm_row) or float('nan')):.10f}"
    )
    return [
        PlannedCommand(
            stage="bptt",
            label=label,
            reason=reason,
            command=geometry_command(
                label,
                run_version=bptt_run_version(args),
                seed=str(args.seed),
                num_steps=int(args.variant_steps),
                hold_steps=int(args.variant_hold_steps),
                trigram_top_k=2,
                full_val_final=False,
                val_every=256,
                log_every=64,
                log_state_every=256,
                save_every=2048,
                batch_size=int(args.bptt_batch_size),
                bptt_chunks=int(args.bptt_chunks),
                train_label="smoke_bptt2" if bool(args.smoke_test) else "512m_bptt2",
                **shared_command_kwargs(args),
            ),
        )
    ]


def bptt_selected_for_k4(
    args: argparse.Namespace,
    label: str,
) -> tuple[bool, str]:
    """Return whether K4 should combine with BPTT2 for a label.

    :param argparse.Namespace args: Parsed arguments.
    :param str label: Geometry label.
    :return tuple[bool, str]: Whether to use BPTT2 plus a reason.
    """
    geometry = parse_geometry(label)
    base_row = load_summary_row(
        args.repo_root.resolve(),
        geometry,
        run_version=args.run_version,
        seed=str(args.seed),
    )
    bptt_row = load_summary_row(
        args.repo_root.resolve(),
        geometry,
        run_version=bptt_run_version(args),
        seed=str(args.seed),
    )
    if not valid_screen_variant_row(
        bptt_row,
        args,
        top_k=2,
        batch_size=int(args.bptt_batch_size),
        bptt_chunks=int(args.bptt_chunks),
    ):
        return False, "no completed BPTT2 screen"
    base_bpb = screen_bpb(base_row)
    bptt_bpb = screen_bpb(bptt_row)
    if base_bpb is None or bptt_bpb is None:
        return False, "missing base or BPTT2 BPB"
    improvement = base_bpb - bptt_bpb
    if improvement >= float(args.bptt_improvement_bpb):
        return True, f"BPTT2 selected; improvement={improvement:.6f} bpb"
    return False, f"default BPTT1 selected; BPTT2 improvement={improvement:.6f} bpb"


def plan_k4(args: argparse.Namespace) -> list[PlannedCommand]:
    """Plan one K4 screen after confirmation and BPTT selection.

    :param argparse.Namespace args: Parsed arguments.
    :return list[PlannedCommand]: Planned K4 command, or empty list.
    """
    confirmations = completed_confirmations(args)
    if not confirmations:
        return []
    label, confirm_row = confirmations[0]
    bptt_row = load_summary_row(
        args.repo_root.resolve(),
        parse_geometry(label),
        run_version=bptt_run_version(args),
        seed=str(args.seed),
    )
    if not valid_screen_variant_row(
        bptt_row,
        args,
        top_k=2,
        batch_size=int(args.bptt_batch_size),
        bptt_chunks=int(args.bptt_chunks),
    ):
        return []
    use_bptt, bptt_reason = bptt_selected_for_k4(args, label)
    run_version = k4_run_version(args, use_bptt=use_bptt)
    existing = load_summary_row(
        args.repo_root.resolve(),
        parse_geometry(label),
        run_version=run_version,
        seed=str(args.seed),
    )
    if valid_screen_variant_row(
        existing,
        args,
        top_k=4,
        batch_size=int(args.bptt_batch_size) if use_bptt else args.screen_batch_size,
        bptt_chunks=int(args.bptt_chunks) if use_bptt else None,
    ):
        return []

    extra_kwargs = {"batch_size": args.screen_batch_size}
    train_label = "smoke_k4" if bool(args.smoke_test) else "512m_k4"
    if use_bptt:
        extra_kwargs = {
            "batch_size": int(args.bptt_batch_size),
            "bptt_chunks": int(args.bptt_chunks),
        }
        train_label = "smoke_bptt2_k4" if bool(args.smoke_test) else "512m_bptt2_k4"

    reason = (
        "best completed geometry confirmation; "
        f"confirm_bpb={(screen_bpb(confirm_row) or float('nan')):.10f}; {bptt_reason}"
    )
    return [
        PlannedCommand(
            stage="k4",
            label=label,
            reason=reason,
            command=geometry_command(
                label,
                run_version=run_version,
                seed=str(args.seed),
                num_steps=int(args.variant_steps),
                hold_steps=int(args.variant_hold_steps),
                trigram_top_k=4,
                full_val_final=False,
                val_every=256,
                log_every=64,
                log_state_every=256,
                save_every=2048,
                train_label=train_label,
                **extra_kwargs,
                **shared_command_kwargs(args),
            ),
        )
    ]


def plan_commands(args: argparse.Namespace) -> list[PlannedCommand]:
    """Plan commands for the requested stage.

    :param argparse.Namespace args: Parsed arguments.
    :return list[PlannedCommand]: Planned commands.
    """
    if args.stage == "confirmations":
        return plan_confirmations(args)
    if args.stage == "bptt":
        return plan_bptt(args)
    if args.stage == "k4":
        return plan_k4(args)
    raise ValueError(f"unknown stage: {args.stage}")


def write_shell_script(path: Path, commands: list[PlannedCommand], *, repo_root: Path) -> None:
    """Write planned commands to a shell script.

    :param Path path: Output script path.
    :param list[PlannedCommand] commands: Planned commands.
    :param Path repo_root: Repository root for relative launcher commands.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root.resolve()))}",
        "",
    ]
    if not commands:
        lines.append("# No commands selected by the adaptive planner.")
    for item in commands:
        lines.append(f"# stage={item.stage} label={item.label}")
        lines.append(f"# reason={item.reason}")
        lines.append(item.shell)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def emit(commands: list[PlannedCommand], *, mode: str) -> None:
    """Print planned commands.

    :param list[PlannedCommand] commands: Planned commands.
    :param str mode: Output mode.
    """
    if mode == "json":
        payload = [asdict(item) | {"shell": item.shell} for item in commands]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if mode == "shell":
        for item in commands:
            print(item.shell)
        return

    if not commands:
        print("No commands selected by the adaptive planner.")
        return
    print("| stage | label | reason | command |")
    print("| --- | --- | --- | --- |")
    for item in commands:
        print(
            "| {stage} | {label} | {reason} | `{command}` |".format(
                stage=item.stage,
                label=item.label,
                reason=item.reason.replace("|", "\\|"),
                command=item.shell.replace("`", "\\`"),
            )
        )


def main() -> None:
    """Plan the requested adaptive stage."""
    args = parse_args()
    commands = plan_commands(args)
    if args.write_script is not None:
        write_shell_script(args.write_script, commands, repo_root=args.repo_root)
    emit(commands, mode=str(args.emit))


if __name__ == "__main__":
    main()
