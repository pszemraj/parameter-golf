#!/usr/bin/env python3
"""Plan bounded adaptive 5090 closeout experiments."""

from __future__ import annotations

import argparse
import copy
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
    DECISION_PLANNED_STEPS,
    SCREEN_EFFECTIVE_STEP_TOKENS,
    AnalyzedRow,
    Geometry,
    ScreenContract,
    as_float,
    as_int,
    confirmation_train_label,
    decision,
    estimated_time_matched_steps,
    eligibility_errors,
    full_eval_accounting_errors,
    load_benchmark,
    load_summary_row,
    parse_geometry,
    row_matches_contract,
    screen_bpb,
    screen_contract,
    speed_ratio,
)


CONFIRM_STEPS = 8192
CONFIRM_HOLD_STEPS = 7000
PROBE_STEPS = 4096
PROBE_HOLD_STEPS = 3500
DECISION_STEPS = CONFIRM_STEPS
DECISION_HOLD_STEPS = CONFIRM_HOLD_STEPS
BPTT_STEPS = PROBE_STEPS
BPTT_HOLD_STEPS = PROBE_HOLD_STEPS
DEFAULT_BPTT_BATCH_SIZE = 128
DEFAULT_BPTT_CHUNKS = 2
DEFAULT_BPTT_IMPROVEMENT_BPB = 0.005
DEFAULT_LEARNING_RATE = 0.0035
DEFAULT_BATCH_SIZE = 256
DEFAULT_SEQ_LEN = 512
SCREEN_PLANNED_STEPS = 4096
SCREEN_TRIGRAM_TOP_K = 2
ARTIFACT_LIMIT_BYTES = 16_000_000


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


@dataclass(frozen=True)
class StagePlan:
    """Adaptive stage status plus commands."""

    status: str
    reason: str
    commands: list[PlannedCommand]


@dataclass(frozen=True)
class ConfirmationCandidate:
    """One promoted screen row plus the run-version contract it came from."""

    row: AnalyzedRow
    args: argparse.Namespace


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param Optional[list[str]] argv: Optional argument list for tests.
    :return argparse.Namespace: Parsed command-line arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument(
        "--stage",
        choices=("confirmations", "bptt", "k4", "finalist"),
        required=True,
    )
    ap.add_argument("--run-version", default="geom1")
    ap.add_argument(
        "--candidate-run-version",
        action="append",
        default=None,
        help="Additional screen run-version to compare for confirmation selection.",
    )
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
    ap.add_argument(
        "--bptt-improvement-bpb",
        type=float,
        default=DEFAULT_BPTT_IMPROVEMENT_BPB,
        help=(
            "Minimum fixed-token BPB improvement required before combining "
            "BPTT2 with K4. Defaults to a small noise guard."
        ),
    )
    ap.add_argument("--screen-steps", type=int, default=SCREEN_PLANNED_STEPS)
    ap.add_argument("--lr-hold-steps", type=int, default=None)
    ap.add_argument("--effective-step-tokens", type=int, default=SCREEN_EFFECTIVE_STEP_TOKENS)
    ap.add_argument("--confirm-steps", type=int, default=CONFIRM_STEPS)
    ap.add_argument("--confirm-hold-steps", type=int, default=CONFIRM_HOLD_STEPS)
    ap.add_argument(
        "--confirm-full-val-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require and emit a full final validation for confirmation rows.",
    )
    ap.add_argument(
        "--experiment-tier",
        choices=("probe", "decision"),
        default="decision",
        help=(
            "Default contract for adaptive variants. probe is sampled 512M; "
            "decision is 1B plus full final validation."
        ),
    )
    ap.add_argument("--variant-steps", type=int, default=None)
    ap.add_argument("--variant-hold-steps", type=int, default=None)
    ap.add_argument(
        "--variant-full-val-final",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether adaptive variants run a full final validation.",
    )
    ap.add_argument("--screen-trigram-top-k", type=int, default=None)
    ap.add_argument("--screen-batch-size", type=int, default=None)
    ap.add_argument("--screen-bptt-chunks", type=int, default=None)
    ap.add_argument("--bptt-batch-size", type=int, default=None)
    ap.add_argument("--bptt-chunks", type=int, default=DEFAULT_BPTT_CHUNKS)
    ap.add_argument("--finalist-run-version", default=None)
    ap.add_argument("--finalist-seeds", default=None)
    ap.add_argument(
        "--finalist-stability-check",
        action="store_true",
        help=(
            "Allow multiple finalist seeds for an explicit stability report. "
            "Default finalist closeout is single-seed; seeds are not a search axis."
        ),
    )
    ap.add_argument("--finalist-trigram-top-k", type=int, default=None)
    ap.add_argument("--finalist-seq-len", type=int, default=None)
    ap.add_argument("--finalist-batch-size", type=int, default=None)
    ap.add_argument("--finalist-bptt-chunks", type=int, default=None)
    ap.add_argument("--finalist-steps", type=int, default=None)
    ap.add_argument("--finalist-hold-steps", type=int, default=None)
    ap.add_argument(
        "--finalist-full-val-final",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--finalist-train-label", default=None)
    ap.add_argument(
        "--finalist-preflight-only",
        action="store_true",
        help="Prepare the shared spec and run artifact preflight without training.",
    )
    ap.add_argument(
        "--preflight-trainable-payload-bytes",
        type=int,
        default=None,
        help="Trainable payload byte estimate used by artifact preflight-only commands.",
    )
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--val-steps", type=int, default=None)
    ap.add_argument("--count-workers", type=int, default=None)
    ap.add_argument("--spec-max-tokens", type=int, default=None)
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
    contract = screen_contract(args)
    decision_grade = int(contract.planned_steps) >= DECISION_PLANNED_STEPS
    rows: list[AnalyzedRow] = []
    for label in tuple(args.label or DEFAULT_LABELS):
        geometry = parse_geometry(label)
        summary = load_summary_row(
            repo_root,
            geometry,
            run_version=args.run_version,
            seed=str(args.seed),
            contract=contract,
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
                verdict=decision(
                    delta,
                    ratio,
                    valid_screen_row=valid,
                    decision_grade=decision_grade,
                ),
                eligibility_errors=errors,
                estimated_time_matched_steps=estimated_time_matched_steps(
                    summary,
                    ratio,
                    valid_screen_row=valid,
                ),
            )
        )
    return rows


def candidate_run_versions(args: argparse.Namespace) -> tuple[str, ...]:
    """Return source run-versions to compare for confirmation planning.

    :param argparse.Namespace args: Parsed arguments.
    :return tuple[str, ...]: Ordered unique run-version suffixes.
    """
    raw = [str(args.run_version)]
    raw.extend(str(item) for item in (args.candidate_run_version or ()))
    out: list[str] = []
    for item in raw:
        if item not in out:
            out.append(item)
    return tuple(out)


def args_for_run_version(args: argparse.Namespace, run_version: str) -> argparse.Namespace:
    """Return an args copy with a different source run-version.

    :param argparse.Namespace args: Base parsed arguments.
    :param str run_version: Source run-version.
    :return argparse.Namespace: Copied arguments.
    """
    out = copy.copy(args)
    out.run_version = str(run_version)
    return out


def promoted_confirmation_candidates(args: argparse.Namespace) -> list[ConfirmationCandidate]:
    """Return promoted rows across all requested source run-versions.

    :param argparse.Namespace args: Parsed arguments.
    :return list[ConfirmationCandidate]: Promoted confirmation candidates.
    """
    candidates: list[ConfirmationCandidate] = []
    for run_version in candidate_run_versions(args):
        local_args = args_for_run_version(args, run_version)
        candidates.extend(
            ConfirmationCandidate(row=row, args=local_args)
            for row in analyze_geometry_rows(local_args)
            if row.is_valid_screen_row and row.verdict.startswith("promote")
        )
    return candidates


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
    return eligibility_errors(geometry, row, screen_contract(args))


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


def row_float_matches(
    row: dict[str, str], field: str, expected: float, *, tol: float = 1e-9
) -> bool:
    """Return whether a row float field matches an expected value.

    :param dict[str, str] row: Summary row.
    :param str field: Field name.
    :param float expected: Expected numeric value.
    :param float tol: Absolute tolerance.
    :return bool: ``True`` when the field parses and matches.
    """
    actual = as_float(row.get(field))
    return actual is not None and abs(actual - float(expected)) <= float(tol)


def artifact_is_valid(row: dict[str, str], *, limit: int = ARTIFACT_LIMIT_BYTES) -> bool:
    """Return whether a row has a known under-limit artifact estimate.

    :param dict[str, str] row: Summary row.
    :param int limit: Submission artifact byte limit.
    :return bool: ``True`` when artifact status and bytes are evidence-grade.
    """
    status = str(row.get("artifact_status", "")).strip()
    if status not in {"LEFT_ON_TABLE", "EXACT_LIMIT", "UNDER_LIMIT"}:
        return False
    nbytes = as_int(row.get("artifact_estimate_bytes") or row.get("artifact_bytes"))
    return nbytes is not None and 0 < nbytes <= int(limit)


def validation_source_is_valid(row: dict[str, str]) -> bool:
    """Return whether a summary row used the explicit validation shard.

    :param dict[str, str] row: Summary row.
    :return bool: ``True`` when the validation source is evidence-grade.
    """
    return str(row.get("validation_source", "")).strip() == "explicit_val_shard"


def expected_screen_batch_size(args: argparse.Namespace) -> int:
    """Return expected BPTT1 screen/confirmation batch size.

    :param argparse.Namespace args: Parsed arguments.
    :return int: Expected batch size.
    """
    return int(screen_contract(args).batch_size)


def expected_seq_len(args: argparse.Namespace) -> int:
    """Return expected sequence length.

    :param argparse.Namespace args: Parsed arguments.
    :return int: Expected sequence length.
    """
    return int(screen_contract(args).seq_len)


def expected_screen_bptt_chunks(args: argparse.Namespace) -> int:
    """Return expected screen/confirmation BPTT chunks.

    :param argparse.Namespace args: Parsed arguments.
    :return int: Expected BPTT chunks.
    """
    return int(screen_contract(args).bptt_chunks)


def batch_for_effective_tokens(args: argparse.Namespace, *, seq_len: int, bptt_chunks: int) -> int:
    """Return the batch size that preserves the effective token contract.

    :param argparse.Namespace args: Parsed arguments.
    :param int seq_len: Sequence length.
    :param int bptt_chunks: BPTT chunk count.
    :return int: Batch size for one local step.
    :raises SystemExit: If the effective-token contract is not divisible.
    """
    denom = int(seq_len) * int(bptt_chunks)
    if denom <= 0:
        raise SystemExit(f"invalid seq_len*bptt_chunks denominator: {denom}")
    effective = int(args.effective_step_tokens)
    if effective % denom != 0:
        raise SystemExit(
            "effective-step tokens must divide evenly by seq_len*bptt_chunks "
            f"({effective} vs {denom})"
        )
    return max(1, effective // denom)


def expected_bptt_batch_size(args: argparse.Namespace) -> int:
    """Return the expected BPTT2 batch size for the active context.

    :param argparse.Namespace args: Parsed arguments.
    :return int: Explicit or contract-derived BPTT batch size.
    """
    if args.bptt_batch_size is not None:
        return int(args.bptt_batch_size)
    return batch_for_effective_tokens(
        args,
        seq_len=expected_seq_len(args),
        bptt_chunks=int(args.bptt_chunks),
    )


def expected_screen_trigram_top_k(args: argparse.Namespace) -> int:
    """Return expected screen/confirmation trigram top-K.

    :param argparse.Namespace args: Parsed arguments.
    :return int: Expected trigram top-K.
    """
    return int(screen_contract(args).trigram_top_k)


def full_val_accounting_is_valid(row: dict[str, str]) -> bool:
    """Return whether a full-validation row has trustworthy accounting.

    :param dict[str, str] row: Summary row.
    :return bool: ``True`` when no stale full-validation accounting is present.
    """
    return not full_eval_accounting_errors(row)


def valid_confirmation_row(row: dict[str, str], args: argparse.Namespace) -> bool:
    """Return whether a row is a completed 1B geometry confirmation.

    :param dict[str, str] row: Summary row.
    :param argparse.Namespace args: Parsed arguments.
    :return bool: ``True`` when the confirmation contract is satisfied.
    """
    valid = (
        row_is_completed(row)
        and as_int(row.get("planned_steps")) == int(args.confirm_steps)
        and as_int(row.get("effective_step_tokens")) == int(args.effective_step_tokens)
        and as_int(row.get("num_blocks")) == 0
        and as_int(row.get("trigram_top_k")) == expected_screen_trigram_top_k(args)
        and as_int(row.get("batch_size")) == expected_screen_batch_size(args)
        and as_int(row.get("seq_len")) == expected_seq_len(args)
        and as_int(row.get("bptt_chunks")) == expected_screen_bptt_chunks(args)
        and as_int(row.get("lr_hold_steps")) == int(args.confirm_hold_steps)
        and row_float_matches(row, "learning_rate", DEFAULT_LEARNING_RATE)
        and row_bool(row, "exact_val_bpb")
        and artifact_is_valid(row)
        and validation_source_is_valid(row)
    )
    if not valid:
        return False
    if bool(args.confirm_full_val_final):
        if not row_bool(row, "last_eval_full_coverage"):
            return False
        if not full_val_accounting_is_valid(row):
            return False
    return True


def confirmation_contract(args: argparse.Namespace) -> ScreenContract:
    """Return the exact contract for a confirmation row.

    :param argparse.Namespace args: Parsed arguments.
    :return ScreenContract: Confirmation contract.
    """
    contract = screen_contract(args)
    return ScreenContract(
        planned_steps=int(args.confirm_steps),
        effective_step_tokens=contract.effective_step_tokens,
        num_blocks=contract.num_blocks,
        trigram_top_k=contract.trigram_top_k,
        seq_len=contract.seq_len,
        batch_size=contract.batch_size,
        bptt_chunks=contract.bptt_chunks,
        lr_hold_steps=int(args.confirm_hold_steps),
    )


def valid_screen_variant_row(
    row: dict[str, str],
    args: argparse.Namespace,
    *,
    top_k: int,
    batch_size: Optional[int] = None,
    bptt_chunks: Optional[int] = None,
) -> bool:
    """Return whether a row is a completed adaptive variant.

    :param dict[str, str] row: Summary row.
    :param argparse.Namespace args: Parsed arguments.
    :param int top_k: Expected trigram top-K.
    :param Optional[int] batch_size: Optional expected batch size.
    :param Optional[int] bptt_chunks: Optional expected BPTT chunks.
    :return bool: ``True`` when the row matches the requested screen variant.
    """
    if not row_is_completed(row):
        return False
    if as_int(row.get("planned_steps")) != variant_steps(args):
        return False
    if as_int(row.get("effective_step_tokens")) != int(args.effective_step_tokens):
        return False
    if as_int(row.get("num_blocks")) != 0 or as_int(row.get("trigram_top_k")) != int(top_k):
        return False
    if as_int(row.get("seq_len")) != expected_seq_len(args):
        return False
    if as_int(row.get("lr_hold_steps")) != variant_hold_steps(args):
        return False
    if not row_float_matches(row, "learning_rate", DEFAULT_LEARNING_RATE):
        return False
    if not row_bool(row, "exact_val_bpb"):
        return False
    if not validation_source_is_valid(row):
        return False
    if row.get("artifact_status") == "OVER_LIMIT":
        return False
    if variant_full_val_final(args) and not artifact_is_valid(row):
        return False
    if batch_size is not None and as_int(row.get("batch_size")) != int(batch_size):
        return False
    if bptt_chunks is not None and as_int(row.get("bptt_chunks")) != int(bptt_chunks):
        return False
    if variant_full_val_final(args):
        if not row_bool(row, "last_eval_full_coverage"):
            return False
        if not full_val_accounting_is_valid(row):
            return False
    return True


def variant_contract(
    args: argparse.Namespace,
    *,
    top_k: int,
    batch_size: Optional[int] = None,
    bptt_chunks: Optional[int] = None,
) -> ScreenContract:
    """Return the exact contract for an adaptive variant row.

    :param argparse.Namespace args: Parsed arguments.
    :param int top_k: Expected trigram top-K.
    :param Optional[int] batch_size: Optional batch-size override.
    :param Optional[int] bptt_chunks: Optional BPTT chunk override.
    :return ScreenContract: Variant contract.
    """
    contract = screen_contract(args)
    return ScreenContract(
        planned_steps=variant_steps(args),
        effective_step_tokens=contract.effective_step_tokens,
        num_blocks=contract.num_blocks,
        trigram_top_k=int(top_k),
        seq_len=contract.seq_len,
        batch_size=int(batch_size if batch_size is not None else contract.batch_size),
        bptt_chunks=int(bptt_chunks if bptt_chunks is not None else contract.bptt_chunks),
        lr_hold_steps=variant_hold_steps(args),
    )


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
    count_workers: Optional[int] = None,
    spec_max_tokens: Optional[int] = None,
    trigram_max_tokens: Optional[int] = None,
    data_max_tokens: Optional[int] = None,
    no_wandb: bool = False,
    smoke_test: bool = False,
    preflight_only: bool = False,
    preflight_trainable_payload_bytes: Optional[int] = None,
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
    :param Optional[int] count_workers: Optional trigram count worker override.
    :param Optional[int] spec_max_tokens: Optional base spec-build token cap.
    :param Optional[int] trigram_max_tokens: Optional trigram count cap.
    :param Optional[int] data_max_tokens: Optional train-data token cap.
    :param bool no_wandb: Whether to disable W&B.
    :param bool smoke_test: Whether to mark the launcher as smoke/debug.
    :param bool preflight_only: Whether to stop after shared-spec artifact preflight.
    :param Optional[int] preflight_trainable_payload_bytes: Optional trainable
        payload byte estimate for artifact preflight.
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
    if count_workers is not None:
        cmd.extend(["--count-workers", str(count_workers)])
    if spec_max_tokens is not None:
        cmd.extend(["--spec-max-tokens", str(spec_max_tokens)])
    if trigram_max_tokens is not None:
        cmd.extend(["--trigram-max-tokens", str(trigram_max_tokens)])
    if data_max_tokens is not None:
        cmd.extend(["--data-max-tokens", str(data_max_tokens)])
    if preflight_only:
        cmd.append("--preflight-only")
    if preflight_trainable_payload_bytes is not None:
        cmd.extend(
            [
                "--preflight-trainable-payload-bytes",
                str(preflight_trainable_payload_bytes),
            ]
        )
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
    contract = screen_contract(args)
    return {
        "seq_len": contract.seq_len,
        "target_effective_step_tokens": int(args.effective_step_tokens),
        "val_steps": args.val_steps,
        "count_workers": args.count_workers,
        "spec_max_tokens": args.spec_max_tokens,
        "trigram_max_tokens": args.trigram_max_tokens,
        "data_max_tokens": args.data_max_tokens,
        "no_wandb": bool(args.no_wandb),
        "smoke_test": bool(args.smoke_test),
    }


def shared_passthrough_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Return non-contract launcher kwargs shared by adaptive commands.

    :param argparse.Namespace args: Parsed arguments.
    :return dict[str, object]: Keyword arguments for ``geometry_command``.
    """
    return {
        "val_steps": args.val_steps,
        "count_workers": args.count_workers,
        "spec_max_tokens": args.spec_max_tokens,
        "trigram_max_tokens": args.trigram_max_tokens,
        "data_max_tokens": args.data_max_tokens,
        "no_wandb": bool(args.no_wandb),
        "smoke_test": bool(args.smoke_test),
    }


def contract_train_label(contract: ScreenContract, *, prefix: str) -> str:
    """Return a train-label for a screen or confirmation contract.

    :param ScreenContract contract: Contract to encode.
    :param str prefix: Leading token-budget label, such as ``1b`` or ``512m``.
    :return str: Launcher train-label.
    """
    return confirmation_train_label(contract).replace("1b", prefix, 1)


def plan_confirmations(args: argparse.Namespace) -> list[PlannedCommand]:
    """Plan 1B confirmations for promoted geometry screen rows.

    :param argparse.Namespace args: Parsed arguments.
    :return list[PlannedCommand]: Planned confirmation commands.
    """
    candidates = promoted_confirmation_candidates(args)
    candidates.sort(
        key=lambda candidate: (
            0 if candidate.row.verdict == "promote_1b" else 1,
            (
                screen_bpb(candidate.row.summary)
                if screen_bpb(candidate.row.summary) is not None
                else float("inf")
            ),
            -(candidate.row.speed_ratio or 0.0),
        )
    )

    out: list[PlannedCommand] = []
    for candidate in candidates:
        row = candidate.row
        local_args = candidate.args
        contract = screen_contract(local_args)
        geometry = row.geometry
        existing = load_summary_row(
            local_args.repo_root.resolve(),
            geometry,
            run_version=confirmation_run_version(local_args),
            seed=str(local_args.seed),
            contract=confirmation_contract(local_args),
        )
        if valid_confirmation_row(existing, local_args):
            continue
        bpb = screen_bpb(row.summary)
        reason = (
            f"source={local_args.run_version}; {row.verdict}; screen_bpb={bpb:.10f}; "
            f"delta={row.delta_bpb:.6f}; speed_ratio={(row.speed_ratio or 0.0):.3f}"
        )
        out.append(
            PlannedCommand(
                stage="confirmations",
                label=geometry.label,
                reason=reason,
                command=geometry_command(
                    geometry.label,
                    run_version=confirmation_run_version(local_args),
                    seed=str(local_args.seed),
                    num_steps=int(local_args.confirm_steps),
                    hold_steps=int(local_args.confirm_hold_steps),
                    trigram_top_k=contract.trigram_top_k,
                    full_val_final=bool(local_args.confirm_full_val_final),
                    val_every=512,
                    log_every=128,
                    log_state_every=512,
                    save_every=4096,
                    batch_size=contract.batch_size,
                    bptt_chunks=contract.bptt_chunks,
                    train_label=(
                        "smoke_confirm"
                        if bool(local_args.smoke_test)
                        else confirmation_train_label(contract)
                    ),
                    **shared_command_kwargs(local_args),
                ),
            )
        )
        if len(out) >= max(0, int(local_args.max_confirmations)):
            break
    return out


def plan_confirmations_stage(args: argparse.Namespace) -> StagePlan:
    """Plan confirmation commands and classify empty plans.

    :param argparse.Namespace args: Parsed arguments.
    :return StagePlan: Stage status and planned commands.
    """
    commands = plan_confirmations(args)
    if commands:
        return StagePlan("commands", f"{len(commands)} confirmation command(s) selected", commands)
    if completed_confirmations(args):
        return StagePlan("already_complete", "selected confirmations already completed", [])
    promoted = promoted_confirmation_candidates(args)
    if promoted:
        return StagePlan("already_complete", "promoted rows already have valid confirmations", [])
    return StagePlan(
        "blocked",
        "no promoted completed geometry screen rows are available for confirmation",
        [],
    )


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
            contract=confirmation_contract(args),
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
    contract = screen_contract(args)
    label, confirm_row = confirmations[0]
    existing = load_summary_row(
        args.repo_root.resolve(),
        parse_geometry(label),
        run_version=bptt_run_version(args),
        seed=str(args.seed),
        contract=variant_contract(
            args,
            top_k=contract.trigram_top_k,
            batch_size=expected_bptt_batch_size(args),
            bptt_chunks=int(args.bptt_chunks),
        ),
    )
    if valid_screen_variant_row(
        existing,
        args,
        top_k=contract.trigram_top_k,
        batch_size=expected_bptt_batch_size(args),
        bptt_chunks=int(args.bptt_chunks),
    ):
        return []
    reason = (
        "best completed geometry confirmation; "
        f"confirm_bpb={(screen_bpb(confirm_row) or float('nan')):.10f}"
    )
    bptt_batch_size = expected_bptt_batch_size(args)
    return [
        PlannedCommand(
            stage="bptt",
            label=label,
            reason=reason,
            command=geometry_command(
                label,
                run_version=bptt_run_version(args),
                seed=str(args.seed),
                num_steps=variant_steps(args),
                hold_steps=variant_hold_steps(args),
                trigram_top_k=contract.trigram_top_k,
                full_val_final=variant_full_val_final(args),
                val_every=1024 if variant_full_val_final(args) else 512,
                log_every=512 if variant_full_val_final(args) else 256,
                log_state_every=4096 if variant_full_val_final(args) else 2048,
                save_every=4096 if variant_full_val_final(args) else 2048,
                batch_size=bptt_batch_size,
                bptt_chunks=int(args.bptt_chunks),
                train_label=(
                    "smoke_bptt2"
                    if bool(args.smoke_test)
                    else contract_train_label(
                        ScreenContract(
                            planned_steps=contract.planned_steps,
                            effective_step_tokens=contract.effective_step_tokens,
                            num_blocks=contract.num_blocks,
                            trigram_top_k=contract.trigram_top_k,
                            seq_len=contract.seq_len,
                            batch_size=bptt_batch_size,
                            bptt_chunks=int(args.bptt_chunks),
                        ),
                        prefix=token_budget_label(args, variant_steps(args)),
                    )
                ),
                **shared_command_kwargs(args),
            ),
        )
    ]


def plan_bptt_stage(args: argparse.Namespace) -> StagePlan:
    """Plan the BPTT stage and classify empty plans.

    :param argparse.Namespace args: Parsed arguments.
    :return StagePlan: Stage status and planned commands.
    """
    confirmations = completed_confirmations(args)
    if not confirmations:
        return StagePlan("blocked", "no valid completed confirmation is available for BPTT", [])
    commands = plan_bptt(args)
    if commands:
        return StagePlan("commands", "BPTT command selected", commands)
    label, _row = confirmations[0]
    existing = load_summary_row(
        args.repo_root.resolve(),
        parse_geometry(label),
        run_version=bptt_run_version(args),
        seed=str(args.seed),
        contract=variant_contract(
            args,
            top_k=expected_screen_trigram_top_k(args),
            batch_size=expected_bptt_batch_size(args),
            bptt_chunks=int(args.bptt_chunks),
        ),
    )
    if valid_screen_variant_row(
        existing,
        args,
        top_k=expected_screen_trigram_top_k(args),
        batch_size=expected_bptt_batch_size(args),
        bptt_chunks=int(args.bptt_chunks),
    ):
        return StagePlan("already_complete", f"BPTT screen already completed for {label}", [])
    return StagePlan("blocked", f"BPTT prerequisites are inconsistent for {label}", [])


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
        contract=screen_contract(args),
    )
    base_errors = screen_eligibility_errors(geometry, base_row, args)
    if base_errors:
        return False, f"default BPTT1 selected; base screen invalid={','.join(base_errors)}"
    bptt_row = load_summary_row(
        args.repo_root.resolve(),
        geometry,
        run_version=bptt_run_version(args),
        seed=str(args.seed),
        contract=variant_contract(
            args,
            top_k=2,
            batch_size=expected_bptt_batch_size(args),
            bptt_chunks=int(args.bptt_chunks),
        ),
    )
    if not valid_screen_variant_row(
        bptt_row,
        args,
        top_k=2,
        batch_size=expected_bptt_batch_size(args),
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
        contract=variant_contract(
            args,
            top_k=2,
            batch_size=expected_bptt_batch_size(args),
            bptt_chunks=int(args.bptt_chunks),
        ),
    )
    if not valid_screen_variant_row(
        bptt_row,
        args,
        top_k=2,
        batch_size=expected_bptt_batch_size(args),
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
        contract=variant_contract(
            args,
            top_k=4,
            batch_size=expected_bptt_batch_size(args)
            if use_bptt
            else expected_screen_batch_size(args),
            bptt_chunks=int(args.bptt_chunks) if use_bptt else expected_screen_bptt_chunks(args),
        ),
    )
    if valid_screen_variant_row(
        existing,
        args,
        top_k=4,
        batch_size=expected_bptt_batch_size(args) if use_bptt else expected_screen_batch_size(args),
        bptt_chunks=int(args.bptt_chunks) if use_bptt else None,
    ):
        return []

    base_contract = screen_contract(args)
    extra_kwargs = {"batch_size": args.screen_batch_size}
    label_contract = ScreenContract(
        planned_steps=base_contract.planned_steps,
        effective_step_tokens=base_contract.effective_step_tokens,
        num_blocks=base_contract.num_blocks,
        trigram_top_k=4,
        seq_len=base_contract.seq_len,
        batch_size=base_contract.batch_size,
        bptt_chunks=base_contract.bptt_chunks,
    )
    train_label = (
        "smoke_k4"
        if bool(args.smoke_test)
        else contract_train_label(
            label_contract, prefix=token_budget_label(args, variant_steps(args))
        )
    )
    if use_bptt:
        extra_kwargs = {
            "batch_size": expected_bptt_batch_size(args),
            "bptt_chunks": int(args.bptt_chunks),
        }
        label_contract = ScreenContract(
            planned_steps=base_contract.planned_steps,
            effective_step_tokens=base_contract.effective_step_tokens,
            num_blocks=base_contract.num_blocks,
            trigram_top_k=4,
            seq_len=base_contract.seq_len,
            batch_size=expected_bptt_batch_size(args),
            bptt_chunks=int(args.bptt_chunks),
        )
        train_label = (
            "smoke_bptt2_k4"
            if bool(args.smoke_test)
            else contract_train_label(
                label_contract,
                prefix=token_budget_label(args, variant_steps(args)),
            )
        )

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
                num_steps=variant_steps(args),
                hold_steps=variant_hold_steps(args),
                trigram_top_k=4,
                full_val_final=variant_full_val_final(args),
                val_every=1024 if variant_full_val_final(args) else 512,
                log_every=512 if variant_full_val_final(args) else 256,
                log_state_every=4096 if variant_full_val_final(args) else 2048,
                save_every=4096 if variant_full_val_final(args) else 2048,
                train_label=train_label,
                **extra_kwargs,
                **shared_command_kwargs(args),
            ),
        )
    ]


def plan_k4_stage(args: argparse.Namespace) -> StagePlan:
    """Plan the K4 stage and classify empty plans.

    :param argparse.Namespace args: Parsed arguments.
    :return StagePlan: Stage status and planned commands.
    """
    confirmations = completed_confirmations(args)
    if not confirmations:
        return StagePlan("blocked", "no valid completed confirmation is available for K4", [])
    label, _confirm_row = confirmations[0]
    bptt_row = load_summary_row(
        args.repo_root.resolve(),
        parse_geometry(label),
        run_version=bptt_run_version(args),
        seed=str(args.seed),
        contract=variant_contract(
            args,
            top_k=2,
            batch_size=expected_bptt_batch_size(args),
            bptt_chunks=int(args.bptt_chunks),
        ),
    )
    if not valid_screen_variant_row(
        bptt_row,
        args,
        top_k=2,
        batch_size=expected_bptt_batch_size(args),
        bptt_chunks=int(args.bptt_chunks),
    ):
        return StagePlan("blocked", f"K4 requires a completed BPTT read for {label}", [])

    commands = plan_k4(args)
    if commands:
        return StagePlan("commands", "K4 command selected", commands)

    use_bptt, _reason = bptt_selected_for_k4(args, label)
    existing = load_summary_row(
        args.repo_root.resolve(),
        parse_geometry(label),
        run_version=k4_run_version(args, use_bptt=use_bptt),
        seed=str(args.seed),
        contract=variant_contract(
            args,
            top_k=4,
            batch_size=expected_bptt_batch_size(args)
            if use_bptt
            else expected_screen_batch_size(args),
            bptt_chunks=int(args.bptt_chunks) if use_bptt else expected_screen_bptt_chunks(args),
        ),
    )
    if valid_screen_variant_row(
        existing,
        args,
        top_k=4,
        batch_size=expected_bptt_batch_size(args) if use_bptt else expected_screen_batch_size(args),
        bptt_chunks=int(args.bptt_chunks) if use_bptt else None,
    ):
        return StagePlan("already_complete", f"K4 screen already completed for {label}", [])
    return StagePlan("blocked", f"K4 prerequisites are inconsistent for {label}", [])


def variant_steps(args: argparse.Namespace) -> int:
    """Return the resolved adaptive variant step count.

    :param argparse.Namespace args: Parsed arguments.
    :return int: Planned variant steps.
    """
    if args.variant_steps is not None:
        return int(args.variant_steps)
    if str(args.experiment_tier) == "probe":
        return PROBE_STEPS
    return DECISION_STEPS


def variant_hold_steps(args: argparse.Namespace) -> int:
    """Return the resolved adaptive variant LR hold length.

    :param argparse.Namespace args: Parsed arguments.
    :return int: Variant LR hold steps.
    """
    if args.variant_hold_steps is not None:
        return int(args.variant_hold_steps)
    if str(args.experiment_tier) == "probe":
        return PROBE_HOLD_STEPS
    return DECISION_HOLD_STEPS


def variant_full_val_final(args: argparse.Namespace) -> bool:
    """Return whether adaptive variants should run full final validation.

    :param argparse.Namespace args: Parsed arguments.
    :return bool: ``True`` for decision-grade variants.
    """
    if args.variant_full_val_final is not None:
        return bool(args.variant_full_val_final)
    return str(args.experiment_tier) == "decision"


def token_budget_label(args: argparse.Namespace, steps: int) -> str:
    """Return a compact token-budget label for run names.

    :param argparse.Namespace args: Parsed arguments.
    :param int steps: Planned optimizer steps.
    :return str: Token-budget label.
    """
    if bool(args.smoke_test):
        return "smoke"
    planned_tokens = int(steps) * int(args.effective_step_tokens)
    if planned_tokens == 4096 * 131072:
        return "512m"
    if planned_tokens == 8192 * 131072:
        return "1b"
    if planned_tokens % 1_000_000_000 == 0:
        return f"{planned_tokens // 1_000_000_000}b"
    if planned_tokens % 1_000_000 == 0:
        return f"{planned_tokens // 1_000_000}m"
    return f"{int(steps)}steps"


def split_cli_values(raw: Optional[str], *, fallback: str) -> list[str]:
    """Split a comma/space separated CLI value.

    :param Optional[str] raw: Raw CLI string.
    :param str fallback: Fallback string when ``raw`` is empty.
    :return list[str]: Non-empty values in input order.
    """
    text = fallback if raw is None or str(raw).strip() == "" else str(raw)
    return [part for part in text.replace(",", " ").split() if part]


def finalist_run_version(args: argparse.Namespace) -> str:
    """Return the run-version for a finalist contract.

    :param argparse.Namespace args: Parsed arguments.
    :return str: Finalist run-version.
    """
    return str(args.finalist_run_version or args.run_version)


def finalist_seeds(args: argparse.Namespace) -> list[str]:
    """Return seeds requested for a finalist contract.

    :param argparse.Namespace args: Parsed arguments.
    :return list[str]: Seed strings.
    """
    return split_cli_values(args.finalist_seeds, fallback=str(args.seed))


def finalist_contract_errors(args: argparse.Namespace) -> tuple[str, ...]:
    """Return missing finalist contract flags.

    :param argparse.Namespace args: Parsed arguments.
    :return tuple[str, ...]: Missing flag names.
    """
    missing: list[str] = []
    if not args.label or len(args.label) != 1:
        missing.append("exactly one --label")
    for field, flag in (
        ("finalist_trigram_top_k", "--finalist-trigram-top-k"),
        ("finalist_seq_len", "--finalist-seq-len"),
        ("finalist_batch_size", "--finalist-batch-size"),
        ("finalist_bptt_chunks", "--finalist-bptt-chunks"),
    ):
        if getattr(args, field) is None:
            missing.append(flag)
    return tuple(missing)


def finalist_contract(args: argparse.Namespace) -> ScreenContract:
    """Return the exact finalist contract.

    :param argparse.Namespace args: Parsed arguments.
    :return ScreenContract: Finalist run contract.
    """
    seq_len = int(args.finalist_seq_len)
    batch_size = int(args.finalist_batch_size)
    bptt_chunks = int(args.finalist_bptt_chunks)
    actual_effective = batch_size * seq_len * bptt_chunks
    if actual_effective != int(args.effective_step_tokens):
        raise SystemExit(
            "finalist batch contract does not match effective-step tokens: "
            f"batch={batch_size} seq_len={seq_len} bptt={bptt_chunks} "
            f"actual={actual_effective} expected={int(args.effective_step_tokens)}"
        )
    return ScreenContract(
        planned_steps=int(args.finalist_steps or args.confirm_steps),
        effective_step_tokens=int(args.effective_step_tokens),
        num_blocks=0,
        trigram_top_k=int(args.finalist_trigram_top_k),
        seq_len=seq_len,
        batch_size=batch_size,
        bptt_chunks=bptt_chunks,
        lr_hold_steps=int(args.finalist_hold_steps or args.confirm_hold_steps),
    )


def finalist_train_label(args: argparse.Namespace, contract: ScreenContract) -> str:
    """Return the train-label for a finalist command.

    :param argparse.Namespace args: Parsed arguments.
    :param ScreenContract contract: Finalist contract.
    :return str: Train-label suffix.
    """
    if args.finalist_train_label:
        return str(args.finalist_train_label)
    prefix = token_budget_label(args, contract.planned_steps)
    return contract_train_label(contract, prefix=prefix)


def valid_finalist_row(
    row: dict[str, str],
    *,
    geometry: Geometry,
    contract: ScreenContract,
    full_val_final: bool,
) -> bool:
    """Return whether a row satisfies a finalist contract.

    :param dict[str, str] row: Summary row.
    :param Geometry geometry: Expected geometry.
    :param ScreenContract contract: Expected run contract.
    :param bool full_val_final: Whether full final validation is required.
    :return bool: ``True`` when the row is evidence-grade.
    """
    if not row_is_completed(row):
        return False
    if not row_matches_contract(geometry, row, contract):
        return False
    if not row_float_matches(row, "learning_rate", DEFAULT_LEARNING_RATE):
        return False
    if not row_bool(row, "exact_val_bpb"):
        return False
    if not validation_source_is_valid(row):
        return False
    if not artifact_is_valid(row):
        return False
    if full_val_final:
        if not row_bool(row, "last_eval_full_coverage"):
            return False
        if not full_val_accounting_is_valid(row):
            return False
    return True


def finalist_command(args: argparse.Namespace, *, seed: str) -> PlannedCommand:
    """Return a finalist launcher command for one seed or preflight.

    :param argparse.Namespace args: Parsed arguments.
    :param str seed: Seed string.
    :return PlannedCommand: Planned command.
    """
    label = str(args.label[0])
    contract = finalist_contract(args)
    reason = (
        "finalist contract selected"
        if not args.finalist_preflight_only
        else "finalist artifact preflight selected"
    )
    return PlannedCommand(
        stage="finalist",
        label=label,
        reason=f"{reason}; seed={seed}",
        command=geometry_command(
            label,
            run_version=finalist_run_version(args),
            seed=seed,
            num_steps=contract.planned_steps,
            hold_steps=int(contract.lr_hold_steps or args.confirm_hold_steps),
            trigram_top_k=contract.trigram_top_k,
            full_val_final=bool(args.finalist_full_val_final),
            val_every=1024 if args.finalist_full_val_final else 512,
            log_every=512 if args.finalist_full_val_final else 256,
            log_state_every=4096 if args.finalist_full_val_final else 2048,
            save_every=4096 if args.finalist_full_val_final else 2048,
            batch_size=contract.batch_size,
            bptt_chunks=contract.bptt_chunks,
            train_label=finalist_train_label(args, contract),
            seq_len=contract.seq_len,
            target_effective_step_tokens=contract.effective_step_tokens,
            val_steps=args.val_steps,
            count_workers=args.count_workers,
            spec_max_tokens=args.spec_max_tokens,
            trigram_max_tokens=args.trigram_max_tokens,
            data_max_tokens=args.data_max_tokens,
            no_wandb=bool(args.no_wandb),
            smoke_test=bool(args.smoke_test),
            preflight_only=bool(args.finalist_preflight_only),
            preflight_trainable_payload_bytes=args.preflight_trainable_payload_bytes,
        ),
    )


def plan_finalist_stage(args: argparse.Namespace) -> StagePlan:
    """Plan missing runs for an explicit finalist contract.

    :param argparse.Namespace args: Parsed arguments.
    :return StagePlan: Stage status and planned commands.
    """
    errors = finalist_contract_errors(args)
    if errors:
        return StagePlan(
            "blocked",
            "finalist contract must be explicit; missing " + ", ".join(errors),
            [],
        )
    if args.finalist_preflight_only:
        return StagePlan(
            "commands",
            "finalist artifact preflight command selected",
            [finalist_command(args, seed=finalist_seeds(args)[0])],
        )
    seeds = finalist_seeds(args)
    if len(seeds) > 1 and not bool(args.finalist_stability_check):
        return StagePlan(
            "blocked",
            (
                "multiple finalist seeds require --finalist-stability-check; "
                "normal finalist closeout is single-seed and seeds are not a search axis"
            ),
            [],
        )

    geometry = parse_geometry(str(args.label[0]))
    contract = finalist_contract(args)
    commands: list[PlannedCommand] = []
    for seed in seeds:
        row = load_summary_row(
            args.repo_root.resolve(),
            geometry,
            run_version=finalist_run_version(args),
            seed=seed,
            contract=contract,
        )
        if not valid_finalist_row(
            row,
            geometry=geometry,
            contract=contract,
            full_val_final=bool(args.finalist_full_val_final),
        ):
            commands.append(finalist_command(args, seed=seed))
    if commands:
        return StagePlan("commands", f"{len(commands)} finalist command(s) selected", commands)
    return StagePlan(
        "already_complete",
        "all requested finalist seeds already satisfy the exact contract",
        [],
    )


def plan_stage(args: argparse.Namespace) -> StagePlan:
    """Plan and classify the requested adaptive stage.

    :param argparse.Namespace args: Parsed arguments.
    :return StagePlan: Stage status and planned commands.
    """
    if args.stage == "confirmations":
        return plan_confirmations_stage(args)
    if args.stage == "bptt":
        return plan_bptt_stage(args)
    if args.stage == "k4":
        return plan_k4_stage(args)
    if args.stage == "finalist":
        return plan_finalist_stage(args)
    raise ValueError(f"unknown stage: {args.stage}")


def write_shell_script(
    path: Path,
    commands: list[PlannedCommand],
    *,
    repo_root: Path,
    status: str = "commands",
    reason: str = "",
) -> None:
    """Write planned commands to a shell script.

    :param Path path: Output script path.
    :param list[PlannedCommand] commands: Planned commands.
    :param Path repo_root: Repository root for relative launcher commands.
    :param str status: Stage status.
    :param str reason: Stage-status reason.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root.resolve()))}",
        f"# adaptive_status={status}",
        f"# adaptive_reason={reason}",
        "",
    ]
    if not commands:
        lines.append(f"# No commands selected by the adaptive planner: {status}.")
    for item in commands:
        lines.append(f"# stage={item.stage} label={item.label}")
        lines.append(f"# reason={item.reason}")
        lines.append(item.shell)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def emit(stage_plan: StagePlan, *, mode: str) -> None:
    """Print planned commands.

    :param StagePlan stage_plan: Planned stage payload.
    :param str mode: Output mode.
    """
    commands = stage_plan.commands
    if mode == "json":
        payload = {
            "status": stage_plan.status,
            "reason": stage_plan.reason,
            "commands": [asdict(item) | {"shell": item.shell} for item in commands],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if mode == "shell":
        for item in commands:
            print(item.shell)
        return

    if not commands:
        print(f"Stage status: {stage_plan.status}")
        print(f"Reason: {stage_plan.reason}")
        return
    print(f"Stage status: {stage_plan.status}")
    print(f"Reason: {stage_plan.reason}")
    print()
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
    stage_plan = plan_stage(args)
    if args.write_script is not None:
        write_shell_script(
            args.write_script,
            stage_plan.commands,
            repo_root=args.repo_root,
            status=stage_plan.status,
            reason=stage_plan.reason,
        )
    emit(stage_plan, mode=str(args.emit))
    if stage_plan.status == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
