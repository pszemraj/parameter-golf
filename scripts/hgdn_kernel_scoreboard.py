#!/usr/bin/env python3
"""Score one HGDN kernel family against explicit H100 stop rules.

:param str family_name: Human-readable kernel family label.
:param list input_paths: One or more extracted bundle roots or `.7z` archives.
:param list control_perf_prefixes: Prefixes for same-day control perf runs.
:param list candidate_perf_prefixes: Prefixes for same-day candidate perf runs.
:param str control_profile_prefix: Prefix for the control eager/compiled profile.
:param str candidate_profile_prefix: Prefix for the candidate eager/compiled profile.
:param list target_rows: Optional exact or `re:` row selectors for the family.
:param str output: Optional markdown output path.
:return int: Process exit code.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

from _repo_bootstrap import ensure_repo_root_on_sys_path

REPO_ROOT = ensure_repo_root_on_sys_path()

from profiler_report import load_profile_report  # noqa: E402

PERF_RE = re.compile(
    r"perf_summary ignore_steps:\d+ measured_steps:\d+ step_ms:([0-9.]+) tokens_per_s:([0-9.]+)"
)


@dataclass(frozen=True)
class PerfRecord:
    """One parsed perf log row.

    :param str run_stem: File stem used for prefix matching.
    :param float step_ms: Parsed step time in milliseconds.
    :param float tokens_per_s: Parsed throughput.
    :param Path path: Source file path.
    """

    run_stem: str
    step_ms: float
    tokens_per_s: float
    path: Path


@dataclass(frozen=True)
class ScoreResult:
    """Computed scoreboard result.

    :param float control_mean_ms: Mean same-day control perf step time.
    :param float control_noise_ms: Sample stddev of same-day control perf step time.
    :param float candidate_mean_ms: Mean same-day candidate perf step time.
    :param float delta_ms: Candidate mean minus control mean.
    :param float meaningful_win_ms: Promotion threshold.
    :param float flat_band_ms: Noise / flat threshold.
    :param float eager_delta_ms: Candidate eager `ProfilerStep` delta.
    :param float compiled_delta_ms: Candidate compiled `ProfilerStep` delta.
    :param float compile_specific_penalty_ms: Compiled delta minus eager delta.
    :param float compiled_copy_tax_ms: `(aten::copy_ + direct_copy_kernel_cuda)` delta.
    :param float compiled_external_kernel_ms: Candidate-only custom-kernel self time.
    :param float compiled_fxgraph_delta_ms: `CompiledFxGraph` self-time delta.
    :param float compiled_ddp_forward_delta_ms: `DDP.forward` self-time delta.
    :param float targeted_bucket_upper_bound_ms: Control compiled target-family upper bound.
    :param str status: Family status classification.
    :param str rationale: Short explanation of the status.
    """

    control_mean_ms: float
    control_noise_ms: float
    candidate_mean_ms: float
    delta_ms: float
    meaningful_win_ms: float
    flat_band_ms: float
    eager_delta_ms: float
    compiled_delta_ms: float
    compile_specific_penalty_ms: float
    compiled_copy_tax_ms: float
    compiled_external_kernel_ms: float
    compiled_fxgraph_delta_ms: float
    compiled_ddp_forward_delta_ms: float
    targeted_bucket_upper_bound_ms: float
    status: str
    rationale: str


def extract_input(src: Path, dst_root: Path) -> Path:
    """Copy or extract one input tree into a temporary root.

    :param Path src: Directory or `.7z` path.
    :param Path dst_root: Extraction root.
    :raises RuntimeError: If `py7zr` is unavailable for `.7z` extraction.
    :raises ValueError: If the input type is unsupported.
    :return Path: Extracted directory path.
    """

    if src.is_dir():
        dst = dst_root / src.name
        shutil.copytree(src, dst)
        return dst
    if src.suffix == ".7z":
        dst = dst_root / src.stem
        dst.mkdir(parents=True, exist_ok=True)
        try:
            import py7zr  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Missing `py7zr` for .7z extraction. Install it with "
                "`python -m pip install py7zr`."
            ) from exc
        with py7zr.SevenZipFile(src, "r") as archive:
            archive.extractall(dst)
        return dst
    raise ValueError(f"Unsupported input: {src}")


def parse_perf_logs(root: Path) -> dict[str, PerfRecord]:
    """Parse all perf-summary logs under one extracted tree.

    :param Path root: Extracted bundle root.
    :return dict[str, PerfRecord]: Parsed logs keyed by file stem.
    """

    records: dict[str, PerfRecord] = {}
    for path in list(root.rglob("*.txt")) + list(root.rglob("*.log")):
        matches = list(PERF_RE.finditer(path.read_text(errors="ignore")))
        if not matches:
            continue
        # Some archived logs contain an earlier partial perf block followed by the
        # final rerun in the same file. Use the last summary so family scoring
        # reflects the terminal result rather than stale appended output.
        match = matches[-1]
        records[path.stem] = PerfRecord(
            run_stem=path.stem,
            step_ms=float(match.group(1)),
            tokens_per_s=float(match.group(2)),
            path=path,
        )
    return records


def parse_profiles(root: Path) -> dict[str, dict[str, object]]:
    """Load normalized profile reports under one extracted tree.

    :param Path root: Extracted bundle root.
    :return dict[str, dict[str, object]]: Reports keyed by `run_id`.
    """

    records: dict[str, dict[str, object]] = {}
    for json_path in root.rglob("key_averages.json"):
        report = load_profile_report(json_path)
        records[str(report["metadata"]["run_id"])] = report
    return records


def find_perf(records: dict[str, PerfRecord], prefix: str) -> PerfRecord:
    """Find exactly one perf record by prefix.

    :param dict[str, PerfRecord] records: Parsed perf records.
    :param str prefix: Required run prefix.
    :raises RuntimeError: If the match is missing or ambiguous.
    :return PerfRecord: Matching record.
    """

    matches = [record for stem, record in records.items() if stem.startswith(prefix)]
    if len(matches) != 1:
        stems = sorted(stem for stem in records if stem.startswith(prefix))
        raise RuntimeError(
            f"Expected exactly one perf run for prefix {prefix!r}, got {stems}"
        )
    return matches[0]


def find_profile(
    records: dict[str, dict[str, object]],
    prefix: str,
    *,
    compile_mode: bool,
) -> dict[str, object]:
    """Find exactly one eager or compiled profile by prefix.

    :param dict[str, dict[str, object]] records: Parsed profile reports.
    :param str prefix: Required run prefix.
    :param bool compile_mode: Desired compile mode.
    :raises RuntimeError: If the match is missing or ambiguous.
    :return dict[str, object]: Matching normalized profile report.
    """

    matches = [
        report
        for run_id, report in records.items()
        if run_id.startswith(prefix)
        and bool(report["metadata"]["compile"]) is compile_mode
    ]
    if len(matches) != 1:
        ids = sorted(str(report["metadata"]["run_id"]) for report in matches)
        mode = "compiled" if compile_mode else "eager"
        raise RuntimeError(
            f"Expected exactly one {mode} profile for prefix {prefix!r}, got {ids}"
        )
    return matches[0]


def row_sum_ms(report: dict[str, object], selectors: list[str]) -> float:
    """Sum self-device time for exact-name or regex-selected rows.

    Selector syntax:
    - exact row name: `aten::copy_`
    - regex row name: `re:ProfilerStep`

    :param dict[str, object] report: Normalized profile report.
    :param list[str] selectors: Exact names or `re:` selectors.
    :return float: Summed self-device time in milliseconds.
    """

    compiled: list[re.Pattern[str] | str] = []
    for selector in selectors:
        if selector.startswith("re:"):
            compiled.append(re.compile(selector[3:]))
        else:
            compiled.append(selector)
    total_us = 0.0
    for row in report["rows"]:
        name = str(row["name"])
        for selector in compiled:
            if isinstance(selector, str):
                if name == selector:
                    total_us += float(row["self_device_time_us"])
                    break
            elif selector.search(name):
                total_us += float(row["self_device_time_us"])
                break
    return total_us / 1000.0


def mean(values: list[float]) -> float:
    """Compute the mean of a non-empty list.

    :param list[float] values: Values to average.
    :return float: Mean.
    """

    return statistics.mean(values)


def noise(values: list[float]) -> float:
    """Compute sample-noise estimate from repeated control runs.

    :param list[float] values: Repeated control values.
    :return float: Sample stddev, or `0.0` when not defined.
    """

    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def classify_status(
    *,
    delta_ms: float,
    meaningful_win_ms: float,
    flat_band_ms: float,
    eager_delta_ms: float,
    compiled_delta_ms: float,
    compile_specific_penalty_ms: float,
    compiled_copy_tax_ms: float,
    targeted_bucket_upper_bound_ms: float,
) -> tuple[str, str]:
    """Classify one family under the HGDN stop-rule policy.

    :param float delta_ms: Candidate mean minus control mean.
    :param float meaningful_win_ms: Promotion threshold.
    :param float flat_band_ms: Flat / noise threshold.
    :param float eager_delta_ms: Eager `ProfilerStep` delta.
    :param float compiled_delta_ms: Compiled `ProfilerStep` delta.
    :param float compile_specific_penalty_ms: Compiled delta minus eager delta.
    :param float compiled_copy_tax_ms: Compiled copy-tax delta.
    :param float targeted_bucket_upper_bound_ms: Control compiled upper bound.
    :return tuple[str, str]: `(status, rationale)`.
    """

    eager_improved = eager_delta_ms <= -flat_band_ms
    compiled_worsened = compiled_delta_ms >= flat_band_ms
    perf_worsened = delta_ms >= flat_band_ms
    if 0.0 < targeted_bucket_upper_bound_ms < meaningful_win_ms:
        return (
            "SATURATED",
            "targeted compiled upper bound is below the meaningful-win threshold",
        )
    if eager_improved and (compiled_worsened or perf_worsened):
        return (
            "INTEGRATION_BOTTLENECK",
            "eager improved but compiled/perf worsened at the current boundary",
        )
    if compile_specific_penalty_ms > meaningful_win_ms:
        return (
            "INTEGRATION_BOTTLENECK",
            "compile-specific penalty is larger than the meaningful-win threshold",
        )
    if abs(delta_ms) < flat_band_ms:
        return ("FLAT", "same-day perf delta is inside the flat / noise band")
    if (
        delta_ms <= -meaningful_win_ms
        and compiled_copy_tax_ms <= flat_band_ms
        and compile_specific_penalty_ms <= meaningful_win_ms
    ):
        return (
            "PROMOTE",
            "same-day perf win exceeds threshold without reopened copy tax",
        )
    if delta_ms >= meaningful_win_ms:
        return (
            "KILLED",
            "same-day perf loss exceeds the meaningful-win threshold",
        )
    return ("ACTIVE", "mixed result; keep the family open pending more evidence")


def build_score_result(
    *,
    control_perf_values: list[float],
    candidate_perf_values: list[float],
    control_eager: dict[str, object],
    candidate_eager: dict[str, object],
    control_compiled: dict[str, object],
    candidate_compiled: dict[str, object],
    target_rows: list[str],
) -> ScoreResult:
    """Compute all scoreboard metrics for one family.

    :param list[float] control_perf_values: Same-day control perf step times.
    :param list[float] candidate_perf_values: Same-day candidate perf step times.
    :param dict[str, object] control_eager: Control eager profile.
    :param dict[str, object] candidate_eager: Candidate eager profile.
    :param dict[str, object] control_compiled: Control compiled profile.
    :param dict[str, object] candidate_compiled: Candidate compiled profile.
    :param list[str] target_rows: Exact or regex target selectors.
    :return ScoreResult: Computed metrics and classification.
    """

    control_mean_ms = mean(control_perf_values)
    control_noise_ms = noise(control_perf_values)
    candidate_mean_ms = mean(candidate_perf_values)
    delta_ms = candidate_mean_ms - control_mean_ms
    meaningful_win_ms = max(5.0, 0.005 * control_mean_ms, 3.0 * control_noise_ms)
    flat_band_ms = max(2.5, 0.0025 * control_mean_ms, 2.0 * control_noise_ms)
    eager_delta_ms = row_sum_ms(candidate_eager, ["re:ProfilerStep"]) - row_sum_ms(
        control_eager, ["re:ProfilerStep"]
    )
    compiled_delta_ms = row_sum_ms(
        candidate_compiled, ["re:ProfilerStep"]
    ) - row_sum_ms(control_compiled, ["re:ProfilerStep"])
    compile_specific_penalty_ms = compiled_delta_ms - eager_delta_ms
    compiled_copy_tax_ms = row_sum_ms(
        candidate_compiled, ["aten::copy_", "re:direct_copy_kernel_cuda"]
    ) - row_sum_ms(control_compiled, ["aten::copy_", "re:direct_copy_kernel_cuda"])
    compiled_external_kernel_ms = row_sum_ms(
        candidate_compiled,
        [
            "re:^hgdn_cuda_v2::",
            "re:^_PackedQKV",
            "re:^_RMSNorm",
            "re:causal_dwconv_",
        ],
    )
    compiled_fxgraph_delta_ms = row_sum_ms(
        candidate_compiled, ["re:## Call CompiledFxGraph"]
    ) - row_sum_ms(control_compiled, ["re:## Call CompiledFxGraph"])
    compiled_ddp_forward_delta_ms = row_sum_ms(
        candidate_compiled, ["DistributedDataParallel.forward"]
    ) - row_sum_ms(control_compiled, ["DistributedDataParallel.forward"])
    targeted_bucket_upper_bound_ms = row_sum_ms(control_compiled, target_rows)
    status, rationale = classify_status(
        delta_ms=delta_ms,
        meaningful_win_ms=meaningful_win_ms,
        flat_band_ms=flat_band_ms,
        eager_delta_ms=eager_delta_ms,
        compiled_delta_ms=compiled_delta_ms,
        compile_specific_penalty_ms=compile_specific_penalty_ms,
        compiled_copy_tax_ms=compiled_copy_tax_ms,
        targeted_bucket_upper_bound_ms=targeted_bucket_upper_bound_ms,
    )
    return ScoreResult(
        control_mean_ms=control_mean_ms,
        control_noise_ms=control_noise_ms,
        candidate_mean_ms=candidate_mean_ms,
        delta_ms=delta_ms,
        meaningful_win_ms=meaningful_win_ms,
        flat_band_ms=flat_band_ms,
        eager_delta_ms=eager_delta_ms,
        compiled_delta_ms=compiled_delta_ms,
        compile_specific_penalty_ms=compile_specific_penalty_ms,
        compiled_copy_tax_ms=compiled_copy_tax_ms,
        compiled_external_kernel_ms=compiled_external_kernel_ms,
        compiled_fxgraph_delta_ms=compiled_fxgraph_delta_ms,
        compiled_ddp_forward_delta_ms=compiled_ddp_forward_delta_ms,
        targeted_bucket_upper_bound_ms=targeted_bucket_upper_bound_ms,
        status=status,
        rationale=rationale,
    )


def render_markdown(
    *,
    family_name: str,
    control_perf_prefixes: list[str],
    candidate_perf_prefixes: list[str],
    control_profile_prefix: str,
    candidate_profile_prefix: str,
    target_rows: list[str],
    result: ScoreResult,
) -> str:
    """Render the scoreboard as markdown.

    :param str family_name: Human-readable family label.
    :param list[str] control_perf_prefixes: Same-day control perf prefixes.
    :param list[str] candidate_perf_prefixes: Same-day candidate perf prefixes.
    :param str control_profile_prefix: Control profile prefix.
    :param str candidate_profile_prefix: Candidate profile prefix.
    :param list[str] target_rows: Target-family selectors.
    :param ScoreResult result: Computed metrics.
    :return str: Markdown report.
    """

    lines = [
        "# HGDN Kernel Family Scoreboard",
        "",
        f"- family: `{family_name}`",
        f"- control perf prefixes: `{', '.join(control_perf_prefixes)}`",
        f"- candidate perf prefixes: `{', '.join(candidate_perf_prefixes)}`",
        f"- control profile prefix: `{control_profile_prefix}`",
        f"- candidate profile prefix: `{candidate_profile_prefix}`",
        f"- target rows: `{', '.join(target_rows) if target_rows else '(none)'}`",
        "",
        "## Thresholds",
        "",
        "| metric | value_ms |",
        "|---|---:|",
        f"| control_mean_ms | {result.control_mean_ms:.3f} |",
        f"| control_noise_ms | {result.control_noise_ms:.3f} |",
        f"| meaningful_win_ms | {result.meaningful_win_ms:.3f} |",
        f"| flat_band_ms | {result.flat_band_ms:.3f} |",
        "",
        "## Metrics",
        "",
        "| metric | value_ms |",
        "|---|---:|",
        f"| candidate_mean_ms | {result.candidate_mean_ms:.3f} |",
        f"| delta_ms | {result.delta_ms:.3f} |",
        f"| eager_profilerstep_delta_ms | {result.eager_delta_ms:.3f} |",
        f"| compiled_profilerstep_delta_ms | {result.compiled_delta_ms:.3f} |",
        f"| compile_specific_penalty_ms | {result.compile_specific_penalty_ms:.3f} |",
        f"| compiled_copy_tax_ms | {result.compiled_copy_tax_ms:.3f} |",
        f"| compiled_external_kernel_ms | {result.compiled_external_kernel_ms:.3f} |",
        f"| compiled_fxgraph_delta_ms | {result.compiled_fxgraph_delta_ms:.3f} |",
        f"| compiled_ddp_forward_delta_ms | {result.compiled_ddp_forward_delta_ms:.3f} |",
        f"| targeted_bucket_upper_bound_ms | {result.targeted_bucket_upper_bound_ms:.3f} |",
        "",
        "## Classification",
        "",
        f"- status: `{result.status}`",
        f"- rationale: {result.rationale}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    :return argparse.Namespace: Parsed args.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--input", dest="input_paths", action="append", required=True)
    parser.add_argument(
        "--control-perf-prefix",
        dest="control_perf_prefixes",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--candidate-perf-prefix",
        dest="candidate_perf_prefixes",
        action="append",
        required=True,
    )
    parser.add_argument("--control-profile-prefix", required=True)
    parser.add_argument("--candidate-profile-prefix", required=True)
    parser.add_argument(
        "--target-row",
        action="append",
        default=[],
        help="Exact row name or regex selector prefixed with 're:'.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    """Run the scoreboard.

    :return int: Exit code.
    """

    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="hgdn-scoreboard-"))
    try:
        perf_records: dict[str, PerfRecord] = {}
        profile_records: dict[str, dict[str, object]] = {}
        for raw in args.input_paths:
            extracted = extract_input(Path(raw), temp_root)
            perf_records.update(parse_perf_logs(extracted))
            profile_records.update(parse_profiles(extracted))
        control_perf_values = [
            find_perf(perf_records, prefix).step_ms
            for prefix in args.control_perf_prefixes
        ]
        candidate_perf_values = [
            find_perf(perf_records, prefix).step_ms
            for prefix in args.candidate_perf_prefixes
        ]
        control_eager = find_profile(
            profile_records, args.control_profile_prefix, compile_mode=False
        )
        candidate_eager = find_profile(
            profile_records, args.candidate_profile_prefix, compile_mode=False
        )
        control_compiled = find_profile(
            profile_records, args.control_profile_prefix, compile_mode=True
        )
        candidate_compiled = find_profile(
            profile_records, args.candidate_profile_prefix, compile_mode=True
        )
        result = build_score_result(
            control_perf_values=control_perf_values,
            candidate_perf_values=candidate_perf_values,
            control_eager=control_eager,
            candidate_eager=candidate_eager,
            control_compiled=control_compiled,
            candidate_compiled=candidate_compiled,
            target_rows=list(args.target_row),
        )
        report = render_markdown(
            family_name=args.family_name,
            control_perf_prefixes=list(args.control_perf_prefixes),
            candidate_perf_prefixes=list(args.candidate_perf_prefixes),
            control_profile_prefix=str(args.control_profile_prefix),
            candidate_profile_prefix=str(args.candidate_profile_prefix),
            target_rows=list(args.target_row),
            result=result,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
        print(report)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "control_mean_ms": result.control_mean_ms,
                    "candidate_mean_ms": result.candidate_mean_ms,
                    "delta_ms": result.delta_ms,
                    "meaningful_win_ms": result.meaningful_win_ms,
                    "flat_band_ms": result.flat_band_ms,
                    "compile_specific_penalty_ms": result.compile_specific_penalty_ms,
                    "compiled_copy_tax_ms": result.compiled_copy_tax_ms,
                    "targeted_bucket_upper_bound_ms": result.targeted_bucket_upper_bound_ms,
                },
                sort_keys=True,
            )
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
