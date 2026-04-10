"""Utilities for structured profiler export and comparison.

This module gives the hybrid trainer and the local HGDN hotpath profiler a
shared machine-readable report format based on JSON and CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

HGDN_TRANSFER_BUCKETS = (
    "aten::copy_",
    "aten::mul",
    "gdn.qkv_conv_packed",
    "gdn.qkv_frontend_nct_cuda",
    "gdn.q_conv",
    "gdn.k_conv",
    "gdn.v_conv",
    "gdn.recurrence",
    "aten::convolution_backward",
    "aten::_conv_depthwise2d",
    "gdn.q_norm",
    "gdn.k_norm",
    "gdn.g_proj",
    "gdn.g_pointwise",
    "gdn.beta_proj",
    "gdn.output_gate_proj",
    "gdn.output_norm",
    "gdn.output_gate_mul",
)


@dataclass(frozen=True)
class ProfileRow:
    """Structured representation of one profiler row.

    :param str name: Event or range label.
    :param int count: Number of calls.
    :param float self_cpu_time_us: Self CPU time in microseconds.
    :param float cpu_time_total_us: Total CPU time in microseconds.
    :param float self_device_time_us: Self CUDA/device time in microseconds.
    :param float device_time_total_us: Total CUDA/device time in microseconds.
    :param int cpu_memory_bytes: Total CPU memory bytes.
    :param int self_cpu_memory_bytes: Self CPU memory bytes.
    :param int device_memory_bytes: Total device memory bytes.
    :param int self_device_memory_bytes: Self device memory bytes.
    """

    name: str
    count: int
    self_cpu_time_us: float
    cpu_time_total_us: float
    self_device_time_us: float
    device_time_total_us: float
    cpu_memory_bytes: int
    self_cpu_memory_bytes: int
    device_memory_bytes: int
    self_device_memory_bytes: int


def _event_float(event: Any, *names: str) -> float:
    """Fetch a float-like event attribute with alias fallback.

    :param Any event: Profiler event row.
    :param str names: Candidate attribute names in priority order.
    :return float: Attribute value or `0.0` if none exist.
    """
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _event_int(event: Any, *names: str) -> int:
    """Fetch an int-like event attribute with alias fallback.

    :param Any event: Profiler event row.
    :param str names: Candidate attribute names in priority order.
    :return int: Attribute value or `0` if none exist.
    """
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return int(value)
    return 0


def build_profile_rows(events: Iterable[Any]) -> list[ProfileRow]:
    """Convert profiler events into a structured row list.

    :param Iterable[Any] events: `prof.key_averages()` iterable.
    :return list[ProfileRow]: Structured row objects.
    """
    rows: list[ProfileRow] = []
    for event in events:
        rows.append(
            ProfileRow(
                name=str(getattr(event, "key", "")),
                count=_event_int(event, "count"),
                self_cpu_time_us=_event_float(event, "self_cpu_time_total"),
                cpu_time_total_us=_event_float(event, "cpu_time_total"),
                self_device_time_us=_event_float(
                    event,
                    "self_cuda_time_total",
                    "self_device_time_total",
                    "cuda_time_total",
                    "device_time_total",
                ),
                device_time_total_us=_event_float(
                    event,
                    "cuda_time_total",
                    "device_time_total",
                    "self_cuda_time_total",
                    "self_device_time_total",
                ),
                cpu_memory_bytes=_event_int(event, "cpu_memory_usage"),
                self_cpu_memory_bytes=_event_int(event, "self_cpu_memory_usage"),
                device_memory_bytes=_event_int(
                    event, "cuda_memory_usage", "device_memory_usage"
                ),
                self_device_memory_bytes=_event_int(
                    event, "self_cuda_memory_usage", "self_device_memory_usage"
                ),
            )
        )
    return rows


def _coalesce_profile_rows(rows: Sequence[ProfileRow]) -> list[ProfileRow]:
    """Merge duplicate profiler row names into a single structured row.

    PyTorch profiler exports can contain duplicate keys where one row carries the
    real device timing and a second row is all-zero. Coalescing here keeps the
    structured reports stable for exact-name comparison helpers.

    :param Sequence[ProfileRow] rows: Raw structured rows.
    :return list[ProfileRow]: Rows merged by exact event name.
    """
    merged: dict[str, ProfileRow] = {}
    order: list[str] = []
    for row in rows:
        existing = merged.get(row.name)
        if existing is None:
            merged[row.name] = row
            order.append(row.name)
            continue
        merged[row.name] = ProfileRow(
            name=row.name,
            count=max(existing.count, row.count),
            self_cpu_time_us=existing.self_cpu_time_us + row.self_cpu_time_us,
            cpu_time_total_us=existing.cpu_time_total_us + row.cpu_time_total_us,
            self_device_time_us=existing.self_device_time_us + row.self_device_time_us,
            device_time_total_us=existing.device_time_total_us
            + row.device_time_total_us,
            cpu_memory_bytes=max(existing.cpu_memory_bytes, row.cpu_memory_bytes),
            self_cpu_memory_bytes=max(
                existing.self_cpu_memory_bytes, row.self_cpu_memory_bytes
            ),
            device_memory_bytes=max(
                existing.device_memory_bytes, row.device_memory_bytes
            ),
            self_device_memory_bytes=max(
                existing.self_device_memory_bytes, row.self_device_memory_bytes
            ),
        )
    return [merged[name] for name in order]


def _sort_key_name(sort_by: str) -> str:
    """Map a profiler sort key to a structured row attribute.

    :param str sort_by: Requested sort key.
    :return str: `ProfileRow` field name.
    """
    aliases = {
        "self_cuda_time_total": "self_device_time_us",
        "self_device_time_total": "self_device_time_us",
        "cuda_time_total": "device_time_total_us",
        "device_time_total": "device_time_total_us",
        "self_cpu_time_total": "self_cpu_time_us",
        "cpu_time_total": "cpu_time_total_us",
    }
    return aliases.get(sort_by, sort_by)


def _percent(numerator: float, denominator: float) -> float:
    """Compute a percentage safely.

    :param float numerator: Numerator.
    :param float denominator: Denominator.
    :return float: Percentage, or `0.0` if the denominator is zero.
    """
    if denominator == 0.0:
        return 0.0
    return 100.0 * numerator / denominator


def find_profile_row(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Find one structured profiler row by exact name.

    :param dict[str, Any] report: Structured profile report.
    :param str name: Exact row name.
    :return dict[str, Any] | None: Matching row, if present.
    """
    for row in report["rows"]:
        if row["name"] == name:
            return row
    return None


def profile_row_ms(row: dict[str, Any] | None) -> float:
    """Extract self-device time in milliseconds from one structured row.

    :param dict[str, Any] row: Structured profiler row.
    :return float: Self-device time in milliseconds.
    """
    if row is None:
        return 0.0
    return float(row["self_device_time_us"]) / 1000.0


def profile_row_percent(row: dict[str, Any] | None) -> float:
    """Extract self-device percentage from one structured row.

    :param dict[str, Any] row: Structured profiler row.
    :return float: Self-device percentage.
    """
    if row is None:
        return 0.0
    return float(row["self_device_percent"])


def format_profile_bucket_cell(row: dict[str, Any] | None) -> str:
    """Format one profiler row as `ms / % / calls` for markdown tables.

    :param dict[str, Any] row: Structured profiler row.
    :return str: Compact cell string, or `-` when the row is absent.
    """
    if row is None:
        return "-"
    return (
        f"{profile_row_ms(row):.2f}ms / "
        f"{profile_row_percent(row):.2f}% / "
        f"{row['count']}"
    )


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous dict rows to CSV.

    :param Path path: Output CSV path.
    :param list[dict[str, Any]] rows: Rows to write.
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON-serializable payload with stable formatting.

    :param Path path: Output JSON path.
    :param Any payload: JSON-serializable payload.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json_rows(path: Path | None) -> list[dict[str, Any]]:
    """Load a JSON array-of-rows payload when the file exists.

    :param Path | None path: Optional JSON path.
    :return list[dict[str, Any]]: Parsed row list, or an empty list when absent.
    """
    if path is None or not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_boundary_audit_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """Load flattened HGDN boundary-audit rows from JSONL.

    :param Path | None path: Optional JSONL path.
    :return list[dict[str, Any]]: Flat per-tensor boundary rows.
    """
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for tensor in record["tensors"]:
            rows.append(
                {
                    "call_index": record["call_index"],
                    "boundary": record["boundary"],
                    "tensor": tensor["name"],
                    "dtype": tensor["dtype"],
                    "device": tensor["device"],
                    "shape": tuple(tensor["shape"]),
                    "stride": tuple(tensor["stride"]),
                    "contiguous": int(tensor["contiguous"]),
                }
            )
    return rows


def index_first_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: Sequence[str],
    order_field: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Index rows by key, keeping the smallest `order_field` entry per key.

    :param list[dict[str, Any]] rows: Flat row payloads.
    :param Sequence[str] key_fields: Fields that define one logical row key.
    :param str order_field: Field used to keep the earliest row.
    :return dict[tuple[Any, ...], dict[str, Any]]: Indexed first-row mapping.
    """
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key not in indexed or row[order_field] < indexed[key][order_field]:
            indexed[key] = row
    return indexed


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    aligns: Sequence[str] | None = None,
) -> str:
    """Render a markdown table from headers and row cells.

    :param Sequence[str] headers: Header cells.
    :param Iterable[Sequence[Any]] rows: Table row cells.
    :param Sequence[str] | None aligns: Markdown alignment markers per column.
    :raises ValueError: Raised when `aligns` length does not match `headers`.
    :return str: Markdown table with trailing newline.
    """
    if aligns is None:
        aligns = ["---"] * len(headers)
    if len(aligns) != len(headers):
        raise ValueError("aligns must match header count")
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(aligns)} |",
    ]
    for row in rows:
        lines.append(f"| {' | '.join(str(cell) for cell in row)} |")
    return "\n".join(lines) + "\n"


def build_profile_report(
    rows: Sequence[ProfileRow],
    *,
    sort_by: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable report from structured rows.

    :param Sequence[ProfileRow] rows: Structured profiler rows.
    :param str sort_by: Requested sort key.
    :param dict[str, Any] metadata: Optional extra metadata.
    :return dict[str, Any]: Serializable report dictionary.
    """
    rows = _coalesce_profile_rows(rows)
    sort_attr = _sort_key_name(sort_by)
    sorted_rows = sorted(
        rows,
        key=lambda row: getattr(row, sort_attr, 0.0),
        reverse=True,
    )
    total_self_cpu_us = sum(row.self_cpu_time_us for row in rows)
    total_self_device_us = sum(row.self_device_time_us for row in rows)
    payload_rows: list[dict[str, Any]] = []
    for row in sorted_rows:
        row_dict = asdict(row)
        row_dict["self_cpu_percent"] = _percent(row.self_cpu_time_us, total_self_cpu_us)
        row_dict["self_device_percent"] = _percent(
            row.self_device_time_us, total_self_device_us
        )
        payload_rows.append(row_dict)
    return {
        "metadata": {
            **(metadata or {}),
            "sort_by": sort_by,
            "sort_attr": sort_attr,
            "total_self_cpu_time_us": total_self_cpu_us,
            "total_self_device_time_us": total_self_device_us,
            "row_count": len(payload_rows),
        },
        "rows": payload_rows,
    }


def write_profile_report(
    output_dir: Path,
    *,
    report: dict[str, Any],
    stem: str = "key_averages",
) -> None:
    """Write JSON and CSV profiler artifacts.

    :param Path output_dir: Output directory.
    :param dict[str, Any] report: Structured report payload.
    :param str stem: Basename prefix, defaults to `key_averages`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{stem}.json", report)
    write_rows_csv(output_dir / f"{stem}.csv", report["rows"])


def load_profile_report(path: str | Path) -> dict[str, Any]:
    """Load a structured profile report from a directory or JSON file.

    :param str | Path path: Report directory or JSON file.
    :raises FileNotFoundError: If no supported report is found.
    :return dict[str, Any]: Structured report payload.
    """

    def _normalize_report(report: dict[str, Any]) -> dict[str, Any]:
        """Normalize a stored report back through the current schema builder.

        :param dict[str, Any] report: Raw serialized report payload.
        :return dict[str, Any]: Schema-normalized report.
        """
        rows = [
            ProfileRow(
                name=str(row["name"]),
                count=int(row["count"]),
                self_cpu_time_us=float(row["self_cpu_time_us"]),
                cpu_time_total_us=float(row["cpu_time_total_us"]),
                self_device_time_us=float(row["self_device_time_us"]),
                device_time_total_us=float(row["device_time_total_us"]),
                cpu_memory_bytes=int(row["cpu_memory_bytes"]),
                self_cpu_memory_bytes=int(row["self_cpu_memory_bytes"]),
                device_memory_bytes=int(row["device_memory_bytes"]),
                self_device_memory_bytes=int(row["self_device_memory_bytes"]),
            )
            for row in report["rows"]
        ]
        metadata = dict(report.get("metadata", {}))
        metadata.pop("row_count", None)
        metadata.pop("total_self_cpu_time_us", None)
        metadata.pop("total_self_device_time_us", None)
        metadata.pop("sort_attr", None)
        sort_by = metadata.pop("sort_by", "self_device_time_total")
        return build_profile_report(rows, sort_by=sort_by, metadata=metadata)

    report_path = Path(path)
    if report_path.is_dir():
        json_path = report_path / "key_averages.json"
        if json_path.is_file():
            return _normalize_report(json.loads(json_path.read_text(encoding="utf-8")))
        raise FileNotFoundError(f"No key_averages.json report found in {report_path}")
    if report_path.suffix == ".json":
        return _normalize_report(json.loads(report_path.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"Unsupported profile report path: {report_path}")
