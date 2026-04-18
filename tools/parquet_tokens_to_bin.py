"""Convert a parquet token export into a flat binary token file.

This helper is intentionally simple and readable. It relies on pyarrow rather
than maintaining a bespoke parquet parser in the repo.

Examples:
    python tools/parquet_tokens_to_bin.py fineweb_sample.parquet fineweb_sample.uint16.bin
    python tools/parquet_tokens_to_bin.py fineweb_sample.parquet fineweb_sample.int32.bin --dtype int32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PREFERRED_TOKEN_COLUMNS = (
    "tokens",
    "input_ids",
    "token_ids",
    "ids",
)


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyarrow is required to read parquet token exports. "
            "Install it with: pip install pyarrow"
        ) from exc
    return pa, pc, pq


def _is_integer_list_type(pa, arrow_type) -> bool:
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return pa.types.is_integer(arrow_type.value_type)
    if hasattr(pa.types, "is_fixed_size_list") and pa.types.is_fixed_size_list(arrow_type):
        return pa.types.is_integer(arrow_type.value_type)
    return False


def _find_token_column_name(parquet_file, pa) -> str:
    schema = parquet_file.schema_arrow
    available = set(schema.names)

    for name in PREFERRED_TOKEN_COLUMNS:
        if name in available:
            arrow_type = schema.field(name).type
            if pa.types.is_integer(arrow_type) or _is_integer_list_type(pa, arrow_type):
                return name

    integer_list_columns: list[str] = []
    integer_columns: list[str] = []
    for field in schema:
        if _is_integer_list_type(pa, field.type):
            integer_list_columns.append(field.name)
        elif pa.types.is_integer(field.type):
            integer_columns.append(field.name)

    if len(integer_list_columns) == 1:
        return integer_list_columns[0]
    if len(integer_columns) == 1:
        return integer_columns[0]

    raise ValueError(
        "Could not determine token column automatically. "
        f"Preferred names: {PREFERRED_TOKEN_COLUMNS}. "
        f"Schema fields: {schema.names}"
    )


def _flatten_arrow_column(pa, pc, column) -> np.ndarray:
    combined = column.combine_chunks()
    arrow_type = combined.type

    if pa.types.is_integer(arrow_type):
        values = combined
    elif _is_integer_list_type(pa, arrow_type):
        values = pc.list_flatten(combined)
    else:
        raise TypeError(f"Unsupported token column type: {arrow_type}")

    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.int64)


def extract_tokens(path: str | Path, *, column_name: str | None = None) -> np.ndarray:
    pa, pc, pq = _import_pyarrow()
    parquet_path = Path(path)
    parquet_file = pq.ParquetFile(parquet_path)

    resolved_column = column_name or _find_token_column_name(parquet_file, pa)
    table = parquet_file.read(columns=[resolved_column])
    tokens = _flatten_arrow_column(pa, pc, table[resolved_column])
    if tokens.ndim != 1:
        tokens = tokens.reshape(-1)
    return tokens


def _validate_output_range(tokens: np.ndarray, output_dtype: str) -> None:
    if tokens.size == 0:
        raise ValueError("Parquet file contained zero tokens")
    if tokens.min() < 0:
        raise ValueError(f"Token ids must be non-negative, got min={tokens.min()}")
    if output_dtype == "uint16" and tokens.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"Token ids do not fit uint16: max={tokens.max()}")


def _write_tokens(tokens: np.ndarray, output_path: Path, output_dtype: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_dtype == "uint16":
        tokens.astype(np.uint16, copy=False).tofile(output_path)
        return
    if output_dtype == "int32":
        tokens.astype(np.int32, copy=False).tofile(output_path)
        return
    raise ValueError(f"Unsupported output dtype: {output_dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_parquet", type=Path)
    parser.add_argument("output_bin", type=Path)
    parser.add_argument(
        "--dtype",
        default="uint16",
        choices=("uint16", "int32"),
        help="Output storage dtype for the flat token file.",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Optional explicit parquet column name. By default the script auto-detects a token column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = extract_tokens(args.input_parquet, column_name=args.column)
    _validate_output_range(tokens, args.dtype)
    _write_tokens(tokens, args.output_bin, args.dtype)

    print(
        "Extracted "
        f"{tokens.size:,} tokens from {args.input_parquet} "
        f"(min={int(tokens.min())}, max={int(tokens.max())})"
    )
    print(f"Wrote {args.output_bin} ({args.output_bin.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
