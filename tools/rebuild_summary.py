#!/usr/bin/env python3
"""Rebuild deterministic TSV and Markdown summaries from run artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_amplifier_lm.experiment import (
    collect_summary_rows,
    write_summary_markdown,
    write_summary_tsv,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Directory containing per-run subdirectories")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output TSV path (default: <root>/summary.tsv)",
    )
    ap.add_argument(
        "--md-out",
        type=str,
        default=None,
        help="Output Markdown path (default: <root>/summary.md)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional Markdown title (default: '<root-name> Summary')",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else (root / "summary.tsv")
    md_out = Path(args.md_out).resolve() if args.md_out else (root / "summary.md")
    title = args.title or f"{root.name} Summary"

    rows = collect_summary_rows(root)
    write_summary_tsv(rows, out)
    write_summary_markdown(rows, md_out, title=title)
    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Wrote Markdown summary to {md_out}")


if __name__ == "__main__":
    main()
