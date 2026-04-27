#!/usr/bin/env python3
"""Shared CLI helpers for HGDN Python and shell utilities.

:param None: This module exposes subcommands through ``argparse``.
:return None: Process exit code indicates command success.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import tomllib
from pathlib import Path
from typing import Any

from _repo_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()


def parse_bool_flag(value: str) -> bool:
    """Parse a shell-style boolean flag.

    :param str value: Input string, typically ``0``/``1`` or ``true``/``false``.
    :raises argparse.ArgumentTypeError: Raised when the flag is not recognized.
    :return bool: Parsed boolean value.
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected one of 0/1/true/false/yes/no/on/off, got {value!r}"
    )


def load_toml_env(path: Path, *, alias_aware: bool) -> dict[str, str]:
    """Load environment-style values from one TOML file.

    :param Path path: TOML file to read.
    :param bool alias_aware: Whether to follow the repo's ``alias`` indirection.
    :return dict[str, str]: Flattened environment mapping.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    alias = data.get("alias")
    if alias_aware and alias is not None:
        return load_toml_env((path.parent / alias).resolve(), alias_aware=True)
    env = data.get("env", data)
    merged: dict[str, str] = {}
    for key, value in env.items():
        if isinstance(value, bool):
            value = "1" if value else "0"
        elif isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, bool):
                    items.append("1" if item else "0")
                else:
                    items.append(str(item))
            value = ",".join(items)
        merged[str(key)] = str(value)
    return merged


def require_py7zr() -> Any:
    """Import and return ``py7zr``.

    :raises RuntimeError: Raised when ``py7zr`` is missing.
    :return Any: Imported ``py7zr`` module.
    """
    try:
        import py7zr
    except ModuleNotFoundError as exc:  # pragma: no cover - shell-helper runtime guard
        raise RuntimeError(
            "py7zr is required; install it with `python -m pip install py7zr`"
        ) from exc
    return py7zr


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON document with stable formatting.

    :param Path path: Output path.
    :param Any payload: JSON payload to serialize.
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def cmd_module_exists(args: argparse.Namespace) -> int:
    """Check whether one Python module is importable.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    return 0 if importlib.util.find_spec(args.module) is not None else 1


def cmd_create_7z(args: argparse.Namespace) -> int:
    """Create one ``.7z`` archive from a source path.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    py7zr = require_py7zr()
    archive_output = args.archive_output
    source_path = args.source_path
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    archive_output.unlink(missing_ok=True)
    with py7zr.SevenZipFile(archive_output, "w") as archive:
        archive.writeall(source_path, arcname=source_path.name)
    return 0


def cmd_load_env(args: argparse.Namespace) -> int:
    """Merge TOML ``[env]`` sections and print ``KEY=VALUE`` lines.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Shell-style exit code.
    """
    merged: dict[str, str] = {}
    for path in args.path:
        merged.update(load_toml_env(path, alias_aware=args.alias_aware))
    for key, value in merged.items():
        print(f"{key}={value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    :return argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Shared HGDN shell-helper utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    module_exists = subparsers.add_parser(
        "module-exists", help="return success when one module is importable"
    )
    module_exists.add_argument("--module", required=True)
    module_exists.set_defaults(func=cmd_module_exists)

    create_7z = subparsers.add_parser("create-7z", help="create one .7z archive")
    create_7z.add_argument("--archive-output", type=Path, required=True)
    create_7z.add_argument("--source-path", type=Path, required=True)
    create_7z.set_defaults(func=cmd_create_7z)

    load_env = subparsers.add_parser(
        "load-env", help="merge TOML [env] sections and print KEY=VALUE lines"
    )
    load_env.add_argument("--alias-aware", action="store_true")
    load_env.add_argument("--path", type=Path, nargs="+", required=True)
    load_env.set_defaults(func=cmd_load_env)

    return parser


def main() -> int:
    """Run the shared helper CLI.

    :return int: Shell-style exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
