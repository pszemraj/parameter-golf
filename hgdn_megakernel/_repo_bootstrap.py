"""Shared repo-root bootstrap for HGDN test/data entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_sys_path() -> Path:
    """Return the repo root and prepend it to `sys.path` once.

    :return Path: Repository root for the current checkout.
    """
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
