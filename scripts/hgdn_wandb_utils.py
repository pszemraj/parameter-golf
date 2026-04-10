"""Shared W&B selection helpers for HGDN analysis scripts."""

from __future__ import annotations

from typing import Any

DEFAULT_PROJECT = "pg-hgdn-ablations"


def matches(run_name: str, exact_names: set[str], substrings: list[str]) -> bool:
    """Return whether a run name matches the provided exact or substring filters.

    :param str run_name: Candidate W&B run name.
    :param set[str] exact_names: Exact-match filters.
    :param list[str] substrings: Substring filters.
    :return bool: ``True`` when the run should be selected.
    """
    if exact_names and run_name in exact_names:
        return True
    if substrings and any(substr in run_name for substr in substrings):
        return True
    return not exact_names and not substrings


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop internal W&B config keys from a run config mapping.

    :param dict[str, Any] config: Raw W&B config mapping.
    :return dict[str, Any]: Public config entries only.
    """
    return {k: v for k, v in config.items() if not k.startswith("_")}
