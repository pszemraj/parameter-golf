"""Shared local environment helpers for repo scripts."""

from __future__ import annotations

import os


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag.

    :param str name: Environment variable name.
    :param bool default: Default value when unset.
    :return bool: Parsed boolean flag.
    """
    return bool(int(os.environ.get(name, "1" if default else "0")))
