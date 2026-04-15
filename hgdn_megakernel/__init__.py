"""Repo-backed HGDN megakernel helpers."""

from .hgdn_megakernel_binding import (
    device_report,
    extension_loaded,
    extension_status,
    hgdn_megakernel,
    run_from_gated_delta_net,
)

__all__ = [
    "device_report",
    "extension_loaded",
    "extension_status",
    "hgdn_megakernel",
    "run_from_gated_delta_net",
]
