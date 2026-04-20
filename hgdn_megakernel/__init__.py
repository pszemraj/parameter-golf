"""Repo-backed HGDN CUDA kernel helpers."""

from .hgdn_megakernel_binding import (
    device_report,
    extension_loaded,
    extension_status,
    hgdn_corekernel,
    hgdn_megakernel,
    rec_chunk_t_max,
    resolve_runtime_rec_chunk_t,
    run_from_gated_delta_net,
)

__all__ = [
    "device_report",
    "extension_loaded",
    "extension_status",
    "hgdn_corekernel",
    "hgdn_megakernel",
    "rec_chunk_t_max",
    "resolve_runtime_rec_chunk_t",
    "run_from_gated_delta_net",
]
