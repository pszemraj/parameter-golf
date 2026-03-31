"""Shared loader for optional ALlama C++/CUDA benchmark kernels.

This keeps the extension build contract consistent across standalone benchmarks
and any opt-in model-path integrations.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BUILD_DIR = REPO_ROOT / "runs_allama_validation" / "torch_extensions"
EXTENSION_SOURCES = (
    REPO_ROOT / "csrc" / "allama_ops.cpp",
    REPO_ROOT / "csrc" / "allama_residual_scale_rms_norm.cu",
    REPO_ROOT / "csrc" / "allama_attention_prep.cu",
)


@lru_cache(maxsize=None)
def load_allama_cpp_extension(build_dir: Optional[str] = None):
    """Build and import the optional ALlama C++/CUDA extension once.

    :param Optional[str] build_dir: Override build cache directory.
    :return Any: Imported extension module.
    """
    resolved_build_dir = (
        Path(build_dir)
        if build_dir is not None
        else Path(
            os.environ.get("ALLAMA_CPP_BUILD_DIR", str(DEFAULT_BUILD_DIR))
        ).expanduser()
    )
    resolved_build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="allama_cpp_ops",
        sources=[str(path) for path in EXTENSION_SOURCES],
        build_directory=str(resolved_build_dir),
        extra_cuda_cflags=["--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )
