"""Build script for the optional HGDN megakernel extension.

Usage:
  conda run -s --name pg python setup_hgdn_megakernel.py build_ext --inplace
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
SOURCE = str(ROOT / "hgdn_megakernel" / "hgdn_megakernel.cu")


def maybe_set_arch_list() -> None:
    """Set a reasonable default `TORCH_CUDA_ARCH_LIST` when unset."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    arches = {f"{major}.{minor}"}
    if bool(int(os.environ.get("HGDN_MEGAKERNEL_INCLUDE_H100", "0"))):
        arches.add("9.0")
    if bool(int(os.environ.get("HGDN_MEGAKERNEL_INCLUDE_H100A", "0"))):
        arches.add("9.0a")
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(arches))


maybe_set_arch_list()

setup(
    name="hgdn_megakernel_ext",
    ext_modules=[
        CUDAExtension(
            name="hgdn_megakernel_ext",
            sources=[SOURCE],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
