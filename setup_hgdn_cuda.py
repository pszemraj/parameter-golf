"""Build script for the optional HGDN CUDA extension.

Usage:
  python setup_hgdn_cuda.py build_ext --inplace
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
SOURCES = [
    str(ROOT / "hgdn_cuda" / "csrc" / "binding.cpp"),
    str(ROOT / "hgdn_cuda" / "csrc" / "frontend_kernel.cu"),
    str(ROOT / "hgdn_cuda" / "csrc" / "output_kernel.cu"),
]

setup(
    name="hgdn_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="hgdn_cuda_ext",
            sources=SOURCES,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
