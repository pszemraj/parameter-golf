"""Build script for the optional HGDN megakernel extension.

Usage:
  conda run -s --name pg python setup_hgdn_megakernel.py build_ext --inplace

Optional compile-time tuning:
  HGDN_THREADS=256 conda run -s --name pg python setup_hgdn_megakernel.py build_ext --inplace
  HGDN_GEMM_ATB_SPLIT_M_THRESHOLD=1024 conda run -s --name pg python setup_hgdn_megakernel.py build_ext --inplace
  HGDN_REC_V_TILE=16 conda run -s --name pg python setup_hgdn_megakernel.py build_ext --inplace
  HGDN_REC_CHUNK_T=16 conda run -s --name pg python setup_hgdn_megakernel.py build_ext --inplace
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

nvcc_args = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
]

threads = os.environ.get("HGDN_THREADS")
if threads:
    nvcc_args.append(f"-DHGDN_THREADS={int(threads)}")

split_m_threshold = os.environ.get("HGDN_GEMM_ATB_SPLIT_M_THRESHOLD")
if split_m_threshold:
    nvcc_args.append(f"-DHGDN_GEMM_ATB_SPLIT_M_THRESHOLD={int(split_m_threshold)}")

rec_v_tile = os.environ.get("HGDN_REC_V_TILE")
if rec_v_tile:
    nvcc_args.append(f"-DHGDN_REC_V_TILE={int(rec_v_tile)}")

rec_chunk_t = os.environ.get("HGDN_REC_CHUNK_T")
if rec_chunk_t:
    nvcc_args.append(f"-DHGDN_REC_CHUNK_T={int(rec_chunk_t)}")

setup(
    name="hgdn_megakernel_ext",
    ext_modules=[
        CUDAExtension(
            name="hgdn_megakernel_ext",
            sources=[SOURCE],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
