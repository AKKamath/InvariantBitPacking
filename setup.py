import os
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "ibp"

cmdclass = {}
ext_modules = []

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_70,code=sm_70")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")

repo_dir = Path(this_dir).parent
sources = [
    "src/ibp_api.cu",
]
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    "-lineinfo",
]
include_dirs = [
    Path(this_dir) / "include",
]

ext_modules.append(
    CUDAExtension(
        name="ibp_cuda",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags + cc_flag,
        },
        include_dirs=include_dirs,
    )
)

setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "include",
        )
    ),
    description="Invariant Bit Packing",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "packaging",
        "ninja",
    ],
)