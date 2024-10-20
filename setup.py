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
include_dirs = [
    "include/",
    "include/preproc/",
    "include/misc/",
    "include/compress/",
]
include_files = [
    "include/ibp_helpers.cuh",
    "include/decompress/ibp_decompress_kernel.cuh",
    "include/decompress/ibp_decompress_host.cuh",
    "include/decompress/ibp_decompress_dev.cuh",
    "include/preproc/ibp_preproc_kmeans.cuh",
    "include/preproc/ibp_preproc_host.cuh",
    "include/preproc/ibp_preproc_kernels.cuh",
    "include/ibp_dev_func.cuh",
    "include/misc/ibp_misc_dev.cuh",
    "include/misc/ibp_misc_kernels.cuh",
    "include/compress/ibp_compress_dev.cuh",
    "include/compress/ibp_compress_kernel.cuh",
    "include/compress/ibp_compress_host.cuh",
]
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    "-lineinfo",
]

ext_modules.append(
    CUDAExtension(
        name="ibp_cuda",
        sources=sources,
        include_dirs=[Path(this_dir) / i for i in include_dirs],
        depends=sources + include_files,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags + cc_flag,
        },
    )
)

setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "include",
            "ibp.egg-info",
        )
    ),
    description="Invariant Bit Packing",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    depends=sources + include_files,
    install_requires=[
        "packaging",
        "ninja",
    ],
)