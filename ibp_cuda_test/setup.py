import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
repo_root = this_dir.parent

PACKAGE_NAME = "ibp_cuda_test"

cc_flag = [
    "-gencode", "arch=compute_70,code=sm_70",
    "-gencode", "arch=compute_80,code=sm_80",
]

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-lineinfo",
]

include_dirs = [
    repo_root / "include",
    repo_root / "include/preproc",
    repo_root / "include/misc",
    repo_root / "include/compress",
    repo_root / "ndzip/include",
    Path(os.environ["CONDA_PREFIX"]) / "include",
]

sources = [str(repo_root / "src/compress_test.cu")]

include_files = [
    str(repo_root / "include/ibp_helpers.cuh"),
    str(repo_root / "include/decompress/ibp_decompress_kernel.cuh"),
    str(repo_root / "include/decompress/ibp_decompress_host.cuh"),
    str(repo_root / "include/decompress/ibp_decompress_dev.cuh"),
    str(repo_root / "include/preproc/ibp_preproc_kmeans.cuh"),
    str(repo_root / "include/preproc/ibp_preproc_host.cuh"),
    str(repo_root / "include/preproc/ibp_preproc_kernels.cuh"),
    str(repo_root / "include/ibp_dev_func.cuh"),
    str(repo_root / "include/misc/ibp_misc_dev.cuh"),
    str(repo_root / "include/misc/ibp_misc_kernels.cuh"),
    str(repo_root / "include/misc/compress_test.cuh"),
    str(repo_root / "include/compress/ibp_compress_dev.cuh"),
    str(repo_root / "include/compress/ibp_compress_kernel.cuh"),
    str(repo_root / "include/compress/ibp_compress_host.cuh"),
]

ext_modules = [
    CUDAExtension(
        name="ibp_cuda_test",
        sources=sources,
        include_dirs=include_dirs,
        libraries=["nvcomp", "ndzip-cuda", "ndzip"],
        library_dirs=[str(repo_root / "ndzip/build")],
        depends=sources + include_files,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags + cc_flag,
        },
    )
]

setup(
    name=PACKAGE_NAME,
    packages=[],
    description="Invariant Bit Packing - benchmark extension (nvcomp/ndzip comparisons)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    depends=sources + include_files,
    install_requires=[
        "packaging",
        "ninja",
        "torch",
    ],
)
