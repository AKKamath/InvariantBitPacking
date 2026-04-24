# Invariant Bit Packing
[![DOI](https://zenodo.org/badge/862131761.svg)](https://doi.org/10.5281/zenodo.18869046)

This repository contains the source code, profiling scripts, and workloads evaluated for Invariant Bit Packing (IBP) compression, introduced in the EuroSys 2026 paper titled "Reducing the GPU Memory Bottleneck with Lossless Compression for ML". IBP identifies and eliminates low-entropy, invariant bits _across_ sets of tensors, improving throughput by employing GPU-optimized decompression mechanisms, leveraging warp parallelism, low-overhead bit operations, and asynchronous GPU-optimized PCIe transfers. We provide easy-to-use APIs, showcasing them by adding IBP support to GNN training, as well as DLRM and LLM inference frameworks.

Full details of our implementation can be found in our [paper](https://akkamath.github.io/files/EuroSys26_IBP.pdf):
<pre>
<b>Reducing the GPU Memory Bottleneck with Lossless Compression for ML</b>
Aditya K Kamath, Arvind Krishnamurthy, Marco Canini, Simon Peter
<i>21st European Conference on Computer Systems (EuroSys), 2026</i>
DOI: https://doi.org/10.1145/3767295.3803595
</pre>

## Hardware and software requirements
Hardware:
* An NVIDIA A100 GPU.
* Minimum 300 GB CPU memory.
* 1 TB disk space.

Docker and NVIDIA Container Toolkit (installation instructions given below) are enough for software; all other software requirements are handled within the Docker container. The machine we evaluated had:
* Ubuntu 22.04 OS
* CUDA 11.7
* Python 3.9
* Pytorch 1.13.1

Parts of the work have also been tested with CUDA 12.1 and PyTorch 2.1, but various libraries need to be adjusted to ensure compatibility with these versions.


## Install
First download this repo and setup the submodules. Then download the requisite datasets. 
The GNN dataset requires DGL to be installed first, so we'll download that separately once DGL is installed.
```
git clone https://github.com/AKKamath/InvariantBitPacking.git
cd InvariantBitPacking
git submodule update --init --recursive

make download_dlrm # 16GB download
make download_llm # 8GB download
```

### Docker installation (Recommended)
We provide a docker image for IBP with all its dependencies installed. To access the docker image, you need to have [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your system. You can then launch the docker container and navigate to the folder containing the artifact, as follows:
```
# Build the local docker image [~1 hour]
docker build -t ibp-image -f Dockerfile .
# Run the docker image
docker run --gpus 0 \
  -e OMP_NUM_THREADS=32  -e MKL_NUM_THREADS=32 \
  -e OPENBLAS_NUM_THREADS=32 -e NUMEXPR_NUM_THREADS=32 \
  -e TORCH_NUM_THREADS=32 -e TORCH_INTEROP_THREADS=32 \
  -it -p 8181:8181 --ipc=host --cap-add=SYS_ADMIN ibp-image
```

### Manual installation
The Docker setup is recommended for ease of use. Manual installation can cause problems if the system's library versions mismatch with those expected. Use at your own risk.

```
# If conda not installed:
make install_miniconda
# Setup:
make create_env
conda activate ibp
# If system does not have CUDA 11.7 installed, the below installs it in conda.
make install_cuda
conda env config vars set CUDA_HOME="${CONDA_PREFIX}"
conda env config vars set CUDA_TOOLKIT_ROOT_DIR="${CONDA_PREFIX}"
conda env config vars set CUDACXX="${CONDA_PREFIX}/bin/nvcc"
conda env config vars set PATH="$CONDA_PREFIX/bin:$PATH"
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$(pwd)/ndzip/build:$LD_LIBRARY_PATH
conda env config vars set DGL_HOME=$(pwd)/workloads/DGL-IBP
# Otherwise if CUDA 11.7 is already installed, continue from here.
make install_deps
conda deactivate
conda activate ibp
make install
pip install torchdata==0.7.0
make download_gnn
```

## Replicating experiments
Before running any experiment ensure you are in the ibp conda environment. This can be ensured by running the following command:
```
conda activate ibp
```

### Run commands
```
# Main experiments:
make nvcomp_comparison; # Tables 1, 2, 3
make gnn; # Figure 8
make dlrm; # Figure 10
make llm; # Figure 11
# Other experiments:
make invariance; # Table 4
make decomp_thput; # Figure 7
```

### Results
To generate the graphs for the main experiments, you can run the following commands. The generated graphs will be in the results/ folder.
```
# Main experiments:
make plot_gnn; # Figure 8
make plot_dlrm; # Figure 10
make plot_llm; # Figure 11
```

You can also run ```cat {filename}``` to output the results to the terminal. This can then be copy-pasted into your favorite graph plotting tool (e.g., Excel, Google Sheets, matplotlib).

Main experiments:
* Tables 1, 2 - results/nvcomp_comparison.log
* Figure 8 - results/gnn_perf.log
* Figure 10 - results/dlrm_perf.log
* Figure 11 - results/llm_latency.log

Other experiments:
* Table 4: results/invariance.out
* Figure 7: results/decomp_thput.out

# API Documentation
We provide both high-level PyTorch and low-level CUDA API for integration with projects.

## PyTorch
After installing IBP, you can import it into your program in the following manner:
```
import torch
import ibp
```
Make sure torch is imported before IBP, otherwise you will get errors.
The API for IBP is as follows:
```Python
# Whether to output IBP debug statements.
ibp.print_debug(flag: bool)

# Preprocess the provided dataset. The dataset is expected to be a 2D Tensor [num_vecs x vec_size]
# You can reshape/view your tensor if it is higher dimensional. Threshhold is the invariant percentage to fix.
# If None, we sweep over 0.7 to 0.95 to find a good threshold.
# Returns: mask and bitval GPU tensors
ibp.preprocess(dataset: torch.Tensor, threshold: float | None = None)

# Preprocess the provided dataset using K-means. Specify the number of centroids to use.
# Returns: mask and bitval 2D GPU tensors.
ibp.preprocess_kmeans(dataset: torch.Tensor, centroids: int, threshold: float | None = None)

# Computes the compressed size of a dataset. Non-blocking call.
# This function calculates the size of each element of the compressed dataset.
# Optionally, an index array can be provided to specify which vectors to consider.
# Non-blocking call. Synchronize CUDA before accessing output tensor.
# index_array_ Optional tensor specifying the indices of the vectors to be considered.
# Returns: A GPU tensor representing the compressed size of each element of the dataset.
ibp.get_compress_size(dataset: torch.Tensor, mask: torch.Tensor, bitval: torch.Tensor, index_arr: torch.Tensor | None = None)

# Compresses dataset in-place. Used for CPU-side compression.
# Optionally, an index array can be provided to specify which vectors to consider.
# Returns: A GPU tensor bitmask marking compressed tensors
ibp.compress_inplace(dataset: torch.Tensor, mask: torch.Tensor, bitval: torch.Tensor, index_arr: torch.Tensor | None = None)

# Fetches and decompresses compressed data into GPU memory.
# comp_len provides an estimated compressed size per tensor, which can help improve performance if provided.
# Optionally, an index array can be provided to specify which vectors to consider.
# Returns: A GPU tensor with the indexed tensors decompressed.
ibp.decompress_fetch(comp_dataset: torch.Tensor, mask: torch.Tensor, bitval: torch.Tensor, bitmask: torch.Tensor, device: torch.Device, comp_len: int | None = None,\
                     index_arr: torch.Tensor | None = None)
```

# Source code and repository structure
Our respository has the following folders, with contents as described:
* [ibp](ibp/): Contains the PyTorch module functions.
* [include](include/): Contains the header-only CUDA files for IBP functionality.
* [scripts](scripts/): Contains the plotting/extraction scripts for results from IBP evaluation.
* [src](src/): Contains the source code for the PyTorch module, converting the high-level Python calls to low-level CUDA functions.
* [tests](tests/): Contains scripts for some of the IBP evaluations.
* [workloads](workloads/): Contains submodules for GNN, DLRM, and LLM frameworks used during evaluation.


# Citation
If you use our work, please cite our paper:
<pre>
@inproceedings{IBP:Eurosys:2026,
  author = {Kamath, Aditya K and Krishnamurthy, Arvind and Canini, Marco and Peter, Simon},
  title = {Reducing the GPU Memory Bottleneck with Lossless Compression for ML},
  year = {2026},
  isbn = {9798400722127},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3767295.3803595},
  doi = {10.1145/3767295.3803595},
  abstract = {Machine learning (ML) training and inference often process data sets far exceeding GPU memory capacity, forcing them to rely on PCIe for on-demand tensor transfers, causing critical transfer bottlenecks. Lossy compression has been proposed to relieve bottlenecks but introduces workload-dependent accuracy loss, making it complex or even prohibitive to use in existing ML deployments.We explore lossless compression as an alternative that avoids this deployment complexity. We identify where lossless compression can be integrated into ML pipelines while minimizing interference with GPU execution. Based on our findings, we introduce Invariant Bit Packing (IBP), a novel lossless compression algorithm designed to minimize data transfer time for ML. IBP identifies and eliminates invariant bits across groups of tensors, improving throughput through GPU-optimized decompression that leverages warp parallelism, low-overhead bit operations, and asynchronous PCIe transfers. We provide easy-to-use APIs, showcasing them by adding IBP support to GNN training, as well as DLRM and LLM inference frameworks. IBP achieves, on average, 74\% faster GNN training, 180\% faster DLRM embedding lookup, and 25\% faster LLM inference.},
  booktitle = {Proceedings of the 21st European Conference on Computer Systems},
  pages = {899–918},
  numpages = {20},
  keywords = {lossless compression, GPU systems, PCIe bottleneck, data movement, machine learning systems, tensor compression, GNN, DLRM, LLM inference},
  location = {McEwan Hall/The University of Edinburgh, Edinburgh, Scotland UK},
  series = {EUROSYS '26}
}
</pre>
