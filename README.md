# Invariant Bit Packing
Invariant Bit Packing compression

## Hardware and software requirements
TODO

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
  -e OMP_NUM_THREADS=4  -e MKL_NUM_THREADS=4 \
  -e OPENBLAS_NUM_THREADS=4 -e NUMEXPR_NUM_THREADS=4 \
  -e TORCH_NUM_THREADS=4 -e TORCH_INTEROP_THREADS=4 \
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
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Otherwise if CUDA 11.7 is already installed, continue from here.
make install_deps
conda deactivate
conda activate ibp
make install
pip install torchdata==0.7.0
make download_gnn
```

## Running experiments
Before running any experiment ensure you are in the ibp conda environment. This can be ensured by running the following command:
```
conda activate ibp
```

### Run commands
```
# Main experiments:
make nvcomp_comparison; # Tables 1 and 2
make gnn; # Figure 8
make dlrm; # Figure 9
make llm; # Figure 10
# Other experiments:
make invariance; # Table 3
make decomp_thput; # Figure 7
```

### Results
You can run ```cat {filename}``` to output the results to the terminal. This can then be copy-pasted into your favorite graph plotting tool (e.g., Excel, Google Sheets, matplotlib).

Main experiments:
* Tables 1, 2 - results/nvcomp_comparison.log
* Figure 8 - results/gnn_perf.log
* Figure 9 - results/dlrm_perf.log
* Figure 10 - results/llm_latency.log

Other experiments:
* Table 3: results/invariance.out
* Figure 7: results/decomp_thput.out
