# Invariant Bit Packing
Invariant Bit Packing compression

## Hardware and software requirements
TODO

## Install
First setup git submodules in this repo.
```
git submodule init
git submodule update --init --recursive
```

### Docker installation (Recommended)
We provide a docker image for IBP with all its dependencies installed. To access the docker image, you need to have [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your system. You can then launch the docker container and navigate to the folder containing the artifact, as follows:
```
# docker build -t ibp-image-121 -f Dockerfile-121 --progress=plain .
# Build the local docker image
docker build -t ibp-image .
# Run the docker image
docker run --gpus all -it \
  -p 8181:8181 --rm --ipc=host --cap-add=SYS_ADMIN ibp-image
```


### Manual installation
The Docker setup is recommended for ease of use. Manual installation can have problems if the system's library versions mismatch with those expected. Use at your own risk.
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
conda env config vars set PATH=$CONDA_PREFIX/bin:$PATH
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Otherwise if CUDA 11.7 is already installed, continue from here.
make install_deps
conda deactivate
conda activate ibp
make install
```

## Running experiments
Before running any experiment ensure you are in the ibp conda environment. This can be ensured by running the following command:
```
conda activate ibp
```

### Run commands
```
make nvcomp_comparison; # Tables 1 and 2
make invariance; # Table 3
make decomp_thput; # Figure 7
```

### Results
You can run ```cat {filename}``` to output the results to the terminal. The results are in:
```
Tables 1, 2: results/nvcomp_comparison.log
Table 3: results/invariance.out
Figure 7: results/decomp_thput.out
```