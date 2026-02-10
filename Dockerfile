# Example: CUDA 11.7 development with Ubuntu 22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /workspace

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential git wget curl libcrypt-dev

# Setup miniconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN ~/miniconda3/condabin/conda init

# Copy this folder into the container
COPY . .

SHELL ["/bin/bash", "-i", "-c"]
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN make clean_install
RUN make create_env
RUN conda activate ibp && make install_deps
RUN conda deactivate
RUN conda activate ibp
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV DGLBACKEND="pytorch"
RUN conda activate ibp && make install NVCC=$(which nvcc) GCC=$(which gcc) GPP=$(which g++)
RUN make download_dlrm
RUN conda activate ibp && make download_gnn
RUN echo "conda activate ibp" >> ~/.bashrc