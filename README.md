# IBP
Invariant Bit Packing compression

## Install
```
conda install conda-forge::nvcomp==3.0.1
pushd ndzip
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_CUDA_COMPILER="/home/x_kamathak/miniforge3/envs/dgl-dev-gpu-117/bin/nvcc" -DCMAKE_C_COMPILER="/home/x_kamathak/miniforge3/envs/dgl-dev-gpu-117/bin/gcc" -DCMAKE_CXX_COMPILER="/home/x_kamathak/miniforge3/envs/dgl-dev-gpu-117/bin/g++" -DCMAKE_CUDA_HOST_COMPILER="/home/x_kamathak/miniforge3/envs/dgl-dev-gpu-117/bin/g++"
cmake --build build -j
popd
pip install -v -e .
```