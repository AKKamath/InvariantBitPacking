# Invariant Bit Packing
[![DOI](https://zenodo.org/badge/862131761.svg)](https://doi.org/10.5281/zenodo.18869046)

This README discusses the IBP library. To replicate our paper's experiments check our [Experiment README](Experiments.md).

This repository contains the source code, profiling scripts, and workloads evaluated for Invariant Bit Packing (IBP) compression, introduced in the EuroSys 2026 paper titled "Reducing the GPU Memory Bottleneck with Lossless Compression for ML". 

IBP identifies and eliminates low-entropy, invariant bits _across_ sets of tensors, improving throughput by employing GPU-optimized decompression mechanisms, leveraging warp parallelism, low-overhead bit operations, and asynchronous GPU-optimized PCIe transfers. We provide easy-to-use APIs, showcasing them by adding IBP support to GNN training, as well as DLRM and LLM inference frameworks.


Full details of our implementation can be found in our [paper](https://akkamath.github.io/files/EuroSys26_IBP.pdf):
<pre>
<b>Reducing the GPU Memory Bottleneck with Lossless Compression for ML</b>
Aditya K Kamath, Arvind Krishnamurthy, Marco Canini, Simon Peter
<i>21st European Conference on Computer Systems (EuroSys), 2026</i>
DOI: https://doi.org/10.1145/3767295.3803595
</pre>

## Hardware and software requirements
Hardware:
* NVIDIA Ampere+ GPU, for async memory instructions

Software:
* CUDA 11+
* Python 3.9+
* PyTorch with CUDA support

In setup.py you may need to add flags for newer GPUs. For example for Hopper you can uncomment in line 21/22:
```python
#cc_flag.append("-gencode")
#cc_flag.append("arch=compute_90,code=sm_90")
```

## Install
```
git clone https://github.com/AKKamath/InvariantBitPacking.git
cd InvariantBitPacking
pip install torch # Make sure torch is already installed.
pip install -v . --no-build-isolation
```

You can then use ibp in your Python files by including it, as detailed below.

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

# Computes the compressed size of a dataset.
# Non-blocking call. Synchronize CUDA before accessing output tensor.
# This function calculates the size of each element of the compressed dataset.
# index_array: Optional tensor specifying the indices of the vectors to be considered.
# Returns: A GPU tensor representing the compressed size of each element of the dataset.
ibp.get_compress_size(dataset: torch.Tensor, mask: torch.Tensor, bitval: torch.Tensor, index_arr: torch.Tensor | None = None)

# Computes the average compressed size for each tensor in the dataset.
# Non-blocking call. Synchronize CUDA before accessing output tensor.
# index_array Optional tensor specifying the indices of the vectors to be considered.
# Returns: A GPU tensor representing the compressed size of each element of the dataset.
ibp.get_single_compress_len(dataset: torch.Tensor, mask: torch.Tensor, bitval: torch.Tensor, index_arr: torch.Tensor | None = None)

# Compresses dataset in-place. Used for CPU-side compression.
# Optionally, an index array can be provided to specify which vectors to consider.
# Returns: A GPU tensor bitmask marking compressed tensors
ibp.compress_inplace(dataset: torch.Tensor, mask: torch.Tensor, bitval: torch.Tensor, index_arr: torch.Tensor | None = None)

# Fetches and decompresses compressed data into GPU memory.
# Optionally:
#   comp_len provides an estimated compressed size per tensor, which can help improve performance if provided.
#   index_arr which is an array of indices can be provided to specify which vectors to consider.
#   output_tensor can be provided to store the decompressed tensors into.
# Returns: A GPU tensor with the indexed tensors decompressed.
ibp.decompress_fetch(comp_dataset: torch.Tensor, mask: torch.Tensor, bitval: torch.Tensor, bitmask: torch.Tensor, \
                     device: torch.Device, comp_len: int | None = None, \
                     index_arr: torch.Tensor | None = None, output_tensor: torch.Tensor | None = None)
```

### Example usage
Here we show exactly how IBP could be integrated, taking an example of FlexGen. You can see the real changes in [pytorch_backend.py](https://github.com/AKKamath/InfiniGen-IBP/blob/main/speedup/flexgen/original/pytorch_backend.py) and [flex_gemma.py](https://github.com/AKKamath/InfiniGen-IBP/blob/main/speedup/flexgen/original/flex_gemma.py) (`CTRL + F ibp` to see them). The real changes are a bit more complicated than shown here as they have some fine-tuning to maximize the performance gains from compression.

First, we add the preprocess and compress functions to FlexGen's TorchTensor wrapper class. Later when we've loaded the weights to the CPU, we call them:

```diff
import torch
+ import ibp
...
# Add preprocess and compress functions to FlexGen's TorchTensor class
class TorchTensor:
    ...
+   def ibp_preprocess(self):
+       self.mask, self.bitval = ibp.preprocess(self.data.view(torch.int64))
+
+   def ibp_compress(self):
+       self.bitmask = ibp.compress_inplace(self.data.view(torch.int64), self.mask, self.bitval)
+       self.comp_len = ibp.get_single_compress_len(self.data.view(torch.int64), self.mask, self.bitval)
...
# Then when initializing the weights, call these to compress the weights
def init_weight_list(weight_specs, policy, env):
    ...
+   # Stored in CPU and IBP turned on
+   if home == env.cpu and policy.use_ibp:
+      # For simplicity, we only compress 2D tensors here
+      # In reality, we can view/reshape larger dimension to 2D to compress them as well.
+      if len(weight.data.shape) == 2:
+          weight.ibp_preprocess()
+          weight.ibp_compress()

```

Notice that we change the view of the tensor to torch.int64; IBP operates on bits so the datatype of the input tensor doesn't matter as long as it's the same when both compressing and decompressing. By using int64, we use 8-byte chunks which reduces the 'participation bit' headers in the compressed tensors. We get back our float values losslessly when decompressing.
Below we see how data is decompressed later when transferring from the CPU.

```diff
def general_copy(dst: TorchTensor, dst_indices: Tuple[slice],
                 src: TorchTensor, src_indices: Tuple[slice]):
    ...
    src_tensor = src.data[src_indices] if src_indices else src.data
    dst_tensor = dst.data[dst_indices] if dst_indices else dst.data
+   ibp.decompress_fetch(src_tensor.view(torch.int64), src.mask, src.bitval,
+                        src.bitmask, dst.device.dev, output_tensor=dst_tensor.view(torch.int64),
+                        comp_len=src.comp_len)
-   dst_tensor.copy_(src_tensor, non_blocking=True)
```
IBP acts as a drop-in replacement for the copy operation which moves the data from CPU to GPU. Within this one function we copy and decompress simultaneously. We pass a int64 view of the destination tensor, so that it matches the src. The actual dst_tensor is bfloat16, and will have those values properly.

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
