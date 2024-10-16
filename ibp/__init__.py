import torch
# Need to import torch before CUDA library
import ibp_cuda

# Turn off decode messages by default
ibp_cuda.print_debug(False)

def print_debug(flag):
    ibp_cuda.print_debug(flag)

# Returns mask and bitval tensors
def preprocess(dataset, threshold=None):
    return ibp_cuda.preprocess(dataset, threshold)

def get_compress_size(dataset, mask, bitval, index_arr=None, compress_total=None):
    return ibp_cuda.get_compress_size(dataset, mask, bitval, index_arr, compress_total)

def compress_inplace(dataset, mask, bitval, index_arr=None):
    return ibp_cuda.compress_inplace(dataset, mask, bitval, index_arr)

def compress(dataset, mask, bitval, index_arr=None):
    return ibp_cuda.compress(dataset, mask, bitval, index_arr)