import torch
# Need to import torch before CUDA library
import ibp_cuda

# Turn off decode messages by default
ibp_cuda.print_debug(False)

def print_debug(flag):
    ibp_cuda.print_debug(flag)

# Returns mask and bitval tensors
def preprocess(dataset):
    return ibp_cuda.preprocess(dataset)