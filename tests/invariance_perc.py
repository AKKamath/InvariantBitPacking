import torch
import ibp as ibp
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
import load_graph as ld

def pin_inplace(tensor):
    try:
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        assert tensor.is_pinned()
    except Exception as e:
        print(f"Failed to pin tensor: {e}")
    return tensor

def ibp_ify(dataset):
    mask, bitval = ibp.preprocess(dataset)
    sizes = ibp.get_compress_size(dataset, mask, bitval)
    torch.cuda.synchronize()
    # Add 3 because IBP adds 3% metadata
    return (1 - (torch.sum(sizes) / (dataset.element_size() * dataset.nelement()))) * 100 + 3

def bitpack(dataset: torch.Tensor):
    dim1, dim2 = 0, 1
    tensor1 = dataset.numpy()
    tensor1 = tensor1.view('uint32')
    tensor1 = tensor1.astype('int64')
    tensor = torch.from_numpy(tensor1)
    min_values, _ = torch.min(tensor, dim=dim1)
    max_values, _ = torch.max(tensor, dim=dim1)
    # Typecast up to avoid integer overflow
    max_values = max_values.to(torch.int64)
    min_values = min_values.to(torch.int64)
    diff = max_values - min_values + 1
    diff = torch.ceil(torch.log2(diff))
    diff = torch.nansum(diff)
    return (1 - diff / (tensor.shape[dim2] * 32)) * 100

def bitpack_group(dataset: torch.Tensor):
    dim1, dim2 = 0, 1
    tensor_size = 100
    #tensor1 = dataset.view(torch.int32)
    sums = 0
    val = dataset.shape[dim1] // tensor_size
    #print(val)
    for i in range(val):
        tensor1 = dataset[i * tensor_size : (i + 1) * tensor_size].numpy()
        #print(tensor1)
        tensor1 = tensor1.view('uint32')
        #print(tensor1)
        tensor1 = tensor1.astype('int64')
        #print(tensor1)
        tensor1 = torch.from_numpy(tensor1)
        #print(tensor1)
        tensor = tensor1
        #print(tensor.shape)
        min_values, _ = torch.min(tensor, dim=dim1)
        max_values, _ = torch.max(tensor, dim=dim1)
        max_values = max_values.to(torch.int64)
        min_values = min_values.to(torch.int64)
        #print(min_values.shape)
        #print(max_values.shape)
        #print(min_values, max_values)
        diff = max_values - min_values + 1
        diff = torch.ceil(torch.log2(diff))
        #print(diff)
        sums += torch.nansum(diff)
    #print((1 - sums / (val * dataset.shape[1] * 32)) * 100)
    return (1 - sums / (val * dataset.shape[dim2] * 32)) * 100

#ibp.print_debug(True)
datasets = sys.argv[1].split()
comp_size = {}
bitp_size = {}
bpgp_size = {}
for dataset in datasets:
    g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", dataset)
    print(dataset)
    comp_size[dataset] = ibp_ify(features)
    bitp_size[dataset] = bitpack(features)
    bpgp_size[dataset] = bitpack_group(features)

print("Dataset\tIBP\tBitpack\tBP group")
for dataset in datasets:
    print(f"{dataset}\t{comp_size[dataset]:.2f}\t{bitp_size[dataset]:.2f}\t{bpgp_size[dataset]:.2f}")