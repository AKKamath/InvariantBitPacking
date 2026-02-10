import torch
import ibp as ibp
import numpy as np
import math
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
import load_graph as ld

DLRM_FOLDER = "./dlrm_feats"
KVCACHE_FOLDER = "./kvcache/"

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
    if dataset == "kvcache":
        folders = os.listdir(KVCACHE_FOLDER)
        index = 0
        if len(sys.argv) > 2:
            index = int(sys.argv[2])
        folder = folders[index]
        print(folders, flush=True)
        print(folder, flush=True)
        files = os.listdir(KVCACHE_FOLDER + folder)
        features = {}
        for file in files:
            match = re.search(r"cache_0_([0-9]+)_([0-9]+)_", file)
            if match:
                layer = int(match.group(1))
                batch = int(match.group(2))
                tensor = torch.load(KVCACHE_FOLDER + folder + "/" + file)
                if layer not in features:
                    features[layer] = tensor
                else:
                    features[layer] = torch.cat((features[layer], tensor), dim=0)
        for ind, layer in enumerate(features.keys()):
            '''
            if ind == 0:
                feature = features[layer].view((features[layer].shape[0], features[layer].shape[1] * features[layer].shape[2])).view(torch.int32)
            else:
                feat = features[layer].view((features[layer].shape[0], features[layer].shape[1] * features[layer].shape[2])).view(torch.int32)
                feature = torch.cat((feat, feature))
            '''
            feature = features[layer].view((features[layer].shape[0], features[layer].shape[1] * features[layer].shape[2])).view(torch.int32)
            feature = feature.detach().clone().pin_memory()
            comp_size[dataset + str(layer)] = ibp_ify(feature)
            bitp_size[dataset + str(layer)] = bitpack(feature)
            bpgp_size[dataset + str(layer)] = bitpack_group(feature)
        continue
    # Filtering only the files.
    if dataset == 'dlrm':
        TABLES = 26
        files = os.listdir(DLRM_FOLDER)
        files = [DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy' for f in range(TABLES) if os.path.isfile(DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy')]

        # PREPROCESSING AND SETTING UP DATA STRUCTURES
        for ind, file in enumerate(files):
            weights = np.load(file)
            tensor = torch.from_numpy(weights).pin_memory()
            print(tensor.shape)
            if ind == 0:
                features = tensor
            else:
                features = torch.cat((tensor, features))
        features = features.detach().clone().pin_memory()
    else:
        g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", dataset)
    print(dataset)
    comp_size[dataset] = ibp_ify(features)
    bitp_size[dataset] = bitpack(features)
    bpgp_size[dataset] = bitpack_group(features)

print("Dataset\tIBP\tBitpack\tBP group")
for dataset in comp_size.keys():
    print(f"{dataset}\t{comp_size[dataset]:.2f}\t{bitp_size[dataset]:.2f}\t{bpgp_size[dataset]:.2f}")