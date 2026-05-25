import sys
import os
import re
import torch
import numpy as np
import ibp_cuda_test
import ibp

LLM_FOLDER = os.path.expanduser("~/opt_weights/gemma-7b-pytorch-np/")

def test_type(dataset):
    mask, bitval = ibp.preprocess(dataset)
    sizes = ibp.get_compress_size(dataset, mask, bitval)
    torch.cuda.synchronize()
    return (dataset.element_size() * dataset.nelement()) / torch.sum(sizes)

files = os.listdir(LLM_FOLDER)
features = {}
for file in files:
    match = re.search(r"layers\.([0-9]+\.mlp.*)\.weight", file)
    if match:
        layer = match.group(1)
        tensor = torch.load(LLM_FOLDER + file)
        features[layer] = tensor

layers = sorted(features.keys())
for layer in layers:
    #print(features[layer], flush=True)
    if len(features[layer].shape) == 3:
        feature = features[layer].view((features[layer].shape[0], features[layer].shape[1] * features[layer].shape[2])).view(torch.int64)
    else:
        feature = features[layer].view(torch.int64)
    #print(feature)
    print("Layer", layer, flush=True)
    print(feature.shape, feature.dtype, flush=True)
    #print("Non-zero columns", torch.count_nonzero(torch.count_nonzero(feature, dim=1)), "of", feature.shape[0], flush=True)
    if  (feature.shape[1] > 8192):
        split_feats = torch.split(feature, 8192, dim=1)
        for split_feat in split_feats:
            #print(split_feat[0], split_feat.shape, split_feat.dtype, flush=True)
            #for f in split_feat[0]:
            #    print(f, flush=True)
            split_feat = split_feat.contiguous().pin_memory()
            ibp_cuda_test.test_compress(split_feat)
    else:
        ibp_cuda_test.test_compress(feature.pin_memory())