import torch
import time
import sys
import numpy as np
import ibp

def compress_kmeans(orig_tensor, print_out = False):
    comp_tensor = orig_tensor.detach().clone().pin_memory()

    centroids = 1
    while centroids < comp_tensor.shape[0] * 0.2:
        masks,vals,clusterids = ibp.preprocess_kmeans(comp_tensor, centroids, None)
        centroids *= 2
    torch.cuda.synchronize()

import os
DATASET_FOLDER = "./dlrm_feats"
if len(sys.argv) > 1:
    DATASET_FOLDER = sys.argv[1]

files = [DATASET_FOLDER+'/asteroid.f32']

ibp.print_debug(True)

# PREPROCESSING AND SETTING UP DATA STRUCTURES
sizes = []
for file in files:
    weights = np.memmap(file, dtype='float32', mode='r')
    tensor = torch.from_numpy(weights).pin_memory()
    tensor = tensor.reshape([tensor.shape[0], 1])
    sizes.append(tensor.shape[0])
    print(tensor.shape)
    compress_kmeans(tensor)
sizes = torch.tensor(sizes)
