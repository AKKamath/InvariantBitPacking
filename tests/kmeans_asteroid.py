import torch
import time
import sys
import numpy as np
import ibp

def compress_kmeans(orig_tensor, print_out = False):
    comp_tensor = orig_tensor.detach().clone().pin_memory()

    centroids = 1
    while centroids < comp_tensor.shape[0] * 0.1:
        masks,vals,clusterids = ibp.preprocess_kmeans(comp_tensor, centroids, None)
        centroids *= 2
    torch.cuda.synchronize()

import os
DATASET_FOLDER = "./dlrm_feats"
if len(sys.argv) > 1:
    DATASET_FOLDER = sys.argv[1]

ibp.print_debug(True)

print("Asteroid")
file = DATASET_FOLDER+'/asteroid.f32'
# PREPROCESSING AND SETTING UP DATA STRUCTURES
weights = np.memmap(file, dtype='float32', mode='r')
tensor = torch.from_numpy(weights).pin_memory()
tensor = tensor.reshape([tensor.shape[0] // 500, 500])
print(tensor.shape)
compress_kmeans(tensor)

print("Reddit")
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
import load_graph as ld
g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", "reddit")
compress_kmeans(features)

INT_MAX = 2147483647
NUM_VEC_SETS = 250000
VEC_SIZE = 500

# Uniform distribution
print("Uniform")
random_vec = torch.randint(INT_MAX, [NUM_VEC_SETS, VEC_SIZE], dtype=torch.int32).pin_memory()
compress_kmeans(random_vec)

# Normal distribution
print("Normal")
random_vec = torch.normal(float(0), float(INT_MAX / 3), size=(NUM_VEC_SETS, VEC_SIZE)).type(torch.int32).pin_memory()
compress_kmeans(random_vec)