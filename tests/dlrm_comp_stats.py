import torch
import time
import sys
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
import ibp

ibp.print_debug(True)

def compress(orig_tensor, print_out = False):
    # Preprocess to generate mask and bitval
    mask, bitval = ibp.preprocess(orig_tensor)
    comp_tensor = orig_tensor.detach().clone().pin_memory()
    # Check compressed size
    comp_sizes = ibp.get_compress_size(comp_tensor, mask, bitval)
    torch.cuda.synchronize()

    comp_vecs = (comp_sizes != (comp_tensor.shape[1] * comp_tensor.element_size())).nonzero()
    print("Num comp vecs: ", comp_vecs.shape)
    og_size = comp_vecs.shape[0] * comp_tensor.shape[1] * comp_tensor.element_size()
    comp_size = int(torch.sum(comp_sizes[comp_vecs]))
    ratio = og_size / comp_size
    comp_size = int(comp_tensor.shape[1] * (comp_size / og_size))

    # Calculations
    '''
    og_size = comp_tensor.numel() * comp_tensor.element_size()
    comp_size = int(torch.sum(comp_sizes))
    ratio = og_size / comp_size
    comp_size = int(comp_tensor.shape[1] * (comp_size / og_size))
    '''
    print(f"OG {comp_tensor.shape[1]} Compress: {comp_size} ({ratio:.2f}x)")
    #print("Compressed tensor ", ibp.compress(tensor, mask, bitval))
    bitmask = ibp.compress_inplace(comp_tensor, mask, bitval)
    torch.cuda.synchronize()
    return comp_tensor, mask, bitval, bitmask, comp_size, comp_sizes

TABLES = 26
import os
DLRM_FOLDER = "./dlrm_feats"
if len(sys.argv) > 1:
    DLRM_FOLDER = sys.argv[1]

files = os.listdir(DLRM_FOLDER)
# Filtering only the files.
files = [DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy' for f in range(TABLES) if os.path.isfile(DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy')]

# PREPROCESSING AND SETTING UP DATA STRUCTURES
embedding_tables = []
sizes = []
for ind, file in enumerate(files):
    weights = np.load(file)
    tensor = torch.from_numpy(weights).pin_memory()
    sizes.append(tensor.shape[0])
    embedding_tables.append(tensor)
    print(tensor.shape)
sizes = torch.tensor(sizes)

# Compressed tensors
comp_tensors = []
masks = []
bitvals = []
bitmasks = []
bitmasks0 = []
comp_sizes = []
comp_arrs = []

# Construct a tensor for each table weights
for i, tensor in enumerate(embedding_tables):
    comp_tensor, mask, bitval, bitmask, comp_size, comp_arr = compress(tensor.view(torch.int64))