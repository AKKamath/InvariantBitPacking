import torch
import time
import sys
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
from colossalai.legacy.nn.parallel.layers import CachedEmbeddingBag
import ibp

ITERS = 100
TABLES = 26
import os
DLRM_FOLDER = "./dlrm_feats"
if len(sys.argv) > 1:
    DLRM_FOLDER = sys.argv[1]

files = os.listdir(DLRM_FOLDER)
# Filtering only the files.
files = [DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy' for f in range(TABLES) if os.path.isfile(DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy')]

BATCHES = [8192, 16384, 32768]#[1024, 2048, 4096, 8192, 16384, 32768]

# PREPROCESSING AND SETTING UP DATA STRUCTURES
embedding_tables = []
sizes = []
for file in files:
    weights = np.load(file)
    tensor = torch.from_numpy(weights).pin_memory()
    sizes.append(tensor.shape[0])
    embedding_tables.append(tensor)
    print(tensor.shape)
sizes = torch.tensor(sizes)

offset = 0
table_weights = torch.empty([torch.sum(sizes), 128]).pin_memory()

# Make sizes a cumulative sum (inclusive scan)
sizes = torch.cumsum(sizes, 0)
# Construct a single tensor for all table weights
for i, tensor in enumerate(embedding_tables):
    if i == 0:
        table_weights[0 : sizes[i]] = tensor
    else:
        table_weights[sizes[i - 1] : sizes[i]] = tensor
# Increase cache ratio from default 0.01 if batch size is too big
cache_ratio = min(0.01, max(BATCHES) * TABLES / sizes[-1])
comp_emb_bags = CachedEmbeddingBag(table_weights.shape[0], table_weights.shape[1], _weight=table_weights, comp=True, cache_ratio=0.02, comp_imp=0).to("cuda")
def_emb_bags  = CachedEmbeddingBag(table_weights.shape[0], table_weights.shape[1], _weight=table_weights, comp=True, cache_ratio=0.02, comp_imp=1).to("cuda")

def compress(orig_tensor, print_out = False):
    # Preprocess to generate mask and bitval
    mask, bitval = ibp.preprocess(orig_tensor)
    comp_tensor = orig_tensor.detach().clone().pin_memory()
    # Check compressed size
    comp_sizes = ibp.get_compress_size(comp_tensor, mask, bitval)
    torch.cuda.synchronize()
    # Calculations
    og_size = comp_tensor.numel() * comp_tensor.element_size()
    comp_size = int(torch.sum(comp_sizes))
    ratio = og_size / comp_size
    comp_size = int(comp_tensor.shape[1] * (comp_size / og_size))
    print(f"OG {comp_tensor.shape[1]} Compress: {comp_size} ({ratio:.2f}x)")
    #print("Compressed tensor ", ibp.compress(tensor, mask, bitval))
    bitmask = ibp.compress_inplace(comp_tensor, mask, bitval)
    torch.cuda.synchronize()
    return comp_tensor, mask, bitval, bitmask, comp_size

comp_tensor, mask, bitval, bitmask, comp_size = compress(table_weights)

comp_transfer = {}
base_transfer = {}
comp_runtime = {}
base_runtime = {}
# PERFORM EXPERIMENT
for BATCH in BATCHES:
    offsets = torch.arange(BATCH * TABLES)

    # Untimed warmup
    indices = torch.empty([TABLES, BATCH], dtype=torch.int64)
    for i, tensor in enumerate(embedding_tables):
        if i == 0:
            indices[i] = torch.randint(0, tensor.shape[0], [BATCH])
        else:
            indices[i] = torch.randint(sizes[i - 1], tensor.shape[0] + sizes[i - 1], [BATCH])

    indices = indices.view(TABLES * BATCH)
    indices_cuda = indices.to("cuda")
    torch.cuda.synchronize()

    # DGL data transfer
    tensor_tran2 = gather_pinned_tensor_rows(table_weights, indices_cuda)
    offsets_cuda = offsets.to("cuda:0")
    # Comp transfer
    decomp_tensor = ibp.decompress_fetch(comp_tensor, mask, bitval, bitmask, torch.device('cuda'), comp_size, indices_cuda)
    # Embedding bag time
    return_val = comp_emb_bags(indices_cuda, offsets=offsets_cuda)
    #return_val2 = def_emb_bags(indices_cuda, offsets=offsets_cuda)
    torch.cuda.synchronize()

    tran_time = 0
    tran2_time = 0
    ibp_time = 0
    ibp2_time = 0
    bag_time = 0
    bag2_time = 0
    comp_emb_bags.cache_weight_mgr._reset_comm_stats()
    def_emb_bags.cache_weight_mgr._reset_comm_stats()
    for iter in range(ITERS):
        # Construct indices
        # Get offset for each individual table and add to index value
        indices = torch.empty([TABLES, BATCH], dtype=torch.int64)
        for i, tensor in enumerate(embedding_tables):
            if i == 0:
                indices[i] = torch.randint(0, tensor.shape[0], [BATCH])
            else:
                indices[i] = torch.randint(sizes[i - 1], tensor.shape[0] + sizes[i - 1], [BATCH])
        #print(indices)

        indices = indices.view(TABLES * BATCH)
        
        # Simple data transfer
        '''
        start = time.time_ns()
        if indices is not None:
            tensor_tran = table_weights.index_select(0, indices).to(torch.device('cuda'))
        else:
            tensor_tran = table_weights.to(torch.device('cuda'))
        torch.cuda.synchronize()
        end = time.time_ns()
        tran_time += end - start
        '''

        indices_cuda = indices.to("cuda")
        torch.cuda.synchronize()

        # DGL data transfer
        torch.cuda.synchronize()
        start = time.time_ns()
        tensor_tran2 = gather_pinned_tensor_rows(table_weights, indices_cuda)
        torch.cuda.synchronize()
        end = time.time_ns()
        tran2_time += end - start

        # IBP transfer time
        torch.cuda.synchronize()
        start = time.time_ns()
        decomp_tensor = ibp.decompress_fetch(comp_tensor, mask, bitval, bitmask, torch.device('cuda'), comp_size, indices_cuda)
        torch.cuda.synchronize()
        end = time.time_ns()
        ibp_time += end - start

        # IBP transfer time
        torch.cuda.synchronize()
        start = time.time_ns()
        decomp_tensor = ibp.decompress_fetch(comp_tensor, mask, bitval, bitmask, torch.device('cuda'), comp_size, indices_cuda, None, None, 1)
        torch.cuda.synchronize()
        end = time.time_ns()
        ibp2_time += end - start
        
        offsets_cuda = offsets.to("cuda:0")
        # Comp. Embedding bag time
        torch.cuda.synchronize()
        start = time.time_ns()
        return_val = comp_emb_bags(indices_cuda, offsets=offsets_cuda)
        torch.cuda.synchronize()
        end = time.time_ns()
        bag_time += end - start
        if not(torch.equal(tensor_tran2, return_val)):
            print("Bag1 mismatch")
            print(tensor_tran2, return_val)
            print(torch.sum(torch.eq(tensor_tran2, return_val)).item()/tensor_tran2.nelement())
            exit(1)

        # Embedding bag time
        torch.cuda.synchronize()
        start = time.time_ns()
        return_val2 = def_emb_bags(indices_cuda, offsets=offsets_cuda)
        torch.cuda.synchronize()
        end = time.time_ns()
        bag2_time += end - start
        return_val2_cuda = return_val2.to("cuda")
        if not(torch.equal(tensor_tran2, return_val2_cuda)):
            print("Bag2 mismatch")
            print(tensor_tran2, return_val2_cuda)
            print(torch.sum(torch.eq(tensor_tran2, return_val2_cuda)).item()/tensor_tran2.nelement())
            exit(1)
        

    print(f"Batch: {BATCH}")
    #print(f"Transfer (index + copy) time: {tran_time / ITERS / 1e6:.3f} ms")
    print(f"Transfer (GPU copy) time: {tran2_time / ITERS / 1e6:.3f} ms")
    print(f"IBP time: {ibp_time / ITERS / 1e6:.3f} ms")
    print(f"IBP2 time: {ibp2_time / ITERS / 1e6:.3f} ms")
    print(f"Cache_Emb_bag time: {bag_time / ITERS / 1e6:.3f} ms")
    print(f"Emb_bag time: {bag2_time / ITERS / 1e6:.3f} ms")
    comp_runtime[BATCH] = bag_time / ITERS / 1e6
    base_runtime[BATCH] = bag2_time / ITERS / 1e6
    # Time is in s, convert to ms
    comp_transfer[BATCH] = ibp_time / ITERS / 1e6
    base_transfer[BATCH] = tran2_time / ITERS / 1e6

    d = comp_emb_bags.cache_weight_mgr._elapsed_dict
    myKeys = list(d.keys())
    myKeys.sort()
    sd = {i: d[i] for i in myKeys}
    print(sd)

    d = def_emb_bags.cache_weight_mgr._elapsed_dict
    myKeys = list(d.keys())
    myKeys.sort()
    sd = {i: d[i] for i in myKeys}
    print(sd)

print("Batch\tRuntime speedup\tTransfer speedup\tcomp_runtime\tbase_runtime\tcomp_transfer\tbase_transfer")
for BATCH in BATCHES:
    run_sp = base_runtime[BATCH] / comp_runtime[BATCH]
    transfer_sp = base_transfer[BATCH] / comp_transfer[BATCH]
    print(f"{BATCH}\t{run_sp:.4f}\t{transfer_sp:.4f}\t{comp_runtime[BATCH]:.4f}\t{base_runtime[BATCH]:.4f}\t" +
          f"{comp_transfer[BATCH]:.4f}\t{base_transfer[BATCH]:.4f}")