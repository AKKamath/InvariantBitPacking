import torch
import time
import sys
from dgl.utils import gather_pinned_tensor_rows
import numpy as np
from colossalai.legacy.nn.parallel.layers import CachedEmbeddingBag
import ibp

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

ITERS = 1000
TABLES = 26
import os
DLRM_FOLDER = "./dlrm_feats"
if len(sys.argv) > 1:
    DLRM_FOLDER = sys.argv[1]

files = os.listdir(DLRM_FOLDER)
# Filtering only the files.
files = [DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy' for f in range(TABLES) if os.path.isfile(DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy')]

BATCHES = [1024, 2048, 4096, 8192]

#ibp.print_debug(True)

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

# Embedding bags
comp_bags = []
fake_bags = []
def_bags = []

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
    comp_emb_bag = CachedEmbeddingBag(tensor.shape[0], tensor.shape[1], min_cache=max(BATCHES), _weight=tensor.view(torch.float64), comp=True, comp_imp=-1).to("cuda")
    fake_emb_bag = CachedEmbeddingBag(tensor.shape[0], tensor.shape[1], min_cache=max(BATCHES), _weight=tensor, comp=True, comp_imp=3).to("cuda")
    def_emb_bag  = CachedEmbeddingBag(tensor.shape[0], tensor.shape[1], min_cache=max(BATCHES), _weight=tensor, comp=False).to("cuda")

    comp_bags.append(comp_emb_bag)
    fake_bags.append(fake_emb_bag)
    def_bags.append(def_emb_bag)

    comp_tensor, mask, bitval, bitmask, comp_size, comp_arr = compress(tensor.view(torch.int64))
    comp_tensors.append(comp_tensor)
    masks.append(mask)
    bitvals.append(bitval)
    bitmasks.append(bitmask)
    comp_sizes.append(comp_size)
    comp_arrs.append(comp_arr)

    bitmask_zero = bitmask.clone()
    bitmask_zero = bitmask_zero.zero_()
    bitmasks0.append(bitmask_zero)

comp_transfer = {}
base_transfer = {}
comp_runtime = {}
base_runtime = {}
# PERFORM EXPERIMENT
for BATCH in BATCHES:
    offsets = torch.arange(BATCH)
    offsets_cuda = offsets.to("cuda:0")
    for i, tensor in enumerate(embedding_tables):
        # Untimed warmup
        indices = torch.randint(0, tensor.shape[0], [BATCH])
        indices_cuda = indices.to("cuda")
        torch.cuda.synchronize()

        print(torch.sum(comp_arrs[i][indices]) / (BATCH * 512))

        # Embedding bag time
        return_val = comp_bags[i](indices_cuda, offsets=offsets_cuda)
        return_val2 = def_bags[i](indices_cuda, offsets=offsets_cuda)
        return_val3 = fake_bags[i](indices_cuda, offsets=offsets_cuda)
        torch.cuda.synchronize()

        tran_time = 0
        tran2_time = 0
        ibp_time = 0
        ibp2_time = 0
        bag_time = 0
        bag2_time = 0
        bag3_time = 0
        comp_bags[i].cache_weight_mgr._reset_comm_stats()
        def_bags[i].cache_weight_mgr._reset_comm_stats()
        fake_bags[i].cache_weight_mgr._reset_comm_stats()

    for i, tensor in enumerate(embedding_tables):
        # Construct indices
        indices = torch.randint(0, tensor.shape[0], [BATCH])
        indices_cuda = indices.to("cuda")
        torch.cuda.synchronize()

        # DGL data transfer
        torch.cuda.synchronize()
        start = time.time_ns()
        for iter in range(ITERS):
            tensor_tran2 = gather_pinned_tensor_rows(tensor, indices_cuda)
        torch.cuda.synchronize()
        end = time.time_ns()
        tran2_time += end - start
        '''
        if not(torch.equal(decomp_tensor1, tensor_tran2)):
            print("IBP mismatch")
            print(decomp_tensor1, tensor_tran2)
            print(torch.sum(torch.eq(decomp_tensor1, tensor_tran2)).item()/decomp_tensor1.nelement())
            #exit(1)
        '''

        # Fake IBP transfer time
        torch.cuda.synchronize()
        start = time.time_ns()
        for iter in range(ITERS):
            decomp_tensor2 = ibp.decompress_fetch(comp_tensors[i], masks[i], bitvals[i], bitmasks0[i], torch.device('cuda'), comp_sizes[i], indices_cuda)
        torch.cuda.synchronize()
        end = time.time_ns()
        ibp2_time += end - start

        # IBP transfer time
        torch.cuda.synchronize()
        start = time.time_ns()
        for iter in range(ITERS):
            decomp_tensor1 = ibp.decompress_fetch(comp_tensors[i].view(torch.int64), masks[i], bitvals[i], bitmasks[i], torch.device('cuda'), comp_sizes[i], indices_cuda).view(torch.float32)
        torch.cuda.synchronize()
        end = time.time_ns()
        ibp_time += end - start

        for iter in range(ITERS):
            # Construct indices
            indices = torch.randint(0, tensor.shape[0], [BATCH])
            indices_cuda = indices.to("cuda")
            torch.cuda.synchronize()
            
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
            
            offsets_cuda = offsets.to("cuda:0")
            # Comp. Embedding bag time
            torch.cuda.synchronize()
            start = time.time_ns()
            return_val = comp_bags[i](indices_cuda, offsets=offsets_cuda).view(torch.float32)
            torch.cuda.synchronize()
            end = time.time_ns()
            bag_time += end - start
            '''
            if not(torch.equal(tensor_tran2, return_val)):
                print("Bag1 mismatch")
                print(tensor_tran2, return_val)
                print(torch.sum(torch.eq(tensor_tran2, return_val)).item()/tensor_tran2.nelement())
                #exit(1)
            '''

            # Embedding bag time
            torch.cuda.synchronize()
            start = time.time_ns()
            return_val2 = def_bags[i](indices_cuda, offsets=offsets_cuda)
            torch.cuda.synchronize()
            end = time.time_ns()
            bag2_time += end - start
            return_val2_cuda = return_val2.to("cuda")
            '''
            if not(torch.equal(tensor_tran2, return_val2_cuda)):
                print("Bag2 mismatch")
                print(tensor_tran2, return_val2_cuda)
                print(torch.sum(torch.eq(tensor_tran2, return_val2_cuda)).item()/tensor_tran2.nelement())
                exit(1)
            '''

            # Embedding bag time
            torch.cuda.synchronize()
            start = time.time_ns()
            return_val2 = fake_bags[i](indices_cuda, offsets=offsets_cuda)
            torch.cuda.synchronize()
            end = time.time_ns()
            bag3_time += end - start
            return_val2_cuda = return_val2.to("cuda")
            '''
            if not(torch.equal(tensor_tran2, return_val2_cuda)):
                print("Bag2 mismatch")
                print(tensor_tran2, return_val2_cuda)
                print(torch.sum(torch.eq(tensor_tran2, return_val2_cuda)).item()/tensor_tran2.nelement())
                #exit(1)
            '''

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


print("Batch\tRuntime speedup\tTransfer speedup\tcomp_runtime\tbase_runtime\tcomp_transfer\tbase_transfer")
for BATCH in BATCHES:
    run_sp = base_runtime[BATCH] / comp_runtime[BATCH]
    transfer_sp = base_transfer[BATCH] / comp_transfer[BATCH]
    print(f"{BATCH}\t{run_sp:.4f}\t{transfer_sp:.4f}\t{comp_runtime[BATCH]:.4f}\t{base_runtime[BATCH]:.4f}\t" +
          f"{comp_transfer[BATCH]:.4f}\t{base_transfer[BATCH]:.4f}")