import torch
import ibp
import time

ITERS = 100

def bitsoncount(x):
    count = 0
    for i in range(32):
        if(x & (1 << i)):
            count += 1
    return count


def make_mask_and_bitval(tensor, compression):
    VEC_BITS = 32 * tensor.shape[1]
    mask = torch.zeros(tensor.shape[1], dtype=torch.int32)
    bitval = tensor[0].clone()
    indices = torch.randperm(VEC_BITS)
    i = 0
    for index in indices:
        mask[index // 32] |= 1 << (index % 32)
        i += 1
        if i >= compression * VEC_BITS + tensor.shape[1] + 1:
            break
    return mask, bitval

def transfer(tensor):
    torch.cuda.synchronize()
    start = time.time_ns()
    for i in range(ITERS):
        tensor_copy = tensor.to(torch.device('cuda'))
    torch.cuda.synchronize()
    end = time.time_ns()
    print(f"Transfer time: {(end - start) / 1e6 / ITERS:.3f}ms")
    tot_time = (end - start) / 1e6 / ITERS
    return tot_time

def compress_decompress(tensor, mask, bitval):
    mask = mask.to(torch.device('cuda'))
    bitval = bitval.to(torch.device('cuda'))
    copy_tensor = tensor.detach().clone().pin_memory()
    comp_sizes = ibp.get_compress_size(copy_tensor, mask, bitval)
    torch.cuda.synchronize()
    og_size = copy_tensor.numel() * copy_tensor.element_size()
    comp_size = torch.sum(comp_sizes)
    ratio = 1 - (comp_size / og_size)
    bitmask = ibp.compress_inplace(copy_tensor, mask, bitval)
    torch.cuda.synchronize()

    # Timed decompress
    start = time.time_ns()
    for i in range(ITERS):
        decomp_tensor = ibp.decompress_fetch(copy_tensor, mask, bitval, bitmask, \
            torch.device('cuda'), (comp_size / og_size) * 4 * tensor.shape[1])
    torch.cuda.synchronize()
    end = time.time_ns()
    tot_time = (end - start) / 1e6 / ITERS
    print(f"{ratio*100:.0f}%", end="\t")
    print(f"Decompress time: {(end - start) / 1e6 / ITERS:.3f}ms")

    if(not torch.equal(tensor, decomp_tensor.to(torch.device('cpu')))):
        print("mismatch")
        print("Original ", tensor)
        print("Decompressed ", decomp_tensor)
    return ratio.item(), tot_time

# Turn on debug print messages
#ibp.print_debug(True)

TARGET = [0.125, 0.25, 0.5, 0.75, 0.9, 0.95, 0.97]
SIZES = [256, 1024, 4 * 1024, 16 * 1024]
NUM_VECS = 10000

runtime = {}
for j in SIZES:
    runtime[j] = {}

for size in SIZES:
    print(f"Size: {size} bytes")
    # Create a tensor of size bytes
    tensor = torch.zeros([NUM_VECS, size // 4], dtype=torch.int32).pin_memory()
    runtime[size][0] = transfer(tensor)
    for rate in TARGET:
        mask, bitval = make_mask_and_bitval(tensor, rate)
        rate, time_taken = compress_decompress(tensor, mask, bitval)
        runtime[size][rate] = runtime[size][0] / time_taken
    print()
    # Transfer thput

print("Size\t", end="")
for j in TARGET:
    print(f"{j:.3f}", end="\t")
print()

for i in SIZES:
    print(i, end="\t")
    for j in runtime[i].keys():
        if(j == 0):
            continue
        print(f"{runtime[i][j]:.3f}", end="\t")
    print()