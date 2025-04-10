import ibp
import torch

def do_the_process(orig_tensor, print_out = False):
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
    print(f"OG {og_size} Compress: {comp_size} ({ratio:.2f}x)")
    #print("Compressed tensor ", ibp.compress(tensor, mask, bitval))
    bitmask = ibp.compress_inplace(comp_tensor, mask, bitval)
    torch.cuda.synchronize()
    # Decompress
    decomp_tensor = ibp.decompress_fetch(comp_tensor, mask, bitval, bitmask, torch.device('cuda'))
    print("Match?", torch.equal(orig_tensor.to(torch.device('cuda')), decomp_tensor))
    if(not torch.equal(orig_tensor.to(torch.device('cuda')), decomp_tensor)):
        print("Original ", orig_tensor)
        print("Decompressed ", decomp_tensor)

# Turn on debug print messages
#ibp.print_debug(True)

# Create a tensor
sp_tensor32 = torch.rand([1200,128], dtype=torch.float32).pin_memory()
sp_tensor64 = sp_tensor32.clone().pin_memory().view(dtype=torch.float64)
do_the_process(sp_tensor32)
do_the_process(sp_tensor64)

# Create a tensor
dense_tensor32 = torch.randint(2147483647, [100, 100], dtype=torch.int32).pin_memory()
dense_tensor64 = dense_tensor32.clone().pin_memory().view(dtype=torch.int64)

do_the_process(dense_tensor32)
do_the_process(dense_tensor64)

# Create a tensor
semidense32 = torch.randint(2147483647 // 100, [6, 100], dtype=torch.int32).pin_memory()
semidense64 = semidense32.clone().pin_memory().view(dtype=torch.int64)
do_the_process(semidense32)
do_the_process(semidense64)