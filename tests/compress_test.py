import ibp
import torch

def do_the_process(tensor):
    # Preprocess to generate mask and bitval
    mask, bitval = ibp.preprocess(tensor)
    # Check compressed size
    comp_sizes = ibp.get_compress_size(tensor, mask, bitval)
    # Calculations
    og_size = tensor.numel() * tensor.element_size()
    comp_size = int(torch.sum(comp_sizes))
    ratio = og_size / comp_size
    print(f"OG {og_size} Compress: {comp_size} ({ratio:.2f}x)")

# Turn on debug print messages
ibp.print_debug(True)

# Create a tensor
sp_tensor = torch.tensor([range(0, 100)] * 8, dtype=torch.int32).pin_memory()
do_the_process(sp_tensor)

sp_tensor = sp_tensor.view(dtype=torch.int64)
do_the_process(sp_tensor)

# Create a tensor
dense_tensor = torch.randint(2147483647, [8, 100], dtype=torch.int32).pin_memory()
do_the_process(dense_tensor)

dense_tensor = dense_tensor.view(dtype=torch.int64)
do_the_process(dense_tensor)

semidense = torch.randint(2147483647 // 100, [8, 100], dtype=torch.int32).pin_memory()
do_the_process(semidense)
semidense = semidense.view(dtype=torch.int64)
do_the_process(semidense)