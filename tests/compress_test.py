import ibp
import torch

# Create a tensor
sparse_tensor = torch.tensor([[4, 2, 5], [5, 4, 7], [6, 8, 13], [7, 16, 21]], dtype=torch.int64).pin_memory()
print(sparse_tensor)
# Turn on debug print messages
ibp.print_debug(True)
mask, bitval = ibp.preprocess(sparse_tensor)
for i in range(mask.size(0)):
    print("{:#04x} {:d}".format(mask[i], bitval[i]))