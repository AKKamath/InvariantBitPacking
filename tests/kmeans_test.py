import torch
import ibp_cuda as ibp

ibp.print_debug(True)

tensor = torch.rand(100, 100).pin_memory()

ibp.preprocess(tensor, None)
masks,vals = ibp.preprocess_kmeans(tensor, 100, None)