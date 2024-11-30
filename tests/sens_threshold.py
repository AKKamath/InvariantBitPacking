import torch
import ibp as ibp
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
import load_graph as ld

def pin_inplace(tensor):
    try:
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        assert tensor.is_pinned()
    except Exception as e:
        print(f"Failed to pin tensor: {e}")
    return tensor

def test_type(dataset, dtype):
    dataset = dataset.view(dtype=dtype)
    mask, bitval = ibp.preprocess(dataset)
    sizes = ibp.get_compress_size(dataset, mask, bitval)
    torch.cuda.synchronize()
    #print(sizes.view(dtype=torch.int64), dataset.shape[1])
    #elem_size = dataset.element_size()
    #print(torch.sum(sizes == dataset.shape[1] * elem_size))


g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", "products")
print(features.shape)
ibp.print_debug(True)

test_type(features, torch.int8)
test_type(features, torch.int16)
test_type(features, torch.int32)
test_type(features, torch.int64)