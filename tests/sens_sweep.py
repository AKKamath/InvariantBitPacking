import torch
import ibp as ibp
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
import load_graph as ld
KVCACHE_FOLDER = "./kvcache/"

def pin_inplace(tensor):
    try:
        cudart = torch.cuda.cudart()
        r = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)
        assert tensor.is_pinned()
    except Exception as e:
        print(f"Failed to pin tensor: {e}")
    return tensor

def test_type(dataset, perc):
    mask, bitval = ibp.preprocess(dataset[:int(dataset.shape[0] * perc)])
    sizes = ibp.get_compress_size(dataset, mask, bitval)
    torch.cuda.synchronize()
    return 1 - (torch.sum(sizes) / (dataset.element_size() * dataset.nelement()))

sample_sizes = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
datasets = sys.argv[1].split()
comp_size = {}
for dataset in datasets:
    if dataset == "kvcache":
        folders = os.listdir(KVCACHE_FOLDER)
        index = 0
        if len(sys.argv) > 2:
            index = int(sys.argv[2])
        folder = folders[index]
        print(folder)
        files = os.listdir(KVCACHE_FOLDER + folder)
        features_read = {}
        for file in files:
            match = re.search(r"cache_0_([0-9]+)_([0-9]+)_", file)
            if match:
                layer = int(match.group(1))
                batch = int(match.group(2))
                tensor = torch.load(KVCACHE_FOLDER + folder + "/" + file)
                if layer not in features_read:
                    features_read[layer] = tensor
                else:
                    features_read[layer] = torch.cat((features_read[layer], tensor), dim=0)
        features = None
        for layer in features_read.keys():
            print(layer)
            feature = features_read[layer].view((features_read[layer].shape[0], features_read[layer].shape[1] * features_read[layer].shape[2])).view(torch.int64)
            if features is None:
                features = feature
            else:
                features = torch.cat((features, feature), dim=1)
            print(features.shape, features.dtype)
        features = features.pin_memory()
    else:
        g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", dataset)
        print(dataset)
    comp_size[dataset] = {}
    for sample in sample_sizes:
        comp_size[dataset][sample] = test_type(features, sample)


print(f"Dataset", end="\t")
for sample in sample_sizes:
    print(f"{sample * 100}%", end="\t")
print()

for dataset in datasets:
    print(dataset, end="\t")
    for sample in sample_sizes:
        print(f"{1 / (1 - comp_size[dataset][sample]):.2f}x", end="\t")
    print()