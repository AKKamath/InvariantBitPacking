import sys
import os
import re
import torch
import numpy as np
import ibp_cuda_test


if len(sys.argv) < 2:
    print("Usage: python nvcomp_comparison.py <benchmark> [<folder>]")
    print("benchmark: dlrm, kvcache")
    print("folder: folder containing the dlrm/kvcache features")
    sys.exit(1)
benchmark = sys.argv[1]
print(benchmark)
if benchmark == "dlrm":
    TABLES = 26
    DLRM_FOLDER = "./dlrm_feats"
    if len(sys.argv) > 2:
        DLRM_FOLDER = sys.argv[2]

    #files = os.listdir(DLRM_FOLDER)
    # Filtering only the files.
    files = [DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy' for f in range(TABLES) if os.path.isfile(DLRM_FOLDER+'/feature_' + str(f) + '_part0.npy')]
    tensors = []
    for file in files:
        weights = np.load(file)
        features = torch.from_numpy(weights).pin_memory()
        if features.shape[0] < 100000:
            continue
        features = features.view(torch.int64)
        print(file)
        print(features.shape, features.dtype)
        #features = torch.cat(tensors, dim=0)
        ibp_cuda_test.test_compress(features)
elif benchmark == "kvcache":
    KVCACHE_FOLDER = "./kvcache/"
    folders = os.listdir(KVCACHE_FOLDER)
    index = 0
    if len(sys.argv) > 2:
        index = int(sys.argv[2])
    folder = folders[index]
    print(folder)
    files = os.listdir(KVCACHE_FOLDER + folder)
    features = {}
    for file in files:
        match = re.search(r"cache_0_([0-9]+)_([0-9]+)_", file)
        if match:
            layer = int(match.group(1))
            batch = int(match.group(2))
            tensor = torch.load(KVCACHE_FOLDER + folder + "/" + file)
            if layer not in features:
                features[layer] = tensor
            else:
                features[layer] = torch.cat((features[layer], tensor), dim=0)
    '''
    for layer in features.keys():
        feature = features[layer].view((features[layer].shape[0], features[layer].shape[1] * features[layer].shape[2])).view(torch.int64)
        non_zero_columns = torch.count_nonzero(torch.count_nonzero(feature, dim=1))
        print(layer, non_zero_columns, non_zero_columns.shape)
        #features[layer] = features[layer][:, non_zero_columns]
    '''
    for layer in features.keys():
        print(layer)
        feature = features[layer].view((features[layer].shape[0], features[layer].shape[1] * features[layer].shape[2])).view(torch.int64).pin_memory()
        print(feature.shape, feature.dtype)
        if  (feature.shape[1] > 8192):
            split_feats = torch.split(feature, 8192, dim=1)
            for split_feat in split_feats:
                ibp_cuda_test.test_compress(split_feat)
        else:
            ibp_cuda_test.test_compress(feature)

else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
    import load_graph as ld
    g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", benchmark, False)
    print(features.shape, features.dtype)

    ibp_cuda_test.test_compress(features)