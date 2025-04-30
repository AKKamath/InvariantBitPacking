import sys
import os
import torch
import numpy as np

benchmark = sys.argv[1]
print(benchmark)
if benchmark == "dlrm":
    TABLES = 26
    DLRM_FOLDER = "./dlrm_feats"
    if len(sys.argv) > 2:
        DLRM_FOLDER = sys.argv[2]

    files = os.listdir(DLRM_FOLDER)
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
        import ibp_cuda_test
        ibp_cuda_test.test_compress(features)
    exit()

else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
    import load_graph as ld
    g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", "reddit", False)

print(features.shape, features.dtype)

import ibp_cuda_test
ibp_cuda_test.test_compress(features)