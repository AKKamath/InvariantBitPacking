import torch
import ibp_cuda as ibp
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../training_backend/")
import load_graph as ld
import matplotlib.pyplot as plt
import pandas as  pd
import numpy as np

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.datasets.samples_generator import make_blobs

#from pandas.tools.plotting import parallel_coordinates


ibp.print_debug(True)

'''
g, features, labels, training_ids, validation_ids, testing_ids = ld.load("../../../dataset/", "reddit")

ibp.preprocess(features, None)
centroids = 4096 * 8
#while centroids < features.size()[0]:
if 1:
    masks, bitvals, clusterIds = ibp.preprocess_kmeans(features, centroids, None)
    #centroids *= 2


px = pd.DataFrame(features.numpy())
X_norm = (px - px.min())/(px.max() - px.min())
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
for i in range(centroids):
    indices = clusterIds == i
    plt.scatter(transformed[indices.tolist()][0], transformed[indices.tolist()][1], label='Class ' + str(i), c='C' + str(i))

#plt.legend()
plt.savefig("pca.png")
'''

INT_MAX = 2147483647
NUM_VEC_SETS = 1024 * 32
NUM_VECS = 1
VEC_SIZE = 2

# Uniform distribution
random_vec = torch.randint(INT_MAX, [VEC_SIZE], dtype=torch.int32).repeat(NUM_VECS, 1)
for i in range(NUM_VEC_SETS - 1):
    random_vec_add = torch.randint(INT_MAX, [VEC_SIZE], dtype=torch.int32)
    random_vec = torch.cat((random_vec, random_vec_add.repeat(NUM_VECS, 1)))
random_vec = random_vec[torch.randperm(random_vec.size()[0])].pin_memory()
#print(random_vec)
ibp.preprocess(random_vec, None)
centroids = 8
#if 1:
while centroids < NUM_VEC_SETS * 2:
    masks,vals,clusterids = ibp.preprocess_kmeans(random_vec, centroids, None)
    centroids *= 2

# Normal distribution
random_vec = torch.normal(float(0), float(INT_MAX / 3), size=(VEC_SIZE,)).repeat(NUM_VECS, 1).type(torch.int32)
for i in range(NUM_VEC_SETS - 1):
    random_vec_add = torch.normal(float(0), float(INT_MAX / 3), size=(VEC_SIZE,)).type(torch.int32)
    random_vec = torch.cat((random_vec, random_vec_add.repeat(NUM_VECS, 1)))
random_vec = random_vec[torch.randperm(random_vec.size()[0])].pin_memory()
#print(random_vec)
ibp.preprocess(random_vec, None)
centroids = 8
#if 1:
while centroids < NUM_VEC_SETS * 2:
    masks,vals,clusterids = ibp.preprocess_kmeans(random_vec, centroids, None)
    centroids *= 2

'''
a = 4
# Zipf distribution
random_vec = torch.from_numpy(np.random.zipf(a, (VEC_SIZE))).repeat(NUM_VECS, 1).type(torch.int32)
for i in range(NUM_VEC_SETS - 1):
    random_vec_add = torch.from_numpy(np.random.zipf(a, (VEC_SIZE))).repeat(NUM_VECS, 1).type(torch.int32)
    random_vec = torch.cat((random_vec, random_vec_add.repeat(NUM_VECS, 1)))
random_vec = random_vec[torch.randperm(random_vec.size()[0])].pin_memory()
#print(random_vec)
ibp.preprocess(random_vec, None)
centroids = 8
#if 1:
while centroids < NUM_VEC_SETS * 2:
    masks,vals,clusterids = ibp.preprocess_kmeans(random_vec, centroids, None)
    centroids *= 2


print(random_vec.shape)
print(clusterids.shape)
indices = clusterids == 0
random_vec = pd.DataFrame(random_vec.numpy())
print(random_vec[indices.tolist()])
print(random_vec[indices.tolist()][0])

for i in range(centroids):
    indices = clusterids == i
    plt.scatter(random_vec[indices.tolist()][0], random_vec[indices.tolist()][1], label='Class ' + str(i), c='C' + str(i))

#plt.legend()
plt.savefig("pca.png")
'''