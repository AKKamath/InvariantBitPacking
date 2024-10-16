#ifndef IBP_PREPROC_KMEANS
#define IBP_PREPROC_KMEANS
#include "ibp_helpers.cuh"
namespace ibp {
__global__ void classify_nodes(int *masks, int *vals, int num_centroids,
    int32_t *typecast_feats, int32_t *index_arr, int num_nodes, int32_t *dev_cluster, int feature_len) {
    __shared__ int min_dist;
    int warpId = threadIdx.x / DWARP_SIZE;
    int numWarps = blockDim.x / DWARP_SIZE;
    const bool warpLeader = (threadIdx.x % DWARP_SIZE == 0);
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        min_dist = INT_MAX;
        int64_t nodeId = index_arr[i];
        __syncthreads();
        // Align to warp factor
        for(int j = warpId; j < (num_centroids + numWarps - (num_centroids % numWarps)); j += numWarps) {
            int dist = 0;
            if(j < num_centroids) {
                for(int k = threadIdx.x % DWARP_SIZE; k < feature_len; k += DWARP_SIZE) {
                    //typecast_feats[i * feature_len + k] ^ dev_centroids[j * feature_len + k];
                    int32_t val = (typecast_feats[nodeId * feature_len + k] & masks[j * feature_len + k]) ^ vals[j * feature_len + k];
                    dist += __popc(val);
                }
                __syncwarp();
                for (int offset = 16; offset > 0; offset /= 2)
                    dist += __shfl_down_sync(FULL_MASK, dist, offset);
                __syncwarp();
                if(warpLeader) {
                    atomicMin(&min_dist, dist);
                }
            }
            __syncthreads();
            if(warpLeader) {
                if(min_dist == dist) {
                    atomicExch(&dev_cluster[i], j);
                }
            }
        }
    }
}

__global__ void calc_distances(int32_t *centroids, int num_centroids, int32_t *typecast_feats, 
        int32_t *index_arr, int num_nodes, int32_t *dist_vector, int feature_len) {
    int warpId = threadIdx.x / DWARP_SIZE;
    int numWarps = blockDim.x / DWARP_SIZE;
    const bool warpLeader = (threadIdx.x % DWARP_SIZE == 0);
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        int64_t nodeId = index_arr[i];
        dist_vector[i] = INT_MAX;
        __syncthreads();
        // Align to warp factor
        for(int j = warpId; j < (num_centroids + numWarps - (num_centroids % numWarps)); j += numWarps) {
            if(j < num_centroids) {
                int dist = 0;
                for(int k = threadIdx.x % DWARP_SIZE; k < feature_len; k += DWARP_SIZE) {
                    //typecast_feats[i * feature_len + k] ^ dev_centroids[j * feature_len + k];
                    int32_t val = (typecast_feats[nodeId * feature_len + k] ^ centroids[j * feature_len + k]);
                    dist += __popc(val);
                }
                __syncwarp();
                for (int offset = 16; offset > 0; offset /= 2)
                    dist += __shfl_down_sync(FULL_MASK, dist, offset);
                __syncwarp();
                if(warpLeader) {
                    atomicMin(&dist_vector[i], dist);
                }
            }
        }
    }
}

__global__ void pick_max_distance(int32_t *dist_vector, int num_nodes, int *choice) 
{
    if(blockIdx.x > 0)
        return;
    
    __shared__ int max_dist;
    max_dist = 0;
    __syncthreads();
    for(int i = threadIdx.x; i < num_nodes; i += blockDim.x) {
        if(dist_vector[i] > max_dist)
            atomicMax(&max_dist, dist_vector[i]);
        __syncthreads();
        if(max_dist == dist_vector[i]) {
            atomicExch(choice, i);
        }
        __syncthreads();
    }
}

#define SHARED_MEM (32 * 256)
__global__ void compute_new_centroids(int32_t *dev_centroids, int num_centroids, int *centroid_count,
    int32_t *typecast_feats, int32_t *index_arr, int num_nodes, int32_t *dev_cluster, int feature_len) {
    __shared__ int bits_set[SHARED_MEM];
    for(int cur_centroid = blockIdx.x; cur_centroid < num_centroids; cur_centroid += gridDim.x) {
        // Reset bit distances for this centroid
        for(int index = threadIdx.x; index < SHARED_MEM; index += blockDim.x)
            bits_set[index] = 0;
        centroid_count[cur_centroid] = 0;
        __syncthreads();
        for(int i = 0; i < num_nodes; ++i) {
            int64_t nodeId = index_arr[i];
            // Find relevant node
            if(dev_cluster[i] == cur_centroid) {
                if(threadIdx.x == 0)
                    centroid_count[cur_centroid]++;
                __syncthreads();
                // Compute distance
                for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
                    for(int bit = 0; bit < 32; ++bit) {
                        if(typecast_feats[nodeId * feature_len + id] & (1 << bit))
                            bits_set[id * 32 + bit]++;
                    }
                }
            }
        }
        __syncthreads();
        // Compute new centroid based on obtained bit distances
        for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
            int32_t bitmask = 0;
            for(int bit = 0; bit < 32; ++bit) {
                if(bits_set[id * 32 + bit] > centroid_count[cur_centroid] / 2)
                    bitmask |= (1 << bit);
            }
            dev_centroids[cur_centroid * feature_len + id] = bitmask;
        }
    }
}

__global__ void create_mask_many(int *masks, int *vals, int num_centroids, int32_t *centroid_count, int32_t *typecast_feats, int32_t *index_arr, 
        int32_t *dev_cluster, int feature_len, int num_nodes, float threshold) {

    __shared__ float bits_set[SHARED_MEM];
    for(int cur_centroid = blockIdx.x; cur_centroid < num_centroids; cur_centroid += gridDim.x) {
        // Reset bit distances for this centroid
        for(int index = threadIdx.x; index < SHARED_MEM; index += blockDim.x)
            bits_set[index] = 0;
        centroid_count[cur_centroid] = 0;
        __syncthreads();
        for(int i = 0; i < num_nodes; ++i) {
            int64_t nodeId = index_arr[i];
            // Find relevant node
            if(dev_cluster[i] == cur_centroid) {
                if(threadIdx.x == 0)
                    centroid_count[cur_centroid]++;
                __syncthreads();
                // Compute distance
                for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
                    for(int bit = 0; bit < 32; ++bit) {
                        if(typecast_feats[nodeId * feature_len + id] & (1 << bit))
                            bits_set[id * 32 + bit]++;
                    }
                }
            }
        }
        __syncthreads();
        // Compute new masks based on obtained bit distances
        for(int id = threadIdx.x; id < feature_len; id += blockDim.x) {
            int32_t val = 0;
            int32_t masker = 0;
            for(int bit = 0; bit < 32; ++bit) {
                if(bits_set[id * 32 + bit] >= threshold * centroid_count[cur_centroid]) {
                    val |= (1 << bit);
                    masker |= (1 << bit);
                } else if(bits_set[id * 32 + bit] <= (1.0 - threshold) * centroid_count[cur_centroid]) {
                    masker |= (1 << bit);
                }
                /*printf("C%d, B%d: %f %f; %f %f; %d %d\n", cur_centroid, id * 32 + bit, 
                    bits_set[id * 32 + bit],
                    cur_centroid_num_nodes,
                    threshold * cur_centroid_num_nodes,
                    (1.0 - threshold) * cur_centroid_num_nodes, 
                    bits_set[id * 32 + bit] <= (1.0 - threshold) * cur_centroid_num_nodes,
                    bits_set[id * 32 + bit] >= threshold * cur_centroid_num_nodes);*/
            }
            masks[cur_centroid * feature_len + id] = masker;
            vals[cur_centroid * feature_len + id] = val;
        }
        __syncthreads();
    }
}

__global__ void check_feats_many(int32_t *feature_arr, int32_t *index_arr, int num_nodes, int32_t *dev_cluster, 
        int feature_len, int *masks, int *vals, long long unsigned *count, bool strict) {
    __shared__ long long unsigned ctr, ctr2;
    ctr = 0;
    ctr2 = 0;
    __syncthreads();
    for(int i = blockIdx.x; i < num_nodes; i += gridDim.x) {
        int64_t nodeId = index_arr[i];
        int clusterId = dev_cluster[i];
        for(int j = threadIdx.x; j < feature_len; j += blockDim.x) {
            int32_t val = feature_arr[nodeId * feature_len + j];
            if((val & masks[clusterId * feature_len + j]) == vals[clusterId * feature_len + j]) {
                atomicAdd(&ctr, __popc(masks[clusterId * feature_len + j]));
            }
            atomicAdd(&ctr2, __popc(masks[clusterId * feature_len + j]));
        }
        __syncthreads();\
        if(threadIdx.x == 0) {
            // All bits must match!
            if(strict && ctr == ctr2)
                atomicAdd(count, ctr);
            // Relaxed. Enough bits must match to justify the tracking overhead
            else if(!strict && ctr > feature_len)
                atomicAdd(count, ctr - feature_len);
        }
        __syncthreads();
        ctr = 0;
        __syncthreads();
    }
}
} // namespace ibp
#endif // IBP_PREPROC_KMEANS