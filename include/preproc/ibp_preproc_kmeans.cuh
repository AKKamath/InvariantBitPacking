#ifndef IBP_PREPROC_KMEANS
#define IBP_PREPROC_KMEANS
#include "ibp_helpers.cuh"
namespace ibp {
// K-means++ clustering
template<typename T, typename IndexT = void>
__global__ void calc_distances(T *input_arr, ull num_vecs, ull vec_size, 
    T *centroids, int num_centroids, int32_t *dist_vector, IndexT *index_arr = nullptr) {
    int warpId = threadIdx.x / DWARP_SIZE;
    int numWarps = blockDim.x / DWARP_SIZE;
    const bool warpLeader = (threadIdx.x % DWARP_SIZE == 0);
    for(int i = blockIdx.x; i < num_vecs; i += gridDim.x) {
        int64_t nodeId = i;
        if constexpr(!std::is_same<IndexT, void>::value)
            nodeId = index_arr[i];
        dist_vector[i] = INT_MAX;
        __syncthreads();
        // Align to warp factor
        for(int j = warpId; j < (num_centroids + numWarps - (num_centroids % numWarps)); j += numWarps) {
            if(j < num_centroids) {
                int dist = 0;
                for(int k = threadIdx.x % DWARP_SIZE; k < vec_size; k += DWARP_SIZE) {
                    //input_arr[i * vec_size + k] ^ dev_centroids[j * vec_size + k];
                    T val = (input_arr[nodeId * vec_size + k] ^ centroids[j * vec_size + k]);
                    dist += POPC(val);
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

__global__ void pick_max_distance(int32_t *dist_vector, int num_vecs, int *choice) 
{
    if(blockIdx.x > 0)
        return;
    
    __shared__ int max_dist;
    max_dist = 0;
    __syncthreads();
    for(int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        if(dist_vector[i] > max_dist)
            atomicMax(&max_dist, dist_vector[i]);
        __syncthreads();
        if(max_dist == dist_vector[i]) {
            atomicExch(choice, i);
        }
        __syncthreads();
    }
}

/**
 * @brief Kernel function to K-means++ cluster vectors based on input data and centroids.
 *
 * @tparam T Data type of the input array and centroids.
 * @tparam IndexT Data type of the index array, defaults to void.
 * 
 * @param input_arr Pointer to the input array containing the data vectors.
 * @param num_vecs Number of vectors in the input array.
 * @param vec_size Size of each vector in the input array.
 * @param masks Pointer to the array containing masks.
 * @param bitvals Pointer to the array containing values.
 * @param num_centroids Number of centroids used for classification.
 * @param dev_cluster Pointer to the device array where the cluster assignments will be stored.
 * @param index_arr Pointer to the index array, defaults to nullptr.
 */
template<typename T, typename IndexT = void>
__global__ void cluster_vecs(T *input_arr, ull num_vecs, ull vec_size, 
    T *masks, T *bitvals, int num_centroids, int32_t *dev_cluster, IndexT *index_arr = nullptr) {
    __shared__ int min_dist;
    int warpId = threadIdx.x / DWARP_SIZE;
    int numWarps = blockDim.x / DWARP_SIZE;
    const bool warpLeader = (threadIdx.x % DWARP_SIZE == 0);
    for(int i = blockIdx.x; i < num_vecs; i += gridDim.x) {
        min_dist = INT_MAX;
        int64_t nodeId = i;
        // Compile away if unused
        if constexpr(!std::is_same<IndexT, void>::value)
            nodeId = index_arr[i];
        __syncthreads();
        // Align to warp factor
        for(int j = warpId; j < (num_centroids + numWarps - (num_centroids % numWarps)); j += numWarps) {
            int dist = 0;
            if(j < num_centroids) {
                for(int k = threadIdx.x % DWARP_SIZE; k < vec_size; k += DWARP_SIZE) {
                    //input_arr[i * vec_size + k] ^ dev_centroids[j * vec_size + k];
                    T val = (input_arr[nodeId * vec_size + k] & masks[j * vec_size + k]) ^ bitvals[j * vec_size + k];
                    dist += POPC(val);
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

// TODO: Is SHARED_MEM a limitation?
#define SHARED_MEM (32 * 256)
template<typename T, typename IndexT = void>
__global__ void compute_new_centroids(T *input_arr, ull num_vecs, ull vec_size,
    T *dev_centroids, int num_centroids, int *centroid_count, 
    int32_t *dev_cluster, IndexT *index_arr = nullptr) {
    __shared__ int bits_set[SHARED_MEM];
    for(int cur_centroid = blockIdx.x; cur_centroid < num_centroids; cur_centroid += gridDim.x) {
        // Reset bit distances for this centroid
        for(int index = threadIdx.x; index < SHARED_MEM; index += blockDim.x)
            bits_set[index] = 0;
        centroid_count[cur_centroid] = 0;
        __syncthreads();
        for(int i = 0; i < num_vecs; ++i) {
            int64_t nodeId = i;
            if constexpr(!std::is_same<IndexT, void>::value)
                nodeId = index_arr[i];
            // Find relevant node
            if(dev_cluster[i] == cur_centroid) {
                if(threadIdx.x == 0)
                    centroid_count[cur_centroid]++;
                __syncthreads();
                // Compute distance
                for(int id = threadIdx.x; id < vec_size; id += blockDim.x) {
                    for(long long bit = 0; bit < sizeof(T) * 8; ++bit) {
                        if(input_arr[nodeId * vec_size + id] & (1ll << bit))
                            bits_set[id * sizeof(T) * 8 + bit]++;
                    }
                }
            }
        }
        __syncthreads();
        // Compute new centroid based on obtained bit distances
        for(int id = threadIdx.x; id < vec_size; id += blockDim.x) {
            T bitmask = 0;
            for(long long bit = 0; bit < sizeof(T) * 8; ++bit) {
                if(bits_set[id * sizeof(T) * 8 + bit] > centroid_count[cur_centroid] / 2)
                    bitmask |= (1ll << bit);
            }
            dev_centroids[cur_centroid * vec_size + id] = bitmask;
        }
    }
}

template<typename T, typename IndexT = void>
__global__ void create_mask_many(T *input_arr, ull vec_size, ull num_vecs, 
    T *masks, T *bitvals, int num_centroids, int32_t *centroid_count, 
    int32_t *dev_cluster, float threshold, IndexT *index_arr = nullptr) {

    __shared__ float bits_set[SHARED_MEM];
    for(int cur_centroid = blockIdx.x; cur_centroid < num_centroids; cur_centroid += gridDim.x) {
        // Reset bit distances for this centroid
        for(int index = threadIdx.x; index < SHARED_MEM; index += blockDim.x)
            bits_set[index] = 0;
        centroid_count[cur_centroid] = 0;
        __syncthreads();
        for(int i = 0; i < num_vecs; ++i) {
            int64_t nodeId = i;
            if constexpr(!std::is_same<IndexT, void>::value)
                nodeId = index_arr[i];
            // Find relevant node
            if(dev_cluster[i] == cur_centroid) {
                if(threadIdx.x == 0)
                    centroid_count[cur_centroid]++;
                __syncthreads();
                // Compute distance
                for(int id = threadIdx.x; id < vec_size; id += blockDim.x) {
                    for(ull bit = 0; bit < sizeof(T) * 8; ++bit) {
                        if(input_arr[nodeId * vec_size + id] & (1ll << bit))
                            bits_set[id * sizeof(T) * 8 + bit]++;
                    }
                }
            }
        }
        __syncthreads();
        // Compute new masks based on obtained bit distances
        for(int id = threadIdx.x; id < vec_size; id += blockDim.x) {
            T val = 0;
            T masker = 0;
            for(ull bit = 0; bit < sizeof(T) * 8; ++bit) {
                if(bits_set[id * sizeof(T) * 8 + bit] >= threshold * centroid_count[cur_centroid]) {
                    val |= (1ll << bit);
                    masker |= (1ll << bit);
                } else if(bits_set[id * sizeof(T) * 8 + bit] <= (1.0 - threshold) * centroid_count[cur_centroid]) {
                    masker |= (1ll << bit);
                }
                /*printf("C%d, B%d: %f %f; %f %f; %d %d\n", cur_centroid, id * 32 + bit, 
                    bits_set[id * 32 + bit],
                    cur_centroid_num_vecs,
                    threshold * cur_centroid_num_vecs,
                    (1.0 - threshold) * cur_centroid_num_vecs, 
                    bits_set[id * 32 + bit] <= (1.0 - threshold) * cur_centroid_num_vecs,
                    bits_set[id * 32 + bit] >= threshold * cur_centroid_num_vecs);*/
            }
            masks[cur_centroid * vec_size + id] = masker;
            bitvals[cur_centroid * vec_size + id] = val;
        }
        __syncthreads();
    }
}

template<typename T, typename IndexT = void>
__global__ void check_feats_many(T *input_arr, int vec_size, int num_vecs, T *masks, T *vals,
    int32_t *dev_cluster, long long unsigned *count, IndexT *index_arr = nullptr) {
    __shared__ long long unsigned ctr, ctr2;
    ctr = 0;
    ctr2 = 0;
    __syncthreads();
    for(int i = blockIdx.x; i < num_vecs; i += gridDim.x) {
        int64_t nodeId = i;
        if constexpr(!std::is_same<IndexT, void>::value)
            nodeId = index_arr[i];
        int clusterId = dev_cluster[i];
        for(int j = threadIdx.x; j < vec_size; j += blockDim.x) {
            T val = input_arr[nodeId * vec_size + j];
            if((val & masks[clusterId * vec_size + j]) == vals[clusterId * vec_size + j]) {
                atomicAdd(&ctr, POPC(masks[clusterId * vec_size + j]));
            }
            atomicAdd(&ctr2, POPC(masks[clusterId * vec_size + j]));
        }
        __syncthreads();
        if(threadIdx.x == 0 && ctr > vec_size)
            atomicAdd(count, ctr - vec_size);
        __syncthreads();
        ctr = 0;
        __syncthreads();
    }
}
} // namespace ibp
#endif // IBP_PREPROC_KMEANS