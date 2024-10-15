#include "ibp_helpers.cuh"
namespace ibp {

template <typename T, typename IndexT>
__global__ void check_compress_size_kernel(T *input, int64_t num_vecs, 
    int64_t vec_size, T *mask, T *bitval, int64_t *compressed_size, 
    IndexT *index_array = nullptr)
{
    __shared__ ull ctr;
    // Bytes per element
    const int64_t vec_bytes = vec_size * sizeof(T);
    for(ull i = blockIdx.x; i < num_vecs; i += gridDim.x) {
        ctr = 0;
        __syncthreads();
        // Get which vector to look at
        IndexT vectorId = i;
        if(index_array != nullptr) 
            vectorId = index_array[i];
        for(IndexT j = threadIdx.x; j < vec_size; j += blockDim.x) {
            T val = input[vectorId * vec_size + j];
            if((val & mask[j]) == bitval[j]) {
                ull count = 0;
                POPC(count, mask[j]);
                atomicAdd(&ctr, count);
            }
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            // Calc bytes for compressed data
            int64_t metadata_size = BITS_TO_BYTES((vec_bytes + sizeof(T) - 1) / sizeof(T));
            int64_t data_size = vec_bytes - ctr / 8;
            // Datatype-align
            metadata_size = (metadata_size + sizeof(T) - 1) / sizeof(T) * sizeof(T);
            data_size = (data_size + sizeof(T) - 1) / sizeof(T) * sizeof(T);
            int64_t comp_size = metadata_size + data_size;
            compressed_size[i] = min(comp_size, vec_bytes);
        }
        __syncthreads();
    }
}

} // namespace ibp