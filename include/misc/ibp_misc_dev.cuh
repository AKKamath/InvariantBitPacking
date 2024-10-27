#ifndef IBP_MISC_DEV
#define IBP_MISC_DEV

#include "ibp_helpers.cuh"
namespace ibp { 
/**
 * @brief Computes the compressed size of a vector in threadblock-parallel form.
 *
 * This function calculates the size of the compressed vector based on the input vector,
 * mask, and bit value provided.
 *
 * @tparam T The data type of the input vector, mask, and bit value.
 * @param input Pointer to the input vector.
 * @param vec_size The size of the input vector.
 * @param mask Pointer to the mask used for compression.
 * @param bitval Pointer to the bit value used for compression.
 * @return The compressed size of the vector.
 */
template <typename T>
__device__ int64_t check_compress_size_blk(T *input, int64_t vec_size, T *mask, T *bitval)
{
    __shared__ ull workspace;
    const int64_t vec_bytes = vec_size * sizeof(T);
    workspace = 0;
    __syncthreads();
    // Loop through vector elements
    for(ull j = threadIdx.x; j < vec_size; j += blockDim.x) {
        if((input[j] & mask[j]) == bitval[j]) {
            atomicAdd(&workspace, (ull)POPC(mask[j]));
        }
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        // Calc bytes for compressed data
        // Metadata: 1 bit per data element
        int64_t metadata_size = BITS_TO_BYTES(vec_size);
        int64_t data_size = vec_bytes - workspace / 8;
        // Datatype-align
        metadata_size = (metadata_size + sizeof(T) - 1) / sizeof(T) * sizeof(T);
        data_size = (data_size + sizeof(T) - 1) / sizeof(T) * sizeof(T);
        int64_t comp_size = metadata_size + data_size;
        workspace = min(comp_size, vec_bytes);
    }
    __syncthreads();
    return workspace;
}

/**
 * @brief Computes the compressed size of a vector in warp-parallel form.
 *
 * This function calculates the size of the compressed vector based on the input vector,
 * mask, and bit value provided.
 *
 * @tparam T The data type of the input vector, mask, and bit value.
 * @param input Pointer to the input vector.
 * @param vec_size The size of the input vector.
 * @param mask Pointer to the mask used for compression.
 * @param bitval Pointer to the bit value used for compression.
 * @return The compressed size of the vector.
 */
template <typename T>
__inline__ __device__ int64_t check_compress_size_warp(T *input, int64_t vec_size, T *mask, T *bitval)
{
    // CUDA doesn't like INT64 in this function. I dont know why, but it breaks.
    const int32_t vec_bytes = vec_size * sizeof(T);
    const int32_t T_size = sizeof(T);
    ull laneId = threadIdx.x % DWARP_SIZE;
    int32_t ctr = 0;
    for(ull j = laneId; j < vec_size; j += DWARP_SIZE) {
        if((input[j] & mask[j]) == bitval[j]) {
            ctr += POPC(mask[j]);
        }
    }
    // Get sum of ctr value from warp
    ctr = warpInclusiveScanSync(FULL_MASK, ctr);
    long compressed = vec_bytes;
    if(laneId == DWARP_SIZE - 1) {
        // Calc bytes for compressed data
        // Metadata: 1 bit per data element
        int32_t metadata_size = BITS_TO_BYTES(vec_size);
        int32_t data_size = vec_bytes - ctr / 8;
        // Datatype align
        metadata_size = (metadata_size + T_size - 1) / T_size * T_size;
        int32_t data_size2 = ((data_size + T_size) - 1) / T_size * T_size;
        int32_t comp_size = metadata_size + data_size2;
        if(comp_size < vec_bytes)
            compressed = comp_size;
    }
    __syncwarp();
    return __shfl_sync(FULL_MASK, compressed, DWARP_SIZE - 1);
}
} // namespace ibp

#endif // IBP_MISC_DEV