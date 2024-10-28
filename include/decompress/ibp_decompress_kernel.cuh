#ifndef IBP_DECOMPRESS_KERNEL
#define IBP_DECOMPRESS_KERNEL
#include "ibp_helpers.cuh"
#include "ibp_decompress_dev.cuh"

namespace ibp {

template <bool FITS_SHMEM, int SHM_META, int SHM_WORK, typename T, typename IndexT = void>
__global__ void decompress_fetch_cpu_kernel(T *output, T *input, int64_t num_vecs, 
    int64_t vec_size, T *dev_mask, T *dev_bitval, int32_t *bitmask, int shmem_size, 
    int compressed_len, IndexT *index_array = nullptr)
{
    extern __shared__ int shmem[];
    // 32 elements for metadata, 64 elements for working data
    // = 96 elements per warp
    T *workspace = (T*)&shmem[(threadIdx.x / DWARP_SIZE) * (SHM_META + SHM_WORK) / sizeof(int)];
    // Retain shmem_size as the number of elements in shmem thingies
    shmem_size -= (blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (SHM_META + SHM_WORK);
    // Convert bytes to elements per shm_mask/shm_bitval array
    shmem_size /= 2;
    T *shm_mask = (T*)&shmem[(blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (SHM_META + SHM_WORK) / sizeof(int)];
    T *shm_bitval = (T*)&shmem[(blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (SHM_META + SHM_WORK) / sizeof(int) + shmem_size / sizeof(int)];
    for(int i = threadIdx.x; i < shmem_size / sizeof(T); i += blockDim.x) {
        shm_mask[i] = dev_mask[i];
        shm_bitval[i] = dev_bitval[i];
    }
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_vecs; i += numWarps) {
        __syncwarp();
        int64_t index = i;
        if constexpr(!std::is_same<IndexT, void>::value)
            if(index_array != nullptr)
                index = index_array[i];
        // Decompress and write data
        if(bitmask[index / 32] & (1 << (index % 32))) {
            decompress_fetch_cpu<FITS_SHMEM, SHM_META, SHM_WORK>(&output[i * vec_size], 
                &input[index * vec_size], shm_mask, shm_bitval, vec_size, 
                compressed_len, workspace, dev_mask, dev_bitval, shmem_size);
        } else {
            memcpy_warp(&output[i * vec_size], &input[index * vec_size], vec_size);
        }
    }
}
} // namespace ibp
#endif // IBP_DECOMPRESS_KERNEL