#ifndef IBP_DECOMPRESS_HOST
#define IBP_DECOMPRESS_HOST
#include "ibp_helpers.cuh"
#include "ibp_decompress_kernel.cuh"
namespace ibp {
template <typename T, typename IndexT=void>
void decompress_fetch(T *output, T *input, int64_t num_vecs, int64_t vec_size, 
    T *mask, T *bitval, int32_t *bitmask, int compressed_len,
    IndexT *index_array = nullptr, cudaStream_t stream = 0) {
    
    const int NBLOCKS = 16;
    const int NTHREADS = 1024;

    // Get shmem info
    int shmem_size = 0, maxShmem, device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&maxShmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    auto decomp_kernel = &decompress_fetch_cpu_kernel<false, T, IndexT>;
    // TODO: Change maxShmem based on executing GPU. Relevant for heterogeneous GPU machines
    if(maxShmem >= 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META)) {
        shmem_size = 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META);
        decomp_kernel = &decompress_fetch_cpu_kernel<true, T, IndexT>;
        DPRINTF("Have enough shmem (alloc = %d, maxshmem = %d, vec_size = %d, required = %lu)\n", 
            shmem_size, maxShmem, vec_size, 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META));
    } else {
        shmem_size = maxShmem; //256 / 32 * 96 * sizeof(int32_t);
        decomp_kernel = &decompress_fetch_cpu_kernel<false, T, IndexT>;
        DPRINTF("Not enough shmem (alloc = %d, maxshmem = %d, vec_size = %d, required = %lu)\n", 
            shmem_size, maxShmem, vec_size, 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META));
    }
    // Need opt-in for large shmem allocations
    if (shmem_size >= 48 * 1024) {
        cudaFuncSetAttribute(decomp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        cudaCheckError();
    }
    decomp_kernel<<<NBLOCKS, NTHREADS, shmem_size, stream>>>(
        output, input, num_vecs, vec_size, mask, bitval, bitmask, shmem_size, 
        compressed_len, index_array);
    cudaCheckError();
    return;
}
} // namespace ibp
#endif // IBP_DECOMPRESS_HOST