#ifndef IBP_DECOMPRESS_HOST
#define IBP_DECOMPRESS_HOST
#include "ibp_helpers.cuh"
#include "ibp_decompress_kernel.cuh"
#include <cub/cub.cuh>
#include <math.h>
namespace ibp {
template <typename T, typename IndexT=void>
void decompress_fetch(T *output, T *input, int64_t num_vecs, int64_t vec_size, 
    T *mask, T *bitval, int32_t *bitmask, int compressed_len,
    IndexT *index_array = nullptr, int64_t max_num_elems = -1, 
    cudaStream_t stream = 0, int blks = 32, int threads = 512, int impl=0) {

    constexpr int SHM_META = 128;
    constexpr int SHM_WORK = 64 * sizeof(T);
    
    int NBLOCKS = blks;
    int NTHREADS = threads;

    IndexT *offset_array = nullptr;
    // Sort index array to improve memory access locality
    if constexpr(!std::is_same<IndexT, void>::value) {
        IndexT *d_keys_in = index_array, *d_values_in;
        IndexT *d_keys_out, *d_values_out;
        int end_bit = sizeof(IndexT) * 8;
        if(max_num_elems != -1)
            end_bit = int(log2(max_num_elems)) + 1;
        
        cudaMallocAsync(&d_values_in, sizeof(IndexT) * num_vecs, stream);
        range_kernel<<<16, 1024, 0, stream>>>(d_values_in, 0, num_vecs);
        cudaCheckError();

        cudaMallocAsync(&d_keys_out, sizeof(IndexT) * num_vecs, stream);
        cudaMallocAsync(&d_values_out, sizeof(IndexT) * num_vecs, stream);
        cudaCheckError();

        void     *d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_vecs, 
            0, end_bit, stream);
        cudaCheckError();

        // Allocate temporary storage
        cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
        cudaCheckError();

        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_vecs,
            0, end_bit, stream);
        index_array = d_keys_out;
        offset_array = d_values_out;
        cudaFreeAsync(d_values_in, stream);
        cudaFreeAsync(d_temp_storage, stream);
        cudaCheckError();
    }

    // Get shmem info
    int shmem_size = 0, maxShmem, device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&maxShmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Warp-parallel implementation
    if(impl == 0) {
        auto decomp_kernel = &decompress_fetch_cpu_kernel<false, SHM_META, SHM_WORK, T, IndexT>;
        // TODO: Change maxShmem based on executing GPU. Relevant for heterogeneous GPU machines
        if(maxShmem >= 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META)) {
            shmem_size = 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META);
            decomp_kernel = &decompress_fetch_cpu_kernel<true, SHM_META, SHM_WORK, T, IndexT>;
            DPRINTF("Have enough shmem (alloc = %d, maxshmem = %d, vec_size = %d, required = %lu)\n", 
                shmem_size, maxShmem, vec_size, 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META));
        } else {
            shmem_size = maxShmem; //256 / 32 * 96 * sizeof(int32_t);
            decomp_kernel = &decompress_fetch_cpu_kernel<false, SHM_META, SHM_WORK, T, IndexT>;
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
            compressed_len, index_array, offset_array);
        cudaCheckError();
    } 
    // Threadblock-parallel implementation
    else if(impl == 1) {
        while (NTHREADS >= 2 * vec_size) {
            NTHREADS /= 2;
            NBLOCKS *= 2;
        }
        auto decomp_kernel = &decompress_fetch_cpu_tb_kernel<false, SHM_META, SHM_WORK, T, IndexT>;
        decomp_kernel<<<NBLOCKS, NTHREADS, 0, stream>>>(
            output, input, num_vecs, vec_size, mask, bitval, bitmask, shmem_size, 
            compressed_len, index_array);
        cudaCheckError();
    }
    // DGL implementation; To remove later.
    if(impl == 2) {
        const int64_t input_len = num_vecs;
        const int64_t return_len = num_vecs;
        const int64_t original_feature_size = vec_size;
        const auto aligned_feature_size =
            sizeof(int) * original_feature_size / sizeof(int);
        T* input_ptr = input;
        T* ret_ptr = output;

        const IndexT* index_sorted_ptr = index_array;
        const int64_t* permutation_ptr = nullptr;
        
        if constexpr(!std::is_same<IndexT, void>::value)
            permutation_ptr = offset_array;
        constexpr int BLOCK_SIZE = 1024;
        dim3 block(BLOCK_SIZE, 1);
        while (static_cast<int64_t>(block.x) >= 2 * aligned_feature_size) {
            block.x >>= 1;
            block.y <<= 1;
        }
        const dim3 grid(std::min(
            (return_len + block.y - 1) / block.y,
            (1l << 20l) / BLOCK_SIZE));
        IndexSelectMultiKernelAligned<<<grid, block, 0>>>(input_ptr, input_len,
            aligned_feature_size, index_sorted_ptr, return_len, ret_ptr,
            permutation_ptr);
    }

    // Free intermediate arrays
    if constexpr(!std::is_same<IndexT, void>::value) {
        cudaFreeAsync(index_array, stream);
        cudaFreeAsync(offset_array, stream);
    }
    return;
}
} // namespace ibp
#endif // IBP_DECOMPRESS_HOST