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
    cudaStream_t stream = 0, int blks = 32, int threads = 512, int impl=-1,
    IndexT *index_array = nullptr, IndexT *offset_array = nullptr) {

    // Decide whether to perform TB-parallel or warp-parallel compression
    if(impl == -1) {
        // If it's 2x or more compressed, the TB-parallel doesn't
        // generate enough memory accesses fast enough
        /*if(vec_size / compressed_len >= 2)
            impl = 0; // Warp-parallel
        else
            impl = 1; // TB-parallel*/
        impl = 5;
        // If the compression is less than 20% and the vector is very, very large, then TB-parallel is superior
        if((float)vec_size / (float)compressed_len < 1.2 && vec_size >= 2 * threads)
            impl = 1;
    }

    int NBLOCKS = blks;
    int NTHREADS = threads;

    // Get shmem info
    int shmem_size = 0, maxShmem, device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&maxShmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Warp-parallel implementation
    if(impl == 0 || impl == 5) {
        constexpr int SHM_META = 32 * sizeof(T);
        constexpr int SHM_WORK = 64 * sizeof(T);
        const int SHM_ASYNC = (impl == 5 ? sizeof(int) : 0);
        auto decomp_kernel = &decompress_fetch_cpu_kernel<false, SHM_META, SHM_WORK, false, T, IndexT>;
        size_t shmem_size = 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META);
        if (impl == 5)
            shmem_size += NTHREADS / DWARP_SIZE * (SHM_ASYNC + SHM_WORK);
        // TODO: Change maxShmem based on executing GPU. Relevant for heterogeneous GPU machines
        if(maxShmem >= shmem_size) {
            if (impl == 0)
                decomp_kernel = &decompress_fetch_cpu_kernel<true, SHM_META, SHM_WORK, false, T, IndexT>;
            else
                decomp_kernel = &decompress_fetch_cpu_kernel<true, SHM_META, SHM_WORK, true, T, IndexT>;

            DPRINTF("Have enough shmem (alloc = %d, maxshmem = %d, vec_size = %ld, required = %lu)\n",
                shmem_size, maxShmem, vec_size, 2 * vec_size * sizeof(T) + NTHREADS / DWARP_SIZE * (SHM_WORK + SHM_META + SHM_ASYNC));
        } else {
            shmem_size = maxShmem; //256 / 32 * 96 * sizeof(int32_t);
            if (impl == 0)
                decomp_kernel = &decompress_fetch_cpu_kernel<false, SHM_META, SHM_WORK, false, T, IndexT>;
            else
                decomp_kernel = &decompress_fetch_cpu_kernel<false, SHM_META, SHM_WORK, true, T, IndexT>;
            DPRINTF("Not enough shmem (alloc = %d, maxshmem = %d, vec_size = %ld, required = %lu)\n",
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
        dim3 THREADS(NTHREADS, 1, 1);
        while (THREADS.x > compressed_len && THREADS.x > 32) {
            THREADS.x /= 2;
            THREADS.y *= 2;
        }
        int SHM_META = 32 * sizeof(T) * THREADS.x / DWARP_SIZE;
        int SHM_WORK = 64 * sizeof(T) * THREADS.x / DWARP_SIZE;

        // Working space shared memory
        int SHM_SPACE = THREADS.y * (SHM_WORK + SHM_META);
        int SHM_MASK = vec_size * sizeof(T);
        int SHM_BITVAL = vec_size * sizeof(T);
        int SHM_COMM = NTHREADS / DWARP_SIZE * sizeof(T);
        int SHM_TOT = SHM_SPACE + SHM_MASK + SHM_BITVAL + SHM_COMM;

        auto decomp_kernel = &decompress_fetch_cpu_tb_kernel<false, T, IndexT>;
        // TODO: Change maxShmem based on executing GPU. Relevant for heterogeneous GPU machines
        if(maxShmem >= SHM_TOT) {
            shmem_size = SHM_TOT;
            decomp_kernel = &decompress_fetch_cpu_tb_kernel<true, T, IndexT>;
            DPRINTF("Have enough shmem (alloc = %d, maxshmem = %d, vec_size = %ld, required = %d)\n",
                shmem_size, maxShmem, vec_size, SHM_TOT);
        } else {
            shmem_size = maxShmem; //256 / 32 * 96 * sizeof(int32_t);
            decomp_kernel = &decompress_fetch_cpu_tb_kernel<false, T, IndexT>;
            DPRINTF("Not enough shmem (alloc = %d, maxshmem = %d, vec_size = %ld, required = %d)\n",
                shmem_size, maxShmem, vec_size, SHM_TOT);
        }
        // Need opt-in for large shmem allocations
        if (shmem_size >= 48 * 1024) {
            cudaFuncSetAttribute(decomp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            cudaCheckError();
        }
        decomp_kernel<<<NBLOCKS, THREADS, shmem_size, stream>>>(
            output, input, num_vecs, vec_size, mask, bitval, bitmask, shmem_size,
            compressed_len, SHM_META, SHM_WORK, index_array, offset_array);
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
    if(impl == 3) {
        shmem_size = NTHREADS / DWARP_SIZE * vec_size * sizeof(T);
        auto decomp_kernel = &decompress_fake_kernel<T, IndexT>;
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
    if(impl == 4) {
        shmem_size = NTHREADS / DWARP_SIZE * vec_size * sizeof(T);
        auto decomp_kernel = &decompress_fake_kernel2<T, IndexT>;
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
    return;
}
} // namespace ibp
#endif // IBP_DECOMPRESS_HOST