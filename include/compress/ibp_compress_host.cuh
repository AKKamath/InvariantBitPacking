#ifndef IBP_COMPRESS_HOST
#define IBP_COMPRESS_HOST
#include "ibp_helpers.cuh"
#include "ibp_compress_kernel.cuh"
namespace ibp {

template <typename T, typename IndexT=void>
ull compress_inplace(T *output, T *input, int64_t num_vecs, 
    int64_t vec_size, T *mask, T *bitval, int32_t *bitmask = nullptr, 
    IndexT *index_array = nullptr) {
    if(bitmask != nullptr) {
        cudaMemset(bitmask, 0, (num_vecs + 31) / 32 * sizeof(int32_t));
    }

    const int NBLOCKS = 32;
    const int NTHREADS = 512;
    T *working_space;
    cudaMalloc(&working_space, (NBLOCKS * NTHREADS / DWARP_SIZE * vec_size) * sizeof(T));
    cudaMemset(working_space, 0, (NBLOCKS * NTHREADS / DWARP_SIZE * vec_size) * sizeof(T));
    ull *comp_ctr;
    cudaMalloc(&comp_ctr, sizeof(ull));
    cudaMemset(comp_ctr, 0, sizeof(ull));
    cudaCheckError();
    compress_inplace_kernel<<<NBLOCKS, NTHREADS>>>(output, input, num_vecs, vec_size, 
        mask, bitval, working_space, bitmask, index_array, comp_ctr);
    cudaDeviceSynchronize();
    cudaCheckError();
    // Get number of compressed vectors
    ull host_comp_count = 0;
    cudaMemcpy(&host_comp_count, comp_ctr, sizeof(ull), cudaMemcpyDeviceToHost);
    cudaFree(working_space);
    cudaCheckError();
    return host_comp_count;
}
} // namespace ibp
#endif // IBP_COMPRESS_HOST