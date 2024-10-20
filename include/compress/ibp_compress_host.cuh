#ifndef IBP_COMPRESS_HOST
#define IBP_COMPRESS_HOST
#include "ibp_helpers.cuh"
#include "ibp_compress_kernel.cuh"
namespace ibp {

template <typename T, typename IndexT=void, typename CtrT=void, typename SizeT=void>
void compress_inplace(T *output, T *input, int64_t num_vecs, int64_t vec_size, 
    T *mask, T *bitval, int32_t *bitmask = nullptr, IndexT *index_array = nullptr, 
    CtrT *comp_ctr = nullptr, SizeT *comp_size = nullptr, cudaStream_t stream = 0) {
    if(bitmask != nullptr) {
        cudaMemset(bitmask, 0, (num_vecs + 31) / 32 * sizeof(int32_t));
    }

    const int NBLOCKS = 32;
    const int NTHREADS = 512;
    T *working_space;
    cudaMalloc(&working_space, (NBLOCKS * NTHREADS / DWARP_SIZE * vec_size) * sizeof(T));
    cudaMemset(working_space, 0, (NBLOCKS * NTHREADS / DWARP_SIZE * vec_size) * sizeof(T));
    cudaCheckError();
    compress_inplace_kernel<<<NBLOCKS, NTHREADS, 0, stream>>>(output, input, num_vecs, vec_size, 
        mask, bitval, working_space, bitmask, index_array, comp_ctr, comp_size);
    cudaCheckError();
    cudaFree(working_space);
    cudaCheckError();
    return;
}

template <typename T, typename IndexT=void>
void compress_condensed(T *output, T *input, int64_t num_vecs, 
    int64_t vec_size, T *mask, T *bitval, int64_t *comp_offsets, 
    int32_t *bitmask = nullptr, IndexT *index_array = nullptr, cudaStream_t stream = 0) {
    if(bitmask != nullptr) {
        cudaMemset(bitmask, 0, (num_vecs + 31) / 32 * sizeof(int32_t));
    }

    const int NBLOCKS = 32;
    const int NTHREADS = 512;
    cudaCheckError();
    compress_condensed_kernel<<<NBLOCKS, NTHREADS, 0, stream>>>(output, input, num_vecs, 
        vec_size, mask, bitval, comp_offsets, bitmask, index_array);
    cudaCheckError();
    return;
}
} // namespace ibp
#endif // IBP_COMPRESS_HOST