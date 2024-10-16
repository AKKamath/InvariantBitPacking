#ifndef IBP_MISC_KERNELS
#define IBP_MISC_KERNELS

#include "ibp_helpers.cuh"
#include "ibp_misc_dev.cuh"
namespace ibp {
// TODO: Why is this threadblock-parallel instead of warp?
template <typename T, typename IndexT = void, typename CtrT = void>
__global__ void check_compress_size_kernel(T *input, int64_t num_vecs, 
    int64_t vec_size, T *mask, T *bitval, int64_t *compressed_size, 
    IndexT *index_array = nullptr, CtrT *size_ctr = nullptr)
{
    for(ull i = blockIdx.x; i < num_vecs; i += gridDim.x) {
        __syncthreads();
        // Get which vector to look at
        ull vectorId = i;
        // Compile-time this away if unused
        if constexpr(!std::is_same<IndexT, void>::value)
            if(index_array != nullptr)
                vectorId = index_array[i];

        // Call calculation function
        compressed_size[i] = check_compress_size_blk(&input[vectorId * vec_size], 
            vec_size, mask, bitval);
        
        // Compile-time this away if unused
        if constexpr(!std::is_same<CtrT, void>::value)
            if(size_ctr != nullptr && threadIdx.x == 0) {
                atomicAdd(size_ctr, (CtrT)compressed_size[i]);
            }
    }
}

} // namespace ibp

#endif // IBP_MISC_KERNELS