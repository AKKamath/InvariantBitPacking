#ifndef IBP_COMPRESS_KERNEL
#define IBP_COMPRESS_KERNEL
#include "ibp_helpers.cuh"
#include "../misc/ibp_misc_dev.cuh"
#include "ibp_compress_dev.cuh"

namespace ibp {

template <typename T, typename IndexT = void, typename CtrT = void, typename SizeT = void>
__global__ void compress_inplace_kernel(T *output, T *input, int64_t num_vecs, 
    int64_t vec_size, T *mask, T *bitval, T *workspace,
    int32_t *bitmask = nullptr, IndexT *index_array = nullptr, 
    CtrT *comp_ctr = nullptr, SizeT *compress_size = nullptr)
{
    int64_t vec_bytes = vec_size * sizeof(T);
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int laneId = threadIdx.x % DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    T *myworkspace = &workspace[warpId * vec_size];
    for(ull i = warpId; i < num_vecs; i += numWarps) {
        ull index = i;
        if constexpr(!std::is_same<IndexT, void>::value)
            if(index_array != nullptr)
                index = index_array[i];
        __syncwarp();
        // Check compressed size first
        int64_t compressed = check_compress_size_warp(&input[index * vec_size], 
            vec_size, mask, bitval);
        __syncwarp();
        if(compressed != vec_bytes) {
            // Compressed write to myworkspace
            compress_and_write(myworkspace, &input[index * vec_size], vec_size,
                mask, bitval);
            __syncwarp();
            for(int j = laneId; j < vec_size; j += DWARP_SIZE) {
                output[i * vec_size + j] = myworkspace[j];
                myworkspace[j] = 0;
            }
            if(laneId == 0) {
                if(bitmask != nullptr)
                    atomicOr(&bitmask[i / 32], 1 << (i % 32));
                if constexpr(!std::is_same<CtrT, void>::value)
                    if(comp_ctr != nullptr)
                        atomicAdd(comp_ctr, (CtrT)1);
            }
        } else {
            memcpy_warp(&output[i * vec_size], 
                &input[index * vec_size], vec_size);
        }
        if constexpr(!std::is_same<SizeT, void>::value)
            if(laneId == 0 && compress_size != nullptr) {
                atomicAdd(compress_size, (SizeT)compressed);
            }
    }
}
} // namespace ibp
#endif // IBP_COMPRESS_KERNEL