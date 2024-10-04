#ifndef IBP_HELPERS_CUH
#define IBP_HELPERS_CUH
/**************** Commonly used constants ****************/
// GPU cacheline size
#define GPU_CL_SIZE 128
// Warp size
#define DWARP_SIZE 32
// Floats per cacheline
#define GPU_CL_FLOATS (GPU_CL_SIZE / sizeof(float))
// Mask for all threads in warp
#define FULL_MASK 0xffffffff
#define ull unsigned long long

// Optimized padded, aligned CPU -> GPU data copy function
// Performs best with 4+ byte elements, i.e., float32, int32, etc.
typedef uint32_t WORD; 
__inline__ __device__ void memcpy_warp_4byte(WORD *dest, const WORD *src, int size) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    // Offset: First element offset that is CL aligned
    const int align_offset = (GPU_CL_SIZE - (((uint64_t)src)  % GPU_CL_SIZE)) / sizeof(WORD);
    // Padding: Amount to add to size to make it CL aligned
    // TODO: we can reduce this by align_offset to remove unneeded iterations
    const int padding = (GPU_CL_SIZE - (((uint64_t)size * sizeof(WORD)) % GPU_CL_SIZE)) / sizeof(WORD);
    // Round up to nearest warp multiple
    int subval = (align_offset + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE;
    for(int k = threadId + align_offset - subval; k < size + align_offset + padding; k += DWARP_SIZE) {
        __syncwarp();
        // Selective write to GPU memory
        if(k >= 0 && k < size)
            *(((WORD *)dest) + k) = *(((WORD *)src) + k);
    }
}

// Wrapper for generic data type copies
// Only supports datatypes that are multiple of 4 sized or powers of two, 
// e.g., 1, 2, 4, 8, 12, 16, etc.
// 11 byte-sized datatype is not supported
template <typename T>
__inline__ __device__ void memcpy_warp(const T *dest, const T *src, size_t length) 
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) % sizeof(WORD) == 0, 
        "Data type size must be 1 byte, 2 bytes, or multiple of 4 bytes");
    uint64_t bytes = length * sizeof(T);
    int elements = bytes / sizeof(WORD);
    memcpy_warp_4byte((WORD*)dest, (WORD*)src, elements);
    __syncwarp();

    // Manually copy the leftover non-4-byte-aligned parts
    if constexpr(sizeof(T) % sizeof(WORD) != 0) {
        // TODO: Optimize this
        int offset = elements * sizeof(WORD) / sizeof(T);
        int threadId = threadIdx.x % DWARP_SIZE;
        for(int k = threadId + offset; k < length; k += DWARP_SIZE) {
            dest[k] = src[k];
        }
        __syncwarp();
    }
}

// This function needs to be phased out
template <typename T>
__inline__ __device__ void memcpy_warp_isspace(const T *dest, const T *src, size_t size) 
{
    memcpy_warp(dest, src, size);
}

// Inclusive scan: Sum of elements in warp to the left of this thread + this thread element
template<typename T>
__inline__ __device__ T warpInclusiveScanSync(unsigned mask, T val)
{
    #pragma unroll
    for (int offset = 1; offset < DWARP_SIZE; offset <<= 1) {
        val += __shfl_up_sync(mask, val, offset);
        // Needed because non-offset elements just add themselves
        if(threadIdx.x % DWARP_SIZE < offset)
            val /= 2;
    }
    return val;
}

// Exclusive scan: Sum of all elements in warp to the left of this thread
template<typename T>
__inline__ __device__ T warpExclusiveScanSync(unsigned mask, T val)
{
    T initial_val = val;
    #pragma unroll
    for (int offset = 1; offset < DWARP_SIZE; offset <<= 1) {
        val += __shfl_up_sync(mask, val, offset);
        // Needed because non-offset elements just add themselves
        if(threadIdx.x % DWARP_SIZE < offset)
            val /= 2;
    }
    return val - initial_val;
}

#endif /* IBP_HELPERS_CUH */