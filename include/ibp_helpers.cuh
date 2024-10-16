#include <stdint.h>
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
#ifdef __CUDA_ARCH__
#define POPC(count, val) \
    if constexpr(sizeof(val) == 1) { \
        uint8_t val2 = *(uint8_t*)val; \
        count = __popc((unsigned)val2); \
    } else if constexpr(sizeof(val) == 2) { \
        uint16_t val2 = *(uint16_t*)val; \
        count = __popc((unsigned)val2); \
    } else if constexpr(sizeof(val) == 4) { \
        count = __popc(*(uint32_t*)&val); \
    } else if constexpr(sizeof(val) == 8) { \
        count = __popcll(*(uint64_t*)&val); \
    } else { \
        static_assert(sizeof(val) <= 4 || sizeof(val) == 8, "Data type must be 4 or 8 bytes"); \
    }

#else
#define POPC(count, val) \
    if constexpr(sizeof(val) == 1) { \
        uint8_t val2 = *(uint8_t*)val; \
        count = __builtin_popcount((unsigned)val2); \
    } else if constexpr(sizeof(val) == 2) { \
        uint16_t val2 = *(uint16_t*)val; \
        count = __builtin_popcount((unsigned)val2); \
    } else if constexpr(sizeof(val) == 4) { \
        count = __builtin_popcount(*(uint32_t*)&val); \
    } else if constexpr(sizeof(val) == 8) { \
        count = __builtin_popcountll(*(uint64_t*)&val); \
    } else { \
        static_assert(sizeof(val) <= 4 || sizeof(val) == 8, "Data type must be 4 or 8 bytes"); \
    }
#endif

#define BITS_TO_BYTES(x) ((x + 7) / 8)
// If debug prints are desired, define IBP_DEBUG_PRINT
// then declare "bool ibp_print_debug" in a source file
#ifdef IBP_DEBUG_PRINT
extern bool ibp_print_debug;
#define DPRINTF(fmt, ...) {if(ibp_print_debug) printf(fmt, ##__VA_ARGS__); }
#else
#define DPRINTF(fmt, ...)
#endif // IBP_DEBUG_PRINT
// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

/**
 * @brief Optimized padded, aligned CPU to GPU data copy function.
 *
 * This function uses warp-level primitives to efficiently copy data from CPU to GPU memory,
 * ensuring that the data is cache-line-aligned.
 *
 * @param dest Pointer to the destination memory on the GPU.
 * @param src Pointer to the source memory on the CPU.
 * @param size Number of 4-byte elements to copy.
 *
 * @note The function assumes that the source and destination pointers are
 *       4-byte-aligned.
 */
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

/**
 * @brief Wrapper for generic data type copies. It only supports data types 
 * that are multiples of 4 in size or powers of two, such as 1, 2, 4, 8, 12, 16, etc.
 * For example, 11 byte-sized data types are not supported.
 * 
 * @tparam T Data type
 * @param dest Destination memory pointer
 * @param src Source memory pointer
 * @param length Number of elements to copy
 */
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