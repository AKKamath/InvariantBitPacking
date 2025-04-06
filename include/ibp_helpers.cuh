#include <stdint.h>
#ifndef IBP_HELPERS_CUH
#define IBP_HELPERS_CUH
/**************** Commonly used constants ****************/
// GPU cacheline size
#define GPU_CL_SIZE 128
// Warp size
#define DWARP_SIZE 32
// Mask for all threads in warp
#define FULL_MASK 0xffffffff
#define ull unsigned long long
#ifdef __CUDA_ARCH__
template <typename T>
__inline__ __device__ int POPC(T val) {
    if constexpr(sizeof(T) == 1) { \
        unsigned val2 = 0; \
        val2 |= *(uint8_t*)&val; \
        return __popc(val2); \
    } else if constexpr(sizeof(T) == 2) { \
        uint16_t val2 = *(uint16_t*)&val; \
        return __popc((unsigned)val2); \
    } else if constexpr(sizeof(T) == 4) { \
        return __popc(*(uint32_t*)&val); \
    } else if constexpr(sizeof(T) == 8) { \
        return __popcll(*(uint64_t*)&val); \
    } else { \
        static_assert(sizeof(val) <= 4 || sizeof(val) == 8, "Data type must be 4 or 8 bytes"); \
    }
}

template <typename T>
__inline__ __device__ int CLZ(T val) {
    if constexpr(sizeof(T) == 1) {
        uint8_t val2 = *(uint8_t*)&val;
        return __clz((unsigned)val2);
    } else if constexpr(sizeof(val) == 2) {
        uint16_t val2 = *(uint16_t*)&val;
        return __clz((unsigned)val2);
    } else if constexpr(sizeof(val) == 4) {
        return __clz(*(uint32_t*)&val);
    } else if constexpr(sizeof(val) == 8) {
        return __clzll(*(uint64_t*)&val);
    } else {
        static_assert(sizeof(val) <= 4 ||
            sizeof(val) == 8, "Data type must be 4 or 8 bytes");
    }
}

#else
template <typename T>
__inline__ int POPC(T val) {
    if constexpr(sizeof(val) == 1) { \
        uint8_t val2 = *(uint8_t*)&val; \
        return __builtin_popcount((unsigned)val2); \
    } else if constexpr(sizeof(val) == 2) { \
        uint16_t val2 = *(uint16_t*)&val; \
        return __builtin_popcount((unsigned)val2); \
    } else if constexpr(sizeof(val) == 4) { \
        return __builtin_popcount(*(uint32_t*)&val); \
    } else if constexpr(sizeof(val) == 8) { \
        return __builtin_popcountll(*(uint64_t*)&val); \
    } else { \
        static_assert(sizeof(val) <= 4 || sizeof(val) == 8, "Data type must be 4 or 8 bytes"); \
    }
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

__device__ __forceinline__
void async_cp(int *shared, const int *global, int ints) {
    for(int i = 0; i < ints; i++) {
#if __CUDA_ARCH__ >= 800
        int* shared_ptr;
        asm volatile ("cvta.to.shared.u64 %0, %1;" :"=l" (shared_ptr) : "l" (shared + i));
        asm volatile ("cp.async.ca.shared.global [%0], [%1], 4;" :: "l" (shared_ptr), "l" (global + i));
#else
        shared[i] = global[i];
#endif
    }
}

__device__ __forceinline__
void async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__
void async_waitone() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;");
#endif
}

__device__ __forceinline__
void async_waitall() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;");
#endif
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
template <bool async=false>
__inline__ __device__ void memcpy_warp_4byte(WORD *dest, const WORD *src, int size)
{
    int threadId = threadIdx.x % DWARP_SIZE;
    // Offset: First element offset that is CL aligned
    const int align_offset = (GPU_CL_SIZE - (((uint64_t)src) % GPU_CL_SIZE)) / sizeof(WORD);
    // Padding: Amount to add to size to make it CL aligned
    // TODO: we can reduce this by align_offset to remove unneeded iterations
    int padding = ((GPU_CL_SIZE - (((uint64_t)size * sizeof(WORD)) % GPU_CL_SIZE)) % GPU_CL_SIZE) / sizeof(WORD);
    // Round up to nearest warp multiple
    padding = (padding + align_offset) % (GPU_CL_SIZE / sizeof(WORD));

    for(int k = threadId + align_offset; k < size + padding; k += DWARP_SIZE) {
        // Selective write to GPU memory
        if(k < size) {
            if constexpr(async)
                async_cp((int*)(((WORD *)dest) + k), (int*)((WORD *)src) + k, 1);
            else
                *(((WORD *)dest) + k) = *(((WORD *)src) + k);
        }
    }

    for(int k = threadId; k < align_offset; k += DWARP_SIZE) {
        // Selective write to GPU memory
        if(k >= 0) {
            if constexpr(async)
                async_cp((int*)(((WORD *)dest) + k), (int*)(((WORD *)src) + k), 1);
            else
                *(((WORD *)dest) + k) = *(((WORD *)src) + k);
        }
    }
    if constexpr(async)
        async_commit();
    __syncwarp();

}

typedef uint64_t WORD64;
__inline__ __device__ void memcpy_warp_8byte(WORD64 *dest, const WORD64 *src, int size)
{
    int threadId = threadIdx.x % DWARP_SIZE;
    // Offset: First element offset that is CL aligned
    const int align_offset = (32 * sizeof(WORD64) - (((uint64_t)src) % (32 * sizeof(WORD64)))) / sizeof(WORD64);
    // Padding: Amount to add to size to make it CL aligned
    // TODO: we can reduce this by align_offset to remove unneeded iterations
    const int padding = (32 * sizeof(WORD64) - (((uint64_t)size * sizeof(WORD64)) % (32 * sizeof(WORD64)))) / sizeof(WORD64);
    // Round up to nearest warp multiple
    int subval = (align_offset + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE;
    for(int k = threadId + align_offset - subval; k < size + align_offset + padding; k += DWARP_SIZE) {
        __syncwarp();
        // Selective write to GPU memory
        if(k >= 0 && k < size)
            *(((WORD64 *)dest) + k) = *(((WORD64 *)src) + k);
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
template <bool async=false, typename T>
__inline__ __device__ void memcpy_warp(T *dest, const T *src, size_t length)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) % sizeof(WORD) == 0,
        "Data type size must be 1 byte, 2 bytes, or multiple of 4 bytes");

    /*
    uint64_t bytes = length * sizeof(T);
    int elements = bytes / sizeof(WORD64);
    memcpy_warp_8byte((WORD64*)dest, (WORD64*)src, elements);
    dest = (T*)((WORD64*)dest + elements);
    src = (T*)((WORD64*)src + elements);
    length -= elements * sizeof(WORD64) / sizeof(T);
    __syncwarp();
    */

    uint64_t bytes = length * sizeof(T);
    int elements = bytes / sizeof(WORD);
    memcpy_warp_4byte<async>((WORD*)dest, (WORD*)src, elements);
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

template <typename T>
__inline__ __device__ void memcpy_block(T *dest, const T *src, size_t length)
{
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) % sizeof(WORD) == 0,
        "Data type size must be 1 byte, 2 bytes, or multiple of 4 bytes");

    int threadId = threadIdx.x;
    // 8-byte copies if possible
    if(((uint64_t)dest) % sizeof(WORD64) == 0 && ((uint64_t)src) % sizeof(WORD64) == 0) {
        uint64_t bytes = length * sizeof(T);
        int elements = bytes / sizeof(WORD64);
        // Offset: First element offset that is CL aligned
        const int align_offset = ((((uint64_t)src) % (32 * sizeof(WORD64)))) / sizeof(WORD64);
        for(int k = threadId - align_offset; k < elements; k += blockDim.x) {
            // Selective write to GPU memory
            if(k >= 0 && k < elements)
                *(((WORD64 *)dest) + k) = *(((WORD64 *)src) + k);
        }
        length -= elements * sizeof(WORD64) / sizeof(T);
        dest   += elements * sizeof(WORD64) / sizeof(T);
        src    += elements * sizeof(WORD64) / sizeof(T);
    }
    // 4-byte otherwise
    if(((uint64_t)dest) % sizeof(WORD) == 0 && ((uint64_t)src) % sizeof(WORD) == 0) {
        uint64_t bytes = length * sizeof(T);
        int elements = bytes / sizeof(WORD);
        // Offset: First element offset that is CL aligned
        const int align_offset = ((((uint64_t)src) % (32 * sizeof(WORD)))) / sizeof(WORD);
        for(int k = threadId - align_offset; k < elements; k += blockDim.x) {
            // Selective write to GPU memory
            if(k >= 0 && k < elements)
                *(((WORD *)dest) + k) = *(((WORD *)src) + k);
        }
        length -= elements * sizeof(WORD) / sizeof(T);
        dest   += elements * sizeof(WORD) / sizeof(T);
        src    += elements * sizeof(WORD) / sizeof(T);
    }

    // Manually copy the leftover non-4-byte-aligned parts
    if constexpr(sizeof(T) % sizeof(WORD) != 0) {
        // TODO: Optimize this
        int threadId = threadIdx.x;
        for(int k = threadId; k < length; k += blockDim.x) {
            dest[k] = src[k];
        }
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

__inline__ __device__ void __syncthreadsX()
{
    asm("bar.sync %0, %1;" :: "r"(threadIdx.y), "r"(blockDim.x));
}

// Exclusive scan: Sum of all elements in block to the left of this thread
template<typename T>
__inline__ __device__ T blkExclusiveScanSync(T val, T *shm_com)
{
    // Perform intra-warp exclusive scan first
    T initial_val = val;
    #pragma unroll
    for (int offset = 1; offset < DWARP_SIZE; offset <<= 1) {
        val += __shfl_up_sync(FULL_MASK, val, offset);
        // Needed because non-offset elements just add themselves
        if(threadIdx.x % DWARP_SIZE < offset)
            val /= 2;
    }
    // Last thread lets other threads in blk know value
    if(threadIdx.x % DWARP_SIZE == DWARP_SIZE - 1)
        shm_com[threadIdx.x / DWARP_SIZE] = val;

    __syncthreadsX();
    // Inter-warp scan now
    for(int i = threadIdx.x / DWARP_SIZE; i > 0; --i) {
        val += shm_com[i - 1];
    }
    __syncthreadsX();
    return val - initial_val;
}

#endif /* IBP_HELPERS_CUH */