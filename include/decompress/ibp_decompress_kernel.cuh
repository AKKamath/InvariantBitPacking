#ifndef IBP_DECOMPRESS_KERNEL
#define IBP_DECOMPRESS_KERNEL
#include "ibp_helpers.cuh"
#include "ibp_decompress_dev.cuh"
#include <cuda/pipeline>

namespace ibp {

template <typename IndexT>
__global__ void range_kernel(IndexT *array, int start, int end)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i < end - start; i += blockDim.x * gridDim.x)
        array[i] = start + i;
}

template <bool FITS_SHMEM, int SHM_META, int SHM_WORK, bool ASYNC=false, typename T, typename IndexT = void>
__global__ void decompress_fetch_cpu_kernel(T *output, T *input, int64_t num_vecs,
    int64_t vec_size, T *dev_mask, T *dev_bitval, int32_t *bitmask, int shmem_size,
    int compressed_len, IndexT *index_array = nullptr, IndexT *offset_array = nullptr)
{
    // For some reason template datatype gives error
    extern __shared__ int temp_shmem[];
    // So typecast to template
    T *shmem = (T*)temp_shmem;
    // 32 elements for metadata, 64 elements for working data
    // = 96 elements per warp
    T *workspace;
    size_t start_offset = 0;
    int *async_bitmask = nullptr;
    if constexpr(ASYNC) {
        workspace = (T*)&shmem[(threadIdx.x / DWARP_SIZE) * (SHM_META + 2 * SHM_WORK) / sizeof(T)];
        shmem_size -= (blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (SHM_META + 2 * SHM_WORK);
        start_offset = (blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (SHM_META + 2 * SHM_WORK);

        async_bitmask = &temp_shmem[start_offset / sizeof(int) + (threadIdx.x / DWARP_SIZE)];
        shmem_size -= (blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (sizeof(int32_t));
        start_offset += (blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (sizeof(int32_t));
    } else {
        workspace = (T*)&shmem[(threadIdx.x / DWARP_SIZE) * (SHM_META + SHM_WORK) / sizeof(T)];
        // Retain shmem_size as the number of elements in shmem thingies
        shmem_size -= (blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (SHM_META + SHM_WORK);
        start_offset = (blockDim.x + DWARP_SIZE - 1) / DWARP_SIZE * (SHM_META + SHM_WORK);
    }
    // Convert bytes to elements per shm_mask/shm_bitval array
    shmem_size /= 2;
    T *shm_mask = (T*)&shmem[start_offset / sizeof(T)];
    T *shm_bitval = (T*)&shmem[start_offset / sizeof(T) + shmem_size / sizeof(T)];
    for(int i = threadIdx.x; i < shmem_size / sizeof(T); i += blockDim.x) {
        async_cp(&shm_mask[i], &dev_mask[i], sizeof(T) / sizeof(int));
        async_cp(&shm_bitval[i], &dev_bitval[i], sizeof(T) / sizeof(int));
    }
    async_commit();
    __syncthreads();
    async_waitall();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    int read_bitmask = -1;
    // Go through node list
    for(int i = warpId; i < num_vecs; i += numWarps) {
        __syncwarp();
        int64_t input_ind = i;
        int64_t output_ind = i;
        if constexpr(!std::is_same<IndexT, void>::value) {
            if(index_array != nullptr)
                input_ind = index_array[i];
            if(offset_array != nullptr)
                output_ind = offset_array[i];
        }
        if constexpr(ASYNC) {
            async_commit();
            async_waitall();
            __syncwarp();
            // First iteration, must read directly
            if(read_bitmask == -1)
                read_bitmask = bitmask[input_ind / 32];
            else
                read_bitmask = *async_bitmask;
            __syncwarp();

            if(i + numWarps < num_vecs && threadIdx.x % DWARP_SIZE == 0) {
                int64_t new_ind = i + numWarps;
                if constexpr(!std::is_same<IndexT, void>::value) {
                    new_ind = index_array[i + numWarps];
                }
                async_cp(async_bitmask, &bitmask[new_ind / 32], 1);
            }
        } else {
            read_bitmask = bitmask[input_ind / 32];
        }
        // Decompress and write data
        if(read_bitmask & (1 << (input_ind % 32))) {
            decompress_fetch_cpu<FITS_SHMEM, SHM_META, SHM_WORK, ASYNC>(
                &output[output_ind * vec_size], &input[input_ind * vec_size],
                shm_mask, shm_bitval, vec_size, compressed_len,
                workspace, dev_mask, dev_bitval, shmem_size);
        } else {
            memcpy_warp<ASYNC>(&output[output_ind * vec_size],
                &input[input_ind * vec_size], vec_size);
        }
    }
}


template <typename T, typename IndexT = void>
__global__ void decompress_fake_kernel(T *output, T *input, int64_t num_vecs,
    int64_t vec_size, T *dev_mask, T *dev_bitval, int32_t *bitmask, int shmem_size,
    int compressed_len, IndexT *index_array = nullptr, IndexT *offset_array = nullptr)
{
    // For some reason template datatype gives error
    extern __shared__ int temp_shmem[];
    // So typecast to template
    T *shmem = (T*)temp_shmem;
    // 32 elements for metadata, 64 elements for working data
    // = 96 elements per warp
    T *workspace = (T*)&shmem[(threadIdx.x / DWARP_SIZE) * vec_size];
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    int laneId = threadId % DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_vecs; i += numWarps) {
        __syncwarp();
        int64_t input_ind = i;
        int64_t output_ind = i;
        if constexpr(!std::is_same<IndexT, void>::value) {
            if(index_array != nullptr)
                input_ind = index_array[i];
            if(offset_array != nullptr)
                output_ind = offset_array[i];
        }
        // Decompress and write data
        if(bitmask[input_ind / 32] & (1 << (input_ind % 32))) {
            // Read input
            for(int iter = laneId; iter < compressed_len; iter += DWARP_SIZE)
                workspace[iter] = input[input_ind * vec_size + iter];
            //memcpy_warp(workspace, &input[input_ind * vec_size], compressed_len);
            __syncwarp();
            // Write to output
            for(int iter = laneId; iter < vec_size; iter += DWARP_SIZE)
                output[output_ind * vec_size + iter] = workspace[iter % compressed_len];
        } else {
            memcpy_warp(&output[output_ind * vec_size],
                &input[input_ind * vec_size], vec_size);
        }
    }
}


template <typename T, typename IndexT = void>
__global__ void decompress_fake_kernel2(T *output, T *input, int64_t num_vecs,
    int64_t vec_size, T *dev_mask, T *dev_bitval, int32_t *bitmask, int shmem_size,
    int compressed_len, IndexT *index_array = nullptr, IndexT *offset_array = nullptr)
{
    // For some reason template datatype gives error
    extern __shared__ int temp_shmem[];
    // So typecast to template
    T *shmem = (T*)temp_shmem;
    // 32 elements for metadata, 64 elements for working data
    // = 96 elements per warp
    T *workspace = (T*)&shmem[(threadIdx.x / DWARP_SIZE) * vec_size];
    __syncthreads();

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    int laneId = threadId % DWARP_SIZE;
    // Go through node list
    for(int i = warpId; i < num_vecs; i += numWarps) {
        __syncwarp();
        int64_t input_ind = i;
        int64_t output_ind = i;
        if constexpr(!std::is_same<IndexT, void>::value) {
            if(index_array != nullptr)
                input_ind = index_array[i];
            if(offset_array != nullptr)
                output_ind = offset_array[i];
        }
        // Decompress and write data
        if(bitmask[input_ind / 32] & (1 << (input_ind % 32))) {
            // Read input
            for(int iter = laneId; iter < compressed_len; iter += DWARP_SIZE)
                workspace[iter] = input[input_ind * vec_size + iter];
            //memcpy_warp(workspace, &input[input_ind * vec_size], compressed_len);
            __syncwarp();
            // Write to output
            //for(int iter = laneId; iter < vec_size; iter += DWARP_SIZE)
            //    output[output_ind * vec_size + iter] = workspace[iter % compressed_len];
        } else {
            //memcpy_warp(&output[output_ind * vec_size],
            //    &input[input_ind * vec_size], vec_size);
        }
    }
}

template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
    const DType* const input, const int64_t input_len,
    const int64_t feature_size, const IdType* const index,
    const int64_t output_len, DType* const output,
    const int64_t* permutation = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < output_len) {
    int64_t col = threadIdx.x;
    int64_t in_row = out_row_index;
    if constexpr(!std::is_same<IdType, void>::value)
        in_row = index[out_row_index];
    //assert(in_row >= 0 && in_row < input_len);
    const int64_t idx_offset =
        ((uint64_t)(&input[in_row * feature_size]) % 128) /
        sizeof(DType);
    col = col - idx_offset;
    const auto out_row =
        permutation ? permutation[out_row_index] : out_row_index;
    while (col < feature_size) {
      if (col >= 0)
        output[out_row * feature_size + col] =
            input[in_row * feature_size + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}


template <bool FITS_SHMEM, typename T, typename IndexT = void>
__global__ void decompress_fetch_cpu_tb_kernel(T *output, T *input, int64_t num_vecs,
    int64_t vec_size, T *dev_mask, T *dev_bitval, int32_t *bitmask, int shmem_size,
    int compressed_len, int64_t SHM_META, int64_t SHM_WORK, IndexT *index_array = nullptr,
    IndexT *offset_array = nullptr)
{
    // For some reason template datatype gives error
    extern __shared__ int temp_shmem[];
    // So typecast to template
    T *shmem = (T*)temp_shmem;

    int offset = 0;
    T *workspace = (T*)&shmem[(threadIdx.y * (SHM_META + SHM_WORK)) / sizeof(T)];
    offset = (blockDim.y * (SHM_META + SHM_WORK)) / sizeof(T);
    T *shm_comm = (T*)&shmem[offset + threadIdx.y * blockDim.x / DWARP_SIZE];
    offset += blockDim.y * blockDim.x / DWARP_SIZE;
    // Retain shmem_size as the number of elements in shmem thingies
    shmem_size -= offset * sizeof(T);
    // Convert bytes to elements per shm_mask/shm_bitval array
    shmem_size /= 2;
    T *shm_mask = (T*)&shmem[offset];
    T *shm_bitval = (T*)&shmem[offset + shmem_size / sizeof(T)];
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    for(int i = threadIdx.x + threadIdx.y * blockDim.x; i < shmem_size / sizeof(T); i += blockDim.x * blockDim.y) {
        cuda::memcpy_async(&shm_mask[i], &dev_mask[i], sizeof(T), pipe);
        cuda::memcpy_async(&shm_bitval[i], &dev_bitval[i], sizeof(T), pipe);
        //shm_mask[i] = dev_mask[i];
        //shm_bitval[i] = dev_bitval[i];
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    pipe.consumer_release();
    __syncthreads();

    int blockId = blockIdx.x * blockDim.y + threadIdx.y;
    int numBlocks = gridDim.x * blockDim.y;
    // Go through node list
    for(int i = blockId; i < num_vecs; i += numBlocks) {
        int64_t in_index = i;
        int64_t out_index = i;
        if constexpr(!std::is_same<IndexT, void>::value) {
            if(index_array != nullptr)
                in_index = index_array[i];
            if(offset_array != nullptr)
                out_index = offset_array[i];
        }
        // Decompress and write data
        if(bitmask[in_index / 32] & (1 << (in_index % 32))) {
            decompress_fetch_blk_cpu<FITS_SHMEM>(&output[out_index * vec_size],
                &input[in_index * vec_size], shm_mask, shm_bitval, vec_size,
                compressed_len, workspace, shm_comm, SHM_META, SHM_WORK, dev_mask, dev_bitval,
                shmem_size);
        } else {
            memcpy_block(&output[out_index * vec_size], &input[in_index * vec_size], vec_size);
        }
    }
}
} // namespace ibp
#endif // IBP_DECOMPRESS_KERNEL