#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <chrono>
#include <omp.h>
#include "ibp_helpers.cuh"

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(a, b) std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()

#define GB (32 * 1024L * 1024L * 1024L)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define DWARP_SIZE 32
#define GPU_CL_SIZE 128
#define D_WORD float
#define GPU_CL_FLOATS (GPU_CL_SIZE / sizeof(D_WORD))

char *cache;
char *cache_cpu;
size_t cache_size;
int64_t *gpu_indices;

void init_cache(size_t size, size_t index_size) {
    cache_size = size;
    gpuErrchk(cudaMalloc(&cache, size));
    gpuErrchk(cudaMallocHost(&cache_cpu, size));
    gpuErrchk(cudaMalloc(&gpu_indices, index_size * sizeof(int64_t)));
}

__global__ void check_kernel(char* cache, const char* data, const int64_t* indices, int64_t dim, size_t len, int cache_size, int *flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = i / 32;
    int thread = i % 32;
    for(int j = warp; j < len; j += blockDim.x * gridDim.x / 32) {
        uint64_t loc = indices[j];
        for(int k = thread * 4; k < dim; k += 32 * 4) {
            int *cache_arr = (int *)&cache[(dim * j) % (cache_size - dim) + k];
            int *data_arr = (int *)&data[(dim * loc) + k];
            if(*cache_arr != *data_arr) {
                printf("Got %d, expected %d at %d, %d\n", *cache_arr, *data_arr, j, k);
                *flag = 1;
                return;
            }
            if(*flag)
                return;
        }
        if(*flag)
            return;
    }
}

void reset()
{
    cudaMemset(cache, 0, cache_size);
}

void check(char *data, int64_t *indices, size_t element_size, size_t array_size, int *flag) 
{
    cudaMemset(flag, 0, sizeof(int));
    check_kernel <<< 64, 512 >>> (cache, data, indices, element_size, array_size, cache_size, flag);
    gpuErrchk(cudaDeviceSynchronize());
    int h_flag;
    cudaMemcpy(&h_flag, flag, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_flag)
        printf("Mismatch\n");
    //else
    //    printf("All good\n");
}

#define RUN_TEST(function, name) \
    reset(); \
    START = TIME_NOW; \
    function(cpu_buffer, gpu_indices, i, array_size); \
    END = TIME_NOW; \
    TIME = TIME_DIFF(START, END); \
    printf("%s: Element %ld bytes | %.4f MBPS | %.0f ns\n", \
        name, i, (i * array_size / 1000000.0) / (TIME / 1000000000.0), TIME); \
    check(cpu_buffer, gpu_indices, i, array_size, flag);

#define RUN_TEST_CPU(function, name) \
    reset(); \
    cudaDeviceSynchronize(); \
    START = TIME_NOW; \
    function(cpu_buffer, index_buffer, i, array_size); \
    END = TIME_NOW; \
    TIME = TIME_DIFF(START, END); \
    printf("%s: Element %ld bytes | %.4f MBPS | %.0f ns\n", \
        name, i, (i * array_size / 1000000.0) / (TIME / 1000000000.0), TIME); \
    check(cpu_buffer, gpu_indices, i, array_size, flag);

void transfer_cpu(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    //#pragma omp parallel for
    for(int i = 0; i < array_size; ++i) {
        uint64_t loc = indices[i];
        gpuErrchk(cudaMemcpyAsync(&cache[(element_size * i) % (cache_size - element_size)], &data[element_size * loc], element_size, cudaMemcpyHostToDevice, 0));
    }
    cudaDeviceSynchronize();
}

void transfer_cpu2(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    #pragma omp parallel for
    for(int i = 0; i < array_size; ++i) {
        uint64_t loc = indices[i];
        memcpy(&cache_cpu[(element_size * i) % (cache_size - element_size)], &data[element_size * loc], element_size);
    }
    gpuErrchk(cudaMemcpyAsync(cache, cache_cpu, element_size * array_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void transfer_gpu_kernel_int(char* cache, const char* data, const int64_t* indices, int64_t dim, size_t len, int cache_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = i / 32;
    int thread = i % 32;
    for(int j = warp; j < len; j += blockDim.x * gridDim.x / 32) {
        uint64_t loc= indices[j];
        for(int k = thread * 4; k < dim; k += 32 * 4) {
            int *cache_arr = (int *)&cache[((dim * j)) % (cache_size - dim) + k];
            int *data_arr = (int *)&data[(dim * loc) + k];
            *cache_arr = *data_arr;
        }
    }
}

void transfer_gpu_int(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    transfer_gpu_kernel_int <<< blks, threads >>> (cache, data, indices, element_size, array_size, cache_size);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void transfer_gpu_kernel_int64(char* cache, const char* data, const int64_t* indices, int64_t dim, size_t len, int cache_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = i / 32;
    int thread = i % 32;
    for(int j = warp; j < len; j += blockDim.x * gridDim.x / 32) {
        uint64_t loc= indices[j];
        for(int k = thread * 8; k < dim; k += 32 * 8) {
            uint64_t *cache_arr = (uint64_t *)&cache[((dim * j)) % (cache_size - dim) + k];
            uint64_t *data_arr = (uint64_t *)&data[(dim * loc) + k];
            *cache_arr = *data_arr;
        }
    }
}

void transfer_gpu_int64(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    if(element_size % 8 != 0) {
        printf("Element size must be multiple of 8\n");
        return;
    }
    transfer_gpu_kernel_int64 <<< blks, threads >>> (cache, data, indices, element_size, array_size, cache_size);
    gpuErrchk(cudaDeviceSynchronize());
}


__global__ void transfer_gpu_kernel_align(int* cache, const int* data, const int64_t* indices, size_t dim, size_t len, int cache_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = i / 32;
    int thread = i % 32;
    int64_t size = dim / sizeof(int);
    for(int j = warp; j < len; j += blockDim.x * gridDim.x / 32) {
        uint64_t loc = indices[j];
        int offset = ((((uint64_t)&data[(size * loc)]) % 128)) / sizeof(int);
        for(int k = thread - offset; k < size; k += 32) {
            if(k >= 0) {
                int *cache_arr = (int *)&cache[((size * j)) % ((cache_size - dim) / sizeof(int)) + k];
                int *data_arr = (int *)&data[(size * loc) + k];
                *cache_arr = *data_arr;
            }
        }
    }
}

void transfer_gpu_align(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    transfer_gpu_kernel_align <<< blks, threads >>> ((int*)cache, (int*)data, indices, element_size, array_size, cache_size);
    gpuErrchk(cudaDeviceSynchronize());
}


__global__ void transfer_gpu_kernel_align2(int* cache, const int* data, const int64_t* indices, size_t dim, size_t len, int cache_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = i / 32;
    int thread = i % 32;
    int size = dim / 4;
    for(int j = warp; j < len; j += blockDim.x * gridDim.x / 32) {
        uint64_t loc = indices[j];
        int align = (( (uint64_t)&data[(size * loc)]) % 128) / sizeof(int);
        __syncwarp();
        for(int k = thread - align; k < size; k += 32) {
          if(k >= 0) {// Avoid underflow
            int *cache_arr = (int *)(cache + ((size * j)) % ((cache_size - dim) / 4) + k);
            int *data_arr = (int *)(data + (size * loc) + k);
            *cache_arr = *data_arr;
          }
        }
        __syncwarp();
    }
}

void transfer_gpu_align2(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    transfer_gpu_kernel_align2 <<< blks, threads >>> ((int*)cache, (int*)data, indices, element_size, array_size, cache_size);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void transfer_gpu_kernel_align_plus(char* cache, const char* data, const int64_t* indices, int64_t dim, size_t len, int cache_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = i / 32;
    int thread = i % 32;
    for(int j = warp; j < len; j += blockDim.x * gridDim.x / 32) {
        uint64_t loc = indices[j];
        int offset = (128 - (((uint64_t)&data[(dim * loc)]) % 128));
        int padding = (128 - (dim % 128));
        for(int k = thread * 4 + offset; k < dim + offset + padding && (dim * loc) + k < GB; k += 32 * 4) {
            int *cache_arr = (int *)&cache[((dim * j)) % (cache_size - dim) + k];
            int data_arr = *(int *)&data[(dim * loc) + k];
            if(k < dim)
                *cache_arr = data_arr;
            __syncwarp();
        }
        
        for(int k = thread * 4 + (offset - 128); k < offset; k += 32 * 4) {
            int *cache_arr = (int *)&cache[(dim * j) % (cache_size - dim) + k];
            int data_arr = *(int *)&data[(dim * loc) + k];
            if(k >= 0)
                *cache_arr = data_arr;
            __syncwarp();
        }
    }
}

void transfer_gpu_align_plus(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    //
    transfer_gpu_kernel_align_plus <<< blks, threads >>> (cache, data, indices, element_size, array_size, cache_size);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void transfer_kernel(char* cache, const char* data, 
        const int64_t* indices, int64_t dim, size_t len, int cache_size) {
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / DWARP_SIZE;
    int numWarps = (blockDim.x * gridDim.x) / DWARP_SIZE;
    // Go through node list
    for(int j = warpId; j < len; j += numWarps) {
        int64_t nodeId= indices[j];
        // If we have extra space on both sides, we can use optimized padded copy
        memcpy_warp((int*)&cache[(j * dim) % (cache_size - dim)], (int*)&data[nodeId * dim], dim / 4);
        __syncwarp();
    }
}

void transfer_gpu_align_plus2(char *data, int64_t *indices, size_t element_size, size_t array_size, int blks = 32, int threads = 512) {
    //
    transfer_kernel <<< blks, threads >>> (cache, data, indices, element_size, array_size, cache_size);
    gpuErrchk(cudaDeviceSynchronize());
}

#define CACHE_LINE_SIZE 128
template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernelAligned(
    const DType* const array, const int64_t num_feat, const IdType* const index,
    const int64_t length, const int64_t arr_len, DType* const out, const int64_t cache_size,
    const int64_t* perm = nullptr) {
  int64_t out_row_index = blockIdx.x * blockDim.y + threadIdx.y;

  const int64_t stride = blockDim.y * gridDim.x;

  while (out_row_index < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row_index];
    assert(in_row >= 0 && in_row < arr_len);
    const int64_t idx_offset =
        ((uint64_t)(&array[in_row * num_feat]) % CACHE_LINE_SIZE) /
        sizeof(DType);
    col = col - idx_offset;
    const auto out_row = perm ? perm[out_row_index] : out_row_index;
    while (col < num_feat) {
      if (col >= 0)
        out[(out_row * num_feat + col) % cache_size] = array[in_row * num_feat + col];
      col += blockDim.x;
    }
    out_row_index += stride;
  }
}

void transfer_dgl(char *data, int64_t *indices, int64_t element_size, int64_t array_size, int blks = 32, int threads = 512) {
    int64_t num_feat = element_size / sizeof(uint64_t);
    while (static_cast<int64_t>(threads) >= 2 * num_feat) {
      threads /= 2;
      blks *= 2;
    }
    IndexSelectMultiKernelAligned<uint64_t, int64_t>  <<< blks, threads >>> 
        ((uint64_t*)data, num_feat, indices, array_size, 
        GB / sizeof(int64_t), (uint64_t*)cache, cache_size / sizeof(uint64_t));
    gpuErrchk(cudaDeviceSynchronize());
}

#define ARR_SIZE 8
const long int array[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

__global__ void reset_data(char *array, size_t size) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    for(size_t i = threadId; i < size; i += blockDim.x * gridDim.x)
        array[i] = i % 256;
}

int main(int argc, char *argv[]) 
{
    int device = 0;
    if(argc > 1)
        device = atoi(argv[1]);
    cudaSetDevice(device);

    fprintf(stderr, "Set device\n");

    int64_t array_size = 10000;
    const int64_t CACHE_SIZE = 1024 * 1024 * 1024;
    int64_t *index_buffer = (int64_t *)malloc(array_size * sizeof(int64_t));
    fprintf(stderr, "Allocated index buffer\n");

    auto START = TIME_NOW, END = TIME_NOW;
    float TIME = TIME_DIFF(START, END);
    
    init_cache(CACHE_SIZE, array_size);
    fprintf(stderr, "Init cache\n");

    fprintf(stderr, "Alloc host data\n");
    char *cpu_buffer = (char *)malloc(GB * sizeof(char));
    gpuErrchk(cudaHostRegister(cpu_buffer, GB * sizeof(char), cudaHostRegisterDefault));

    fprintf(stderr, "Register data\n");
    //char *cpu_buffer;
    //cudaHostAlloc((void **)&cpu_buffer, GB, cudaHostAllocDefault);
    reset_data<<<64, 512>>>(cpu_buffer, GB);
    gpuErrchk(cudaDeviceSynchronize());

    fprintf(stderr, "Misalignment %ld\n", (128 - (((uint64_t)cpu_buffer) % 128)));
    int *flag;
    cudaMalloc(&flag, sizeof(int));
    //for(int64_t i = 256; i <= 4 * 1024; i *= 2) {
    for(int index = 0; index < ARR_SIZE; ++index) {
        int64_t i = array[index];
        for(int j = 0; j < array_size; ++j) {
            index_buffer[j] = rand() % (GB / i);
        }
        // Sort to improve accesses. Usually done in critical path, but all need it
        std::sort(index_buffer, index_buffer + array_size);
        gpuErrchk(cudaMemcpy(gpu_indices, index_buffer, array_size * sizeof(int64_t), cudaMemcpyHostToDevice));
        
        RUN_TEST_CPU(transfer_cpu, "CPU copy");
        RUN_TEST_CPU(transfer_cpu2, "CPU index");
        RUN_TEST(transfer_gpu_int, "GPU copy");
        RUN_TEST(transfer_gpu_align, "Aligned GPU copy");
        //RUN_TEST(transfer_gpu_int64, "Bigint");
        //RUN_TEST(transfer_gpu_align2, "AlignedDeux");
        RUN_TEST(transfer_gpu_align_plus, "Aligned + Padded");
        //RUN_TEST(transfer_gpu_align_plus2, "AlignedPaddedAgain");

        /*START = TIME_NOW;
        transfer_dgl(cpu_buffer, gpu_indices, i, array_size);
        END = TIME_NOW;
        TIME = TIME_DIFF(START, END);
        printf("DGL: Element %ld bytes | %.4f MBPS | %.0f ns\n", i, (i * array_size / 1000000.0) / (TIME / 1000000000.0), TIME);
    */
    }
    /*for(int index = 0; index < ARR_SIZE; ++index) {
        int64_t i = array[index];
        for(int j = 0; j < array_size; ++j) {
            index_buffer[j] = rand() % (GB / i);
        }
        gpuErrchk(cudaMemcpy(gpu_indices, index_buffer, array_size * sizeof(int64_t), cudaMemcpyHostToDevice));
        for(int blks = 8; blks <= 108; blks += 8) {
            START = TIME_NOW;
            transfer_gpu_align_plus2(cpu_buffer, gpu_indices, i, array_size, blks);
            END = TIME_NOW;
            TIME = TIME_DIFF(START, END);
            printf("AlignedPadded_%d: Element %ld bytes | %.4f MBPS | %.0f ns\n", blks, i, (i * array_size / 1000000.0) / (TIME / 1000000000.0), TIME);
        }
    }*/
}