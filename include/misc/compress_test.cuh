#include "nvcomp.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/cascaded.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/gzip.h"
#include "nvcomp/lz4.hpp"
#include "nvcomp/snappy.hpp"
#include "nvcomp/zstd.hpp"
#include "ndzip/ndzip.hh"
#include "ndzip/cuda.hh"

#define RUN_COMPRESSION(ALGO) \
    {\
        size_t max_out_bytes; \
        nvcompBatched##ALGO##CompressGetMaxOutputChunkSize(chunk_size, nvcompBatched##ALGO##DefaultOpts, &max_out_bytes); \
        cudaCheckError(); \
        printf("%s: Maxoutput: %zu\n", #ALGO, max_out_bytes); \
        size_t temp_bytes; \
        nvcompBatched##ALGO##CompressGetTempSize(batch_size, chunk_size, nvcompBatched##ALGO##DefaultOpts, &temp_bytes); \
        cudaCheckError(); \
        void* device_temp_ptr;\
        cudaMalloc(&device_temp_ptr, temp_bytes); \
        cudaCheckError(); \
        void** device_compressed_ptrs; \
        cudaMalloc(&device_compressed_ptrs, sizeof(size_t) * batch_size); \
        cudaCheckError(); \
        cudaMemcpyAsync( \
            device_compressed_ptrs, host_compressed_ptrs,  \
            sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream); \
        cudaMemset(device_compressed_bytes, 0, sizeof(size_t) * batch_size); \
        cudaCheckError(); \
        nvcompStatus_t comp_res = nvcompBatched##ALGO##CompressAsync( \
            device_uncompressed_ptrs,\
            device_uncompressed_bytes,\
            chunk_size, \
            batch_size, \
            device_temp_ptr, \
            temp_bytes, \
            device_compressed_ptrs, \
            device_compressed_bytes, \
            nvcompBatched##ALGO##DefaultOpts, \
            stream); \
        cudaStreamSynchronize(stream); \
        size_t total_compressed = 0; \
        if (comp_res != nvcompSuccess) \
        { \
            std::cerr << "Failed compression!" << std::endl; \
            assert(comp_res == nvcompSuccess); \
        } else { \
            cudaMemcpy(host_compressed_bytes, device_compressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost); \
            cudaCheckError(); \
            for(size_t i = 0; i < batch_size; ++i) { \
                total_compressed += host_compressed_bytes[i]; \
            } \
            printf("%s: Uncompressed bytes: %ld, compressed bytes: %zu, ratio: %f\n",  \
                #ALGO, in_bytes, total_compressed, (float)in_bytes / (float)total_compressed); \
        } \
        nvcompBatched##ALGO##GetDecompressSizeAsync( \
            device_compressed_ptrs, \
            device_compressed_bytes, \
            device_uncompressed_bytes, \
            batch_size, \
            stream); \
        size_t decomp_temp_bytes; \
        nvcompBatched##ALGO##DecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes);\
        void * device_decomp_temp;\
        cudaMalloc(&device_decomp_temp, decomp_temp_bytes); \
        cudaCheckError(); \
        nvcompStatus_t* device_statuses; \
        cudaMalloc(&device_statuses, sizeof(nvcompStatus_t)*batch_size); \
        cudaCheckError(); \
        size_t* device_actual_uncompressed_bytes; \
        cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t)*batch_size); \
        cudaCheckError(); \
        cudaStreamSynchronize(stream); \
        auto decomp_start = TIME_NOW; \
        nvcompStatus_t decomp_res = nvcompBatched##ALGO##DecompressAsync( \
            device_compressed_ptrs,  \
            device_compressed_bytes,  \
            device_uncompressed_bytes,  \
            device_actual_uncompressed_bytes,  \
            batch_size, \
            device_decomp_temp,  \
            decomp_temp_bytes,  \
            device_uncompressed_ptrs,  \
            device_statuses,  \
            stream); \
        cudaStreamSynchronize(stream); \
        cudaCheckError(); \
        auto decomp_end = TIME_NOW; \
        if (decomp_res != nvcompSuccess) \
        { \
            std::cerr << "Failed compression!" << std::endl; \
            assert(decomp_res == nvcompSuccess); \
        } \
        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", #ALGO, \
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,  \
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end), \
            (float)total_compressed / TIME_DIFF(decomp_start, decomp_end)); \
        cudaFree(device_statuses); \
        cudaFree(device_actual_uncompressed_bytes); \
        cudaFree(device_decomp_temp); \
        cudaFree(device_temp_ptr); \
        cudaFree(device_compressed_ptrs); \
        cudaCheckError(); \
    }
