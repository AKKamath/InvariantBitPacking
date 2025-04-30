#include <torch/python.h>
#include "ibp_helpers.cuh"
#include "misc/compress_test.cuh"
#include "preproc/ibp_preproc_host.cuh"
#include "compress/ibp_compress_host.cuh"
#include "decompress/ibp_decompress_host.cuh"
#include "misc/ibp_misc_kernels.cuh"
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
using namespace nvcomp;
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(t1, t2) std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
#define ull unsigned long long

void test_compress(const at::Tensor &dataset)
{
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(),
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(dataset.dim() == 2,
        "Input tensor must be 2D [num_vecs x vec_size]");
    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4, 8-byte datatype");
    int64_t total_nodes = dataset.size(0);
    int64_t feature_len = dataset.size(1);
    int64_t nodes_per_gpu = min(total_nodes, (int64_t)100000);

    void *cpu_features = dataset.data_ptr();
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void *comp_mask = nullptr;
    void *comp_bitval = nullptr;

    size_t compress_len;
    if (data_size == 4) {
        compress_len = ibp::preproc_data((int32_t*)cpu_features, total_nodes,
            feature_len, (int32_t**)&comp_mask, (int32_t**)&comp_bitval);
    } else {
        compress_len = ibp::preproc_data((int64_t*)cpu_features, total_nodes,
            feature_len, (int64_t**)&comp_mask, (int64_t**)&comp_bitval);
    }
    printf("Finished compression preprocessing; compressed len %zu; orig %zu\n", compress_len * data_size, feature_len * data_size);

    long bytes_per_feat = feature_len * data_size;
    long in_bytes = bytes_per_feat * nodes_per_gpu;

    // compute chunk sizes
    size_t* host_uncompressed_bytes;
    const size_t chunk_size = feature_len * data_size;
    const size_t batch_size = nodes_per_gpu;

    char* device_input_data, *device_output_data;
    cudaMalloc(&device_input_data, in_bytes);
    cudaMemcpyAsync(device_input_data, cpu_features, in_bytes, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&device_output_data, in_bytes);
    cudaCheckError();

    cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t)*batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_bytes[i] = bytes_per_feat;
    }

    // Setup an array of pointers to the start of each chunk
    void ** host_uncompressed_ptrs;
    cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t)*batch_size);
    for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        host_uncompressed_ptrs[ix_chunk] = device_input_data + bytes_per_feat;
    }

    size_t* device_uncompressed_bytes;
    void ** device_uncompressed_ptrs;
    cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
    cudaMalloc(&device_uncompressed_ptrs, sizeof(size_t) * batch_size);
    cudaCheckError();

    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

    // Overallocate output bytes
    void **host_compressed_ptrs;
    char *compressed_buffer;
    cudaMallocHost(&host_compressed_ptrs, sizeof(size_t) * batch_size);
    // HOST ALLOC
    if(1)
        cudaMallocHost(&compressed_buffer,
            batch_size * 32 * feature_len * data_size);
    // DEVICE ALLOC
    else
        cudaMalloc(&compressed_buffer,
            batch_size * 32 * feature_len * data_size);

    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        host_compressed_ptrs[ix_chunk] = &compressed_buffer[
            ix_chunk * 32 * feature_len * data_size];
    }
    cudaCheckError();

    size_t *device_compressed_bytes;
    cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);
    cudaCheckError();
    size_t *host_compressed_bytes;
    cudaMallocHost(&host_compressed_bytes, sizeof(size_t) * batch_size);
    cudaCheckError();

    // COMPRESSION ALGOS:
    // ANS, Bitcomp, Cascaded, LZ4, Snappy, GDeflate, Zstd, Deflate
    RUN_COMPRESSION(ANS);
    cudaCheckError();
    RUN_COMPRESSION(Bitcomp);
    cudaCheckError();
    RUN_COMPRESSION(Cascaded);
    cudaCheckError();
    RUN_COMPRESSION(Gdeflate);
    cudaCheckError();
    RUN_COMPRESSION(LZ4);
    cudaCheckError();
    RUN_COMPRESSION(Snappy);
    cudaCheckError();
    RUN_COMPRESSION(Zstd);
    cudaCheckError();

    float *host_output_data;
    cudaMallocHost(&host_output_data, in_bytes);

    auto decomp_start = TIME_NOW;
    auto decomp_end = TIME_NOW;

    // Init for NDZIP
    ndzip::extent ext(1);
    ext[0] = nodes_per_gpu * feature_len;
    ndzip::compressor_requirements req(ext);

    // NDZip compressor/decompressor
    if(data_size == 4) {
        std::unique_ptr<ndzip::cuda_compressor<float>> ndzip_comp = ndzip::make_cuda_compressor<float>(req, stream);
        std::unique_ptr<ndzip::cuda_decompressor<float>> ndzip_decomp = ndzip::make_cuda_decompressor<float>(1, stream);

        ndzip_comp->compress((float*)device_input_data, ext, (uint32_t*)compressed_buffer,
            (uint32_t*)device_compressed_bytes);
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaMemcpy(host_compressed_bytes, device_compressed_bytes, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        *host_compressed_bytes *= sizeof(uint32_t);
        cudaCheckError();
        printf("%s: Uncompressed bytes: %ld, compressed bytes: %u, ratio: %f\n",
            "ndzip", in_bytes, *(uint32_t*)host_compressed_bytes, (float)in_bytes / *host_compressed_bytes);

        cudaMemset(device_output_data, 0, in_bytes);
        // NDZip decompression
        decomp_start = TIME_NOW;
        ndzip_decomp->decompress((uint32_t*)compressed_buffer, (float*)device_output_data, ext);
        cudaDeviceSynchronize();
        cudaCheckError();
        decomp_end = TIME_NOW;
        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "ndzip",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
            (float)*(uint32_t*)host_compressed_bytes / TIME_DIFF(decomp_start, decomp_end));
        cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);
        cudaCheckError();
        for(size_t i = 0; i < in_bytes / sizeof(float); ++i) {
            if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
                printf("Mismatch at %lu: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
                break;
            }
        }
    } else if(data_size == 8) {
        std::unique_ptr<ndzip::cuda_compressor<double>> ndzip_comp = ndzip::make_cuda_compressor<double>(req, stream);
        std::unique_ptr<ndzip::cuda_decompressor<double>> ndzip_decomp = ndzip::make_cuda_decompressor<double>(1, stream);

        ndzip_comp->compress((double*)device_input_data, ext, (uint64_t*)compressed_buffer,
            (uint32_t*)device_compressed_bytes);
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaMemcpy(host_compressed_bytes, device_compressed_bytes, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        *host_compressed_bytes *= sizeof(uint64_t);
        cudaCheckError();
        printf("%s: Uncompressed bytes: %ld, compressed bytes: %u, ratio: %f\n",
            "ndzip", in_bytes, *(uint32_t*)host_compressed_bytes, (float)in_bytes / *host_compressed_bytes);

        cudaMemset(device_output_data, 0, in_bytes);
        // NDZip decompression
        auto decomp_start = TIME_NOW;
        ndzip_decomp->decompress((uint64_t*)compressed_buffer, (double*)device_output_data, ext);
        cudaDeviceSynchronize();
        cudaCheckError();
        auto decomp_end = TIME_NOW;
        printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "ndzip",
            (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
            (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
            (float)*(uint32_t*)host_compressed_bytes / TIME_DIFF(decomp_start, decomp_end));
        cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);
        cudaCheckError();
        for(size_t i = 0; i < in_bytes / sizeof(double); ++i) {
            if(((int64_t*)cpu_features)[i] != ((int64_t*)host_output_data)[i]) {
                printf("Mismatch at %lu: %x vs %x\n", i, ((int64_t*)cpu_features)[i], ((int64_t*)host_output_data)[i]);
                break;
            }
        }
    }

    // Get SM info
    int device, sm_count;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    int32_t *comp_bitmask;
    cudaMalloc(&comp_bitmask, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
    cudaMemset(comp_bitmask, 0, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
    cudaCheckError();
    // Our scheme
    cudaMemset(compressed_buffer, 0, in_bytes);
    ull *d_comp_size, comp_size = 0;
    cudaMalloc(&d_comp_size, sizeof(ull));
    cudaMemset(d_comp_size, 0, sizeof(ull));
    if(data_size == 4) {
        ibp::compress_inplace((int32_t*)compressed_buffer,
            (int32_t*)cpu_features, nodes_per_gpu, (int64_t)feature_len,
            (int32_t*)comp_mask, (int32_t*)comp_bitval, comp_bitmask, (void*)nullptr, (void*)nullptr,
            d_comp_size, stream);
    } else {
        ibp::compress_inplace((ull*)compressed_buffer,
            (ull*)cpu_features, nodes_per_gpu, (int64_t)feature_len,
            (ull*)comp_mask, (ull*)comp_bitval, comp_bitmask, (void*)nullptr, (void*)nullptr,
            d_comp_size, stream);
    }
    cudaMemcpy(&comp_size, d_comp_size, sizeof(ull), cudaMemcpyDeviceToHost);
    printf("%s: Uncompressed bytes: %ld, compressed bytes: %llu, ratio: %f\n",
        "Us", in_bytes, comp_size, (float)in_bytes / comp_size);
    cudaCheckError();

    /*auto kernel = &test_decompressed_features_kernel<true>;
    int shmem_size;
    if(feature_len * sizeof(float) * 2 < maxShmem[0]){
        shmem_size = feature_len * sizeof(float) * 2;
        kernel = &test_decompressed_features_kernel<true>;
    }
    else {
        shmem_size = 0;
        kernel = &test_decompressed_features_kernel<false>;
    }
    cudaMemset(device_output_data, 0, in_bytes);
    // Need opt-in for large shmem allocations
    if (shmem_size >= 48 * 1024) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
        cudaCheckError();
    }
    decomp_start = TIME_NOW;
    kernel<<<sm_count, 256, shmem_size>>>(
        comp_mask, comp_bitval, (int32_t*)compressed_buffer, (int32_t*)device_output_data,
        nodes_per_gpu, feature_len, comp_bitmask, 4);
    cudaDeviceSynchronize();
    cudaCheckError();
    decomp_end = TIME_NOW;
    cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);

    printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "Us",
        (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
        (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
        (float)comp_size / TIME_DIFF(decomp_start, decomp_end));
    cudaCheckError();
    for(int i = 0; i < in_bytes / sizeof(float); ++i) {
        if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
            printf("Mismatch at %d: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
            break;
        }
    }*/

    cudaMemset(device_output_data, 0, in_bytes);
    decomp_start = TIME_NOW;
    if (data_size == 4) {
        ibp::decompress_fetch<int32_t>((int32_t*)device_output_data, (int32_t*)compressed_buffer,
            nodes_per_gpu, (int64_t)feature_len, (int32_t*)comp_mask, (int32_t*)comp_bitval, comp_bitmask,
            (int)(((float)comp_size / (float)in_bytes) * feature_len), stream, sm_count, 512, 0);
    } else {
        ibp::decompress_fetch<ull>((ull*)device_output_data, (ull*)compressed_buffer,
            nodes_per_gpu, (int64_t)feature_len, (ull*)comp_mask, (ull*)comp_bitval, comp_bitmask,
            (int)(((float)comp_size / (float)in_bytes) * feature_len), stream, sm_count, 512, 0);
    }
    cudaDeviceSynchronize();
    cudaCheckError();
    decomp_end = TIME_NOW;
    cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);

    printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "Us-Warp",
        (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
        (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
        (float)comp_size / TIME_DIFF(decomp_start, decomp_end));
    cudaCheckError();
    for(int i = 0; i < in_bytes / sizeof(float); ++i) {
        if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
            printf("Mismatch at %d: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
            break;
        }
    }

    cudaMemset(device_output_data, 0, in_bytes);
    decomp_start = TIME_NOW;
    if(data_size == 4) {
        ibp::decompress_fetch<int32_t>((int32_t*)device_output_data, (int32_t*)compressed_buffer,
            nodes_per_gpu, (int64_t)feature_len, (int32_t*)comp_mask, (int32_t*)comp_bitval, comp_bitmask,
            (int)(((float)comp_size / (float)in_bytes) * feature_len), stream, sm_count, 512, 1);
    } else {
        ibp::decompress_fetch<ull>((ull*)device_output_data, (ull*)compressed_buffer,
            nodes_per_gpu, (int64_t)feature_len, (ull*)comp_mask, (ull*)comp_bitval, comp_bitmask,
            (int)(((float)comp_size / (float)in_bytes) * feature_len), stream, sm_count, 512, 1);
    }
    cudaDeviceSynchronize();
    cudaCheckError();
    decomp_end = TIME_NOW;
    cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);

    printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "Us-TB",
        (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
        (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
        (float)comp_size / TIME_DIFF(decomp_start, decomp_end));
    cudaCheckError();
    for(int i = 0; i < in_bytes / sizeof(float); ++i) {
        if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
            printf("Mismatch at %d: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
            break;
        }
    }

    cudaMemset(device_output_data, 0, in_bytes);
    decomp_start = TIME_NOW;
    if (data_size == 4) {
        ibp::decompress_fetch<int32_t>((int32_t*)device_output_data, (int32_t*)compressed_buffer,
            nodes_per_gpu, (int64_t)feature_len, (int32_t*)comp_mask, (int32_t*)comp_bitval, comp_bitmask,
            (int)(((float)comp_size / (float)in_bytes) * feature_len), stream, sm_count, 512, 5);
    } else {
        ibp::decompress_fetch<ull>((ull*)device_output_data, (ull*)compressed_buffer,
            nodes_per_gpu, (int64_t)feature_len, (ull*)comp_mask, (ull*)comp_bitval, comp_bitmask,
            (int)(((float)comp_size / (float)in_bytes) * feature_len), stream, sm_count, 512, 5);
    }
    cudaDeviceSynchronize();
    cudaCheckError();
    decomp_end = TIME_NOW;
    cudaMemcpy(host_output_data, device_output_data, in_bytes, cudaMemcpyDeviceToHost);

    printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s; True thput: %f MB/s\n", "Us-Async",
        (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
        (float)in_bytes / TIME_DIFF(decomp_start, decomp_end),
        (float)comp_size / TIME_DIFF(decomp_start, decomp_end));
    cudaCheckError();
    for(int i = 0; i < in_bytes / sizeof(float); ++i) {
        if(((float*)cpu_features)[i] != ((float*)host_output_data)[i]) {
            printf("Mismatch at %d: %x vs %x\n", i, ((int32_t*)cpu_features)[i], ((int32_t*)host_output_data)[i]);
            break;
        }
    }

    // Flat data copy
    cudaMemset(comp_bitmask, 0, (nodes_per_gpu + 31) / 32 * sizeof(int32_t));
    decomp_start = TIME_NOW;
    if(data_size == 4) {
        flat_copy_kernel<<<sm_count, 512>>>((int32_t*)compressed_buffer,
            (int32_t*)device_output_data, nodes_per_gpu, feature_len);
    } else {
        flat_copy_kernel<<<sm_count, 512>>>((int64_t*)compressed_buffer,
            (int64_t*)device_output_data, nodes_per_gpu, feature_len);
    }
    cudaDeviceSynchronize();
    decomp_end = TIME_NOW;
    printf("%s: Time taken to decompress: %f ms. Throughput: %f MB/s\n", "Transfer",
        (float)TIME_DIFF(decomp_start, decomp_end) / 1000.0,
        (float)in_bytes / TIME_DIFF(decomp_start, decomp_end));
    cudaCheckError();

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "InvariantBitPackingTest";
    m.def("test_compress", &test_compress, "Test different compression methods");
}