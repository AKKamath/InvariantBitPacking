#include <torch/python.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "c10/cuda/CUDAStream.h"
#define IBP_DEBUG_PRINT
#include "ibp_helpers.cuh"
#include "preproc/ibp_preproc_host.cuh"
#include "misc/ibp_misc_kernels.cuh"
#include "compress/ibp_compress_host.cuh"
#include "decompress/ibp_decompress_host.cuh"

bool ibp_print_debug = false;

void print_debug_msg(bool print)
{
    ibp_print_debug = print;
}

void cuda_deleter(void *ptr)
{
    cudaFree(ptr);
}

/**
 * @brief Preprocess input dataset. Non-blocking call
 * 
 * @param dataset 2D array of input data to be compressed. [num_vecs x vec_size]
 *                Must be GPU-accessible (pinned or GPU memory)
 * @return std::tuple<at::Tensor, at::Tensor>
 *         - mask: [GPU memory] Returned mask for compression. Size is vec_size.
 *         - bitval: [GPU memory] Returned bit values for compression. Size is vec_size.
 */
std::tuple<at::Tensor, at::Tensor> preprocess(const at::Tensor &dataset, 
    c10::optional<float> threshold_)
{
    // Check input tensor
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(dataset.dim() == 2, "Input tensor must be 2D [num_vecs x vec_size]");
    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4-byte or 8-byte datatype");

    float threshold = -1.0;
    if (threshold_.has_value())
        threshold = threshold_.value();

    auto num_vecs = dataset.size(0);
    auto vec_size = dataset.size(1);
    auto options = torch::TensorOptions().device(torch::kCUDA);
    if(data_size == 4)
        options = options.dtype(torch::kInt32);
    else if(data_size == 8)
        options = options.dtype(torch::kInt64);

    // Prepare output tensors
    at::Tensor mask = torch::zeros({vec_size}, options);
    at::Tensor bitval = torch::zeros({vec_size}, options);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Get input parameters
    void *dataset_data = dataset.data_ptr();
    void *mask_data = mask.data_ptr();
    void *bitval_data = bitval.data_ptr();
    // Call preprocessing function with appropriate template arg
    if(data_size == 4) {
        // Use int32 for 4-byte types
        ibp::preproc_data<int32_t>((int32_t*)dataset_data, num_vecs, vec_size, 
            (int32_t**)&mask_data, (int32_t**)&bitval_data, threshold, stream);
    } else if(data_size == 8) {
        // Use ull for 8-byte types
        ibp::preproc_data<ull>((ull*)dataset_data, num_vecs, vec_size,
            (ull**)&mask_data, (ull**)&bitval_data, threshold, stream);
    }

    // Return output tensors
    return std::make_tuple(mask, bitval);
}

/**
 * @brief Computes the compressed size of a dataset. Non-blocking call.
 *
 * This function calculates the size of each element of the compressed dataset.
 * Optionally, an index array can be provided to specify which vectors to consider.
 * Non-blocking call. Synchronize CUDA before accessing output tensor.
 *
 * @param dataset The input tensor representing the dataset to be compressed. 
 *                [vec_size] or [num_vecs x vec_size]
 * @param mask The preprocessed mask used for compression.
 * @param bitval The preprocessed bitval used for compression.
 * @param index_array_ Optional tensor specifying the indices of the vectors to be considered.
 * @return A tensor representing the compressed size of each element of the dataset.
 */

at::Tensor get_compress_size(const at::Tensor &dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const c10::optional<at::Tensor> &index_array_,
    const c10::optional<at::Tensor> &compress_ctr_) {
    // Check input tensor
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(dataset.dim() == 1 || dataset.dim() == 2, 
        "Input tensor must be 1D [vec_size] or 2D [num_vecs x vec_size]");

    TORCH_CHECK(mask.device().type() == c10::kCUDA || mask.is_pinned(), 
        "Mask tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(bitval.device().type() == c10::kCUDA || bitval.is_pinned(), 
        "Bitval tensor must accessible by CUDA device (GPU or pinned memory)");

    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4-byte or 8-byte datatype");

    size_t data_size_mask = torch::elementSize(torch::typeMetaToScalarType(mask.dtype()));
    size_t data_size_bitval = torch::elementSize(torch::typeMetaToScalarType(bitval.dtype()));
    TORCH_CHECK(data_size_mask == data_size && data_size_bitval == data_size, 
        "Mask and bitval tensors must have the same datatype size (4-byte or 8-byte) as input dataset");

    long num_vecs = dataset.dim() == 1 ? 1 : dataset.size(0);
    long vec_size = dataset.dim() == 1 ? dataset.size(0) : dataset.size(1);

    // Get index array if provided
    int64_t *index_array = nullptr;
    if (index_array_.has_value()) {
        TORCH_CHECK(index_array_.value().device().type() == c10::kCUDA || 
            index_array_.value().is_pinned(), 
            "Index array must accessible by CUDA device (GPU or pinned memory)");
        TORCH_CHECK(index_array_.value().dim() == 1, "Index array should be 1D");
        index_array = index_array_.value().to(torch::kInt64).data_ptr<int64_t>();
        num_vecs = index_array_.value().size(0);
    }

    // Get compress size ctr if provided
    int64_t *compress_ctr = nullptr;
    if (compress_ctr_.has_value()) {
        TORCH_CHECK(compress_ctr_.value().device().type() == c10::kCUDA || 
            compress_ctr_.value().is_pinned(), 
            "Compress ctr must accessible by CUDA device (GPU or pinned memory)");
        TORCH_CHECK(compress_ctr_.value().dim() == 1 && compress_ctr_.value().size(0) == 1, 
            "Compress ctr should be a single element");
        compress_ctr = compress_ctr_.value().to(torch::kInt64).data_ptr<int64_t>();
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    auto options = torch::TensorOptions().device(dataset.device()).dtype(torch::kInt64);
    at::Tensor comp_sizes = torch::empty({num_vecs}, options);
    if(dataset.device().type() != c10::kCUDA)
        comp_sizes = comp_sizes.pin_memory();
    auto comp_sizes_data = comp_sizes.data_ptr<int64_t>();

    // Get input parameters
    void *dataset_data = dataset.data_ptr();
    void *mask_data = mask.data_ptr();
    void *bitval_data = bitval.data_ptr();
    // Call kernel; separate calls for optimized implementation if inputs unused
    if(data_size == 4) {
        if(index_array != nullptr) {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<int32_t><<<num_vecs, 256, 0, stream>>>(
                    (int*)dataset_data, num_vecs, vec_size, (int*)mask_data, (int*)bitval_data, 
                    comp_sizes_data, index_array, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<int32_t><<<num_vecs, 256, 0, stream>>>(
                    (int*)dataset_data, num_vecs, vec_size, (int*)mask_data, (int*)bitval_data, 
                    comp_sizes_data, index_array);
        }
        else {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<int32_t><<<num_vecs, 256, 0, stream>>>(
                    (int*)dataset_data, num_vecs, vec_size, (int*)mask_data, (int*)bitval_data, 
                    comp_sizes_data, (void*)nullptr, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<int32_t><<<num_vecs, 256, 0, stream>>>(
                    (int*)dataset_data, num_vecs, vec_size, (int*)mask_data, (int*)bitval_data, 
                    comp_sizes_data);

        }
    } else if(data_size == 8) {
        if(index_array != nullptr) {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<ull><<<num_vecs, 256, 0, stream>>>(
                    (ull*)dataset_data, num_vecs, vec_size, (ull*)mask_data, (ull*)bitval_data, 
                    comp_sizes_data, index_array, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<ull><<<num_vecs, 256, 0, stream>>>(
                    (ull*)dataset_data, num_vecs, vec_size, (ull*)mask_data, (ull*)bitval_data, 
                    comp_sizes_data, index_array);
        }
        else {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<ull><<<num_vecs, 256, 0, stream>>>(
                    (ull*)dataset_data, num_vecs, vec_size, (ull*)mask_data, (ull*)bitval_data, 
                    comp_sizes_data, (void*)nullptr, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<ull><<<num_vecs, 256, 0, stream>>>(
                    (ull*)dataset_data, num_vecs, vec_size, (ull*)mask_data, (ull*)bitval_data, 
                    comp_sizes_data);
        }
    }
    return comp_sizes;
}

at::Tensor compress_inplace(const at::Tensor &dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const c10::optional<at::Tensor> &index_array_)
{
    // Check input tensor
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(dataset.dim() == 1 || dataset.dim() == 2, 
        "Input tensor must be 1D [vec_size] or 2D [num_vecs x vec_size]");

    TORCH_CHECK(mask.device().type() == c10::kCUDA || mask.is_pinned(), 
        "Mask tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(bitval.device().type() == c10::kCUDA || bitval.is_pinned(), 
        "Bitval tensor must accessible by CUDA device (GPU or pinned memory)");

    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4-byte or 8-byte datatype");

    size_t data_size_mask = torch::elementSize(torch::typeMetaToScalarType(mask.dtype()));
    size_t data_size_bitval = torch::elementSize(torch::typeMetaToScalarType(bitval.dtype()));
    TORCH_CHECK(data_size_mask == data_size && data_size_bitval == data_size, 
        "Mask and bitval tensors must have the same datatype size (4-byte or 8-byte) as input dataset");

    size_t num_vecs = dataset.dim() == 1 ? 1 : dataset.size(0);
    size_t vec_size = dataset.dim() == 1 ? dataset.size(0) : dataset.size(1);

    // Get index array if provided
    int64_t *index_array = nullptr;
    if (index_array_.has_value()) {
        TORCH_CHECK(index_array_.value().device().type() == c10::kCUDA || 
            index_array_.value().is_pinned(), 
            "Index array must accessible by CUDA device (GPU or pinned memory)");
        TORCH_CHECK(index_array_.value().dim() == 1, "Index array should be 1D");
        index_array = index_array_.value().to(torch::kInt64).data_ptr<int64_t>();
        num_vecs = index_array_.value().size(0);
    }

    // Now generate output bitmask tensor
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    at::Tensor bitmask = torch::zeros({(long)((num_vecs + 31) / 32)}, options);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Get input parameters
    void *dataset_data = dataset.data_ptr();
    void *mask_data = mask.data_ptr();
    void *bitval_data = bitval.data_ptr();
    int32_t *bitmask_data = bitmask.data_ptr<int32_t>();
    // Perform in-place compression with appropriate template args
    if(data_size == 4) {
        ibp::compress_inplace((int*)dataset_data, (int*)dataset_data, num_vecs, 
            vec_size, (int*)mask_data, (int*)bitval_data, bitmask_data, index_array, 
            (void*)nullptr, (void*)nullptr, stream);
    } else if(data_size == 8) {
        ibp::compress_inplace((ull*)dataset_data, (ull*)dataset_data, num_vecs, 
        vec_size, (ull*)mask_data, (ull*)bitval_data, bitmask_data, index_array, 
            (void*)nullptr, (void*)nullptr, stream);
    }

    return bitmask;
}

std::tuple<at::Tensor, at::Tensor> compress_condensed(const at::Tensor &dataset, 
    const at::Tensor &mask, const at::Tensor &bitval, 
    const c10::optional<at::Tensor> &index_array_)
{
    // Check input tensor
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(dataset.dim() == 1 || dataset.dim() == 2, 
        "Input tensor must be 1D [vec_size] or 2D [num_vecs x vec_size]");

    TORCH_CHECK(mask.device().type() == c10::kCUDA || mask.is_pinned(), 
        "Mask tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(bitval.device().type() == c10::kCUDA || bitval.is_pinned(), 
        "Bitval tensor must accessible by CUDA device (GPU or pinned memory)");

    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4-byte or 8-byte datatype");

    size_t data_size_mask = torch::elementSize(torch::typeMetaToScalarType(mask.dtype()));
    size_t data_size_bitval = torch::elementSize(torch::typeMetaToScalarType(bitval.dtype()));
    TORCH_CHECK(data_size_mask == data_size && data_size_bitval == data_size, 
        "Mask and bitval tensors must have the same datatype size (4-byte or 8-byte) as input dataset");

    size_t num_vecs = dataset.dim() == 1 ? 1 : dataset.size(0);
    size_t vec_size = dataset.dim() == 1 ? dataset.size(0) : dataset.size(1);

    // Get index array if provided
    int64_t *index_array = nullptr;
    if (index_array_.has_value()) {
        TORCH_CHECK(index_array_.value().device().type() == c10::kCUDA || 
            index_array_.value().is_pinned(), 
            "Index array must accessible by CUDA device (GPU or pinned memory)");
        TORCH_CHECK(index_array_.value().dim() == 1, "Index array should be 1D");
        index_array = index_array_.value().to(torch::kInt64).data_ptr<int64_t>();
        num_vecs = index_array_.value().size(0);
    }

    auto options = torch::TensorOptions().device(c10::kCPU).dtype(torch::kInt64);
    at::Tensor comp_size_total = torch::zeros({1}, options).pin_memory();

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // Get compressed sizes
    at::Tensor byte_offsets = get_compress_size(dataset, mask, bitval, index_array_, comp_size_total);
    cudaStreamSynchronize(stream);
    // Scan to convert sizes to offsets
    thrust::inclusive_scan(thrust::device, byte_offsets.data_ptr<int64_t>(), 
        byte_offsets.data_ptr<int64_t>() + num_vecs, byte_offsets.data_ptr<int64_t>());
    DPRINTF("Compressed size: %lld\n", comp_size_total.item<int64_t>());
    
    // Now generate output compressed tensor
    options = torch::TensorOptions().device(dataset.device()).dtype(dataset.dtype());
    at::Tensor comp_dataset = torch::zeros({comp_size_total.item<int64_t>() / (long)data_size}, options);
    if(dataset.device().type() != c10::kCUDA)
        comp_dataset = comp_dataset.pin_memory();

    // Get input parameters
    void *comp_dataset_data = comp_dataset.data_ptr();
    void *dataset_data = dataset.data_ptr();
    void *mask_data = mask.data_ptr();
    void *bitval_data = bitval.data_ptr();
    auto offsets = byte_offsets.data_ptr<int64_t>();

    // Perform out-of-place compression with appropriate template args
    if(data_size == 4) {
        ibp::compress_condensed((int*)comp_dataset_data, (int*)dataset_data, 
            num_vecs, vec_size, (int*)mask_data, (int*)bitval_data, offsets, 
            nullptr, index_array, stream);
    } else if(data_size == 8) {
        ibp::compress_condensed((ull*)comp_dataset_data, (ull*)dataset_data, 
            num_vecs, vec_size, (ull*)mask_data, (ull*)bitval_data, offsets, 
            nullptr, index_array, stream);
    }

    return std::make_tuple(comp_dataset, byte_offsets);
}

at::Tensor decompress_fetch(const at::Tensor &comp_dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const at::Tensor &bitmask, const torch::Device out_device, 
    const c10::optional<int> &comp_len_, const c10::optional<at::Tensor> &index_array_)
{
    // Check input tensor
    TORCH_CHECK(comp_dataset.device().type() == c10::kCUDA || comp_dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(comp_dataset.dim() == 1 || comp_dataset.dim() == 2, 
        "Input tensor must be 1D [vec_size] or 2D [num_vecs x vec_size] got ", comp_dataset.dim());

    TORCH_CHECK(mask.device().type() == c10::kCUDA || mask.is_pinned(), 
        "Mask tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(bitval.device().type() == c10::kCUDA || bitval.is_pinned(), 
        "Bitval tensor must accessible by CUDA device (GPU or pinned memory)");

    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(comp_dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4-byte or 8-byte datatype");

    size_t data_size_mask = torch::elementSize(torch::typeMetaToScalarType(mask.dtype()));
    size_t data_size_bitval = torch::elementSize(torch::typeMetaToScalarType(bitval.dtype()));
    TORCH_CHECK(data_size_mask == data_size && data_size_bitval == data_size, 
        "Mask and bitval tensors must have the same datatype size (4-byte or 8-byte) as input dataset");

    size_t num_vecs = comp_dataset.dim() == 1 ? 1 : comp_dataset.size(0);
    size_t vec_size = comp_dataset.dim() == 1 ? comp_dataset.size(0) : comp_dataset.size(1);

    // Get index array if provided
    int64_t *index_array = nullptr;
    if (index_array_.has_value()) {
        TORCH_CHECK(index_array_.value().device().type() == c10::kCUDA || 
            index_array_.value().is_pinned(), 
            "Index array must accessible by CUDA device (GPU or pinned memory)");
        TORCH_CHECK(index_array_.value().dim() == 1, "Index array should be 1D");
        index_array = index_array_.value().to(torch::kInt64).data_ptr<int64_t>();
        num_vecs = index_array_.value().size(0);
    }

    int comp_len = vec_size * data_size;
    if (comp_len_.has_value() && comp_len_.value() > 0)
        comp_len = comp_len_.value();
    
    // Now generate output decompressed tensor
    auto options = torch::TensorOptions().device(out_device).dtype(comp_dataset.dtype());
    at::Tensor decomp_dataset = torch::zeros({(long)num_vecs, (long)vec_size}, options);
    if(out_device.type() != c10::kCUDA)
        decomp_dataset = decomp_dataset.pin_memory();
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // Get input parameters
    void *decomp_data = decomp_dataset.data_ptr();
    void *comp_data = comp_dataset.data_ptr();
    void *mask_data = mask.data_ptr();
    void *bitval_data = bitval.data_ptr();
    int32_t *bitmask_data = bitmask.data_ptr<int32_t>();
    // Perform out-of-place compression with appropriate template args
    if(data_size == 4) {
        ibp::decompress_fetch((int*)decomp_data, (int*)comp_data, num_vecs, vec_size, 
        (int*)mask_data, (int*)bitval_data, bitmask_data, comp_len, index_array, stream);
    } else if(data_size == 8) {
        ibp::decompress_fetch((ull*)decomp_data, (ull*)comp_data, num_vecs, vec_size, 
        (ull*)mask_data, (ull*)bitval_data, bitmask_data, comp_len, index_array, stream);
    }
    return decomp_dataset;
}

#include <ATen/Functions.h>
#include <ATen/MapAllocator.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/TensorOptions.h>
at::Tensor read_shared(const char* filename, std::vector<int64_t> &shape, 
    const py::object &dtype)
{
    torch::ScalarType type = torch::python::detail::py_object_to_dtype(dtype);
    struct stat sb;
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file for reading");
        exit(EXIT_FAILURE);
    }
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        exit(EXIT_FAILURE);
    }

    // Create tensor
    at::Tensor tensor = torch::empty(shape, torch::TensorOptions().dtype(type).device(torch::kCPU));

    // Create new shared memory storage
    const at::Storage& origStorage = tensor.storage();

    size_t tensor_size = origStorage.nbytes();
    
    // Copied from pytorch/aten/src/ATen/StorageUtils.cpp
    int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE |
        at::ALLOCATOR_MAPPED_KEEPFD | at::ALLOCATOR_MAPPED_UNLINK;
    std::string handle = at::NewProcessWideShmHandle();
    auto sptr = at::MapAllocator::makeDataPtr(
        handle.c_str(), flags, tensor_size * sizeof(uint8_t), nullptr);
    at::Storage newStorage(c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        tensor_size,
        std::move(sptr),
        /*allocator=*/nullptr,
        /*resizable=*/false));
    
    // Replace the old data_ptr and allocator with the new ones
    c10::StorageImpl* origStorageImpl = origStorage.unsafeGetStorageImpl();
    c10::StorageImpl* newStorageImpl = newStorage.unsafeGetStorageImpl();
    origStorageImpl->set_data_ptr(std::move(newStorageImpl->data_ptr()));
    origStorageImpl->set_allocator(newStorageImpl->allocator());

    // Copy from file into tensor
    // If > 20GB, copy in chunks to avoid OOM
    const off_t FIXED_CHUNK = 20ll * 1024ll * 1024ll * 1024ll;
    size_t chunk_size = std::min(sb.st_size, FIXED_CHUNK);
    size_t total_size = sb.st_size, offset = 0;
    char *data_ptr = (char*)tensor.data_ptr();
    while(total_size > 0) {
        void *addr = mmap(NULL, chunk_size, PROT_READ, MAP_PRIVATE, fd, offset);
        if (addr == MAP_FAILED) {
            perror("Error mapping file to memory");
            exit(EXIT_FAILURE);
        }

        // Copy from mapped memory to tensor
        memcpy(&data_ptr[offset], addr, chunk_size);

        int err = munmap(addr, chunk_size);
        if (err == -1) {
            perror("Error unmapping memory");
            exit(EXIT_FAILURE);
        }
        offset += chunk_size;
        total_size -= chunk_size;
        chunk_size = min(total_size, chunk_size);
    }

    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "InvariantBitPacking";
    m.def("print_debug", &print_debug_msg, "Print debug messages");
    m.def("preprocess", &preprocess, "Generate mask and bitval for input dataset");
    m.def("get_compress_size", &get_compress_size, "Get compressed size of input dataset");
    m.def("compress_inplace", &compress_inplace, "Convert input dataset into compressed form (occupies same space)");
    m.def("compress_condensed", &compress_condensed, "Return compressed form of input and byte offsets");
    m.def("decompress_fetch", &decompress_fetch, "Decompress dataset into new tensor");
    m.def("read_shared", &read_shared, "Read tensor from file into shared memory (avoids excessive mem usage)");
    //m.def("decompress", &decompress, "Decompressed form of input dataset and byte offsets");
}