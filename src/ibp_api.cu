#include <torch/python.h>
#include "c10/cuda/CUDAStream.h"
#define IBP_DEBUG_PRINT
#include "ibp_helpers.cuh"
#include "preproc/ibp_preproc_host.cuh"
#include "misc/ibp_misc_kernels.cuh"

bool ibp_print_debug = false;

namespace ibp {

void print_debug_msg(bool print)
{
    ibp_print_debug = print;
}

void cuda_deleter(void *ptr)
{
    cudaFree(ptr);
}

/**
 * @brief Preprocess input dataset
 * 
 * @param dataset 2D array of input data to be compressed [num_vecs x vec_size]
 * @return std::tuple<at::Tensor, at::Tensor>
 *         - mask: Returned mask for compression. Size is vec_size.
 *         - bitval: Returned bit values for compression. Size is vec_size.
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
    void *mask = nullptr, *bitval = nullptr;
    auto options = torch::TensorOptions().device(torch::kCUDA);
    // Call preprocessing function with appropriate template arg
    if(data_size == 4) {
        // Use int32 for 4-byte types
        options = options.dtype(torch::kInt32);
        // Get input data
        auto dataset_data = dataset.data_ptr<int32_t>();
        // Preprocess data
        preproc_data<int32_t>(dataset_data, num_vecs, vec_size, 
            (int32_t**)&mask, (int32_t**)&bitval, threshold);
    } else if(data_size == 8) {
        // Use int64 for 8-byte types
        options = options.dtype(torch::kInt64);
        // Get input data
        auto dataset_data = dataset.data_ptr<int64_t>();
        // Preprocess data
        preproc_data<int64_t>(dataset_data, num_vecs, vec_size,
            (int64_t**)&mask, (int64_t**)&bitval, threshold);
    }

    // Return output tensors
    at::Tensor comp_mask = torch::from_blob(mask, {vec_size}, cuda_deleter, options);
    at::Tensor comp_bitval = torch::from_blob(bitval, {vec_size}, cuda_deleter, options);
    return std::make_tuple(comp_mask, comp_bitval);
}

at::Tensor get_compress_size(const at::Tensor &dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const c10::optional<at::Tensor> &index_array_) {
    // Check input tensor
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(dataset.dim() == 1 || dataset.dim() == 2, 
        "Input tensor must be 1D [vec_size] or 2D [num_vecs x vec_size]");

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

    at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream();

    auto options = torch::TensorOptions().device(dataset.device()).dtype(torch::kInt64);
    at::Tensor comp_sizes = torch::empty({num_vecs}, options);
    if(dataset.device().type() != c10::kCUDA)
        comp_sizes = comp_sizes.pin_memory();
    auto comp_sizes_data = comp_sizes.data_ptr<int64_t>();
    // Call function with appropriate template arg
    if(data_size == 4) {
        // Get input data
        auto dataset_data = dataset.data_ptr<int32_t>();
        auto mask_data = mask.data_ptr<int32_t>();
        auto bitval_data = bitval.data_ptr<int32_t>();
        // Call kernel
        check_compress_size_kernel<int32_t><<<num_vecs, 256, 0, stream>>>(dataset_data, 
            num_vecs, vec_size, mask_data, bitval_data, comp_sizes_data, index_array);
    } else if(data_size == 8) {
        // Get input data
        auto dataset_data = dataset.data_ptr<int64_t>();
        auto mask_data = mask.data_ptr<int64_t>();
        auto bitval_data = bitval.data_ptr<int64_t>();
        // Call kernel
        check_compress_size_kernel<int64_t><<<num_vecs, 256, 0, stream>>>(dataset_data, 
            num_vecs, vec_size, mask_data, bitval_data, comp_sizes_data, index_array);
    }
    cudaStreamSynchronize(stream);
    return comp_sizes;
}

void compress_inplace(const at::Tensor &dataset, const at::Tensor &mask, 
    const at::Tensor &bitval)
{
    // Check input tensor
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(dataset.dim() == 1 || dataset.dim() == 2, 
        "Input tensor must be 1D [vec_size] or 2D [num_vecs x vec_size]");

    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4-byte or 8-byte datatype");

    size_t data_size_mask = torch::elementSize(torch::typeMetaToScalarType(mask.dtype()));
    size_t data_size_bitval = torch::elementSize(torch::typeMetaToScalarType(bitval.dtype()));
    TORCH_CHECK(data_size_mask == data_size && data_size_bitval == data_size, 
        "Mask and bitval tensors must have the same datatype size (4-byte or 8-byte) as input dataset");

    size_t num_vecs = dataset.dim() == 1 ? 1 : dataset.size(0);
    size_t vec_size = dataset.dim() == 1 ? dataset.size(0) : dataset.size(1);
    // Call preprocessing function with appropriate template arg
    if(data_size == 4) {
        // Get input data
        auto dataset_data = dataset.data_ptr<int32_t>();
        // TODO: Implement compression
    } else if(data_size == 8) {
        // Get input data
        auto dataset_data = dataset.data_ptr<int64_t>();
        // TODO: Implement compression
    }

    return;
}

} // namespace ibp

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "InvariantBitPacking";
    m.def("print_debug", &ibp::print_debug_msg, "Print debug messages");
    m.def("preprocess", &ibp::preprocess, "Generate mask and bitval for input dataset");
    m.def("get_compress_size", &ibp::get_compress_size, "Get compressed size of input dataset");
    m.def("compress_inplace", &ibp::compress_inplace, "Generate mask and bitval for input dataset");
}