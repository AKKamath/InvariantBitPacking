#include <torch/python.h>
#include "preproc/ibp_preproc_host.cuh"
#include "ibp_helpers.cuh"

namespace ibp {

void print_debug_msg(bool print)
{
    print_debug = print;
}

/**
 * @brief Preprocess input dataset
 * 
 * @param dataset 2D array of input data to be compressed [num_elems x elem_size]
 * @return std::tuple<at::Tensor, at::Tensor>
 *         - mask: Returned mask for compression. Size is elem_size.
 *         - bitval: Returned bit values for compression. Size is elem_size.
 */
std::tuple<at::Tensor, at::Tensor> preprocess(const at::Tensor &dataset)
{
    // Check input tensor
    TORCH_CHECK(dataset.device().type() == c10::kCUDA || dataset.is_pinned(), 
        "Input tensor must accessible by CUDA device");
    TORCH_CHECK(dataset.dim() == 2, "Input tensor must be 2D [num_elems x elem_size], got", dataset.dim(), "D");
    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 4 || data_size == 8, "Input tensor must be 4-byte or 8-byte sizes");

    auto num_elems = dataset.size(0);
    auto elem_size = dataset.size(1);
    void *mask = nullptr, *bitval = nullptr;
    auto options = torch::TensorOptions().device(torch::kCUDA);
    // Call preprocessing function with appropriate template arg
    if(data_size == 4) {
        // Use int32 for 4-byte types
        options = options.dtype(torch::kInt32);
        // Get input data
        auto dataset_data = dataset.data_ptr<int32_t>();
        // Preprocess data
        preproc_data<int32_t>(dataset_data, num_elems, elem_size, 
            (int32_t**)&mask, (int32_t**)&bitval);
        int32_t *h_mask;
        cudaMallocHost(&h_mask, elem_size * sizeof(int32_t));
        cudaMemcpy(h_mask, mask, elem_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for(int i = 0; i < elem_size; i++)
            DPRINTF("%d ", h_mask[i]);
    } else if(data_size == 8) {
        // Use int64 for 8-byte types
        options = options.dtype(torch::kInt64);
        // Get input data
        auto dataset_data = dataset.data_ptr<int64_t>();
        // Preprocess data
        preproc_data<int64_t>(dataset_data, num_elems, elem_size,
            (int64_t**)&mask, (int64_t**)&bitval);
        int64_t *h_mask;
        cudaMallocHost(&h_mask, elem_size * sizeof(int64_t));
        cudaMemcpy(h_mask, mask, elem_size * sizeof(int64_t), cudaMemcpyDeviceToHost);
        for(int i = 0; i < elem_size; i++)
            DPRINTF("%ld ", h_mask[i]);
    }

    // Return output tensors
    at::Tensor comp_mask = torch::from_blob(mask, {elem_size}, options);
    at::Tensor comp_bitval = torch::from_blob(bitval, {elem_size}, options);
    return std::make_tuple(comp_mask, comp_bitval);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "InvariantBitPacking";
    m.def("print_debug", &ibp::print_debug_msg, "Print debug messages");
    m.def("preprocess", &ibp::preprocess, "Generate mask and bitval for input dataset");
}