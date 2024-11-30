#include <torch/python.h>

#include "c10/cuda/CUDAStream.h"
#define IBP_DEBUG_PRINT
#include "ibp_helpers.cuh"
#include "preproc/ibp_preproc_host.cuh"

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
    TORCH_CHECK(data_size == 1 || data_size == 2 || data_size == 4 || data_size == 8, 
        "Input tensor must be 1, 2, 4, or 8-byte datatype");

    float threshold = -1.0;
    if (threshold_.has_value())
        threshold = threshold_.value();

    auto num_vecs = dataset.size(0);
    auto vec_size = dataset.size(1);
    auto options = torch::TensorOptions().device(torch::kCUDA);
    if(data_size == 1)
        options = options.dtype(torch::kInt8);
    else if(data_size == 2)
        options = options.dtype(torch::kInt16);
    else if(data_size == 4)
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
    if(data_size == 1) {
        // Use int8_t for 1-byte types
        ibp::preproc_data<int8_t>((int8_t*)dataset_data, num_vecs, vec_size,
            (int8_t**)&mask_data, (int8_t**)&bitval_data, threshold, stream);
    } else if(data_size == 2) {
        // Use int16_t for 2-byte types
        ibp::preproc_data<int16_t>((int16_t*)dataset_data, num_vecs, vec_size,
            (int16_t**)&mask_data, (int16_t**)&bitval_data, threshold, stream);
    } else if(data_size == 4) {
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
 * @brief Preprocess input dataset. Non-blocking call
 * 
 * @param dataset 2D array of input data to be compressed. [num_vecs x vec_size]
 *                Must be GPU-accessible (pinned or GPU memory)
 * @return std::tuple<at::Tensor, at::Tensor>
 *         - mask: [GPU memory] Returned mask for compression. Size is vec_size.
 *         - bitval: [GPU memory] Returned bit values for compression. Size is vec_size.
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> preprocess_kmeans(const at::Tensor &dataset, 
    int num_clusters, c10::optional<float> threshold_)
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
    at::Tensor mask = torch::zeros({num_clusters, vec_size}, options);
    at::Tensor bitval = torch::zeros({num_clusters, vec_size}, options);

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Now generate output cluster IDs tensor
    auto options_cluster = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
    at::Tensor clusters = torch::zeros({(long)num_vecs}, options_cluster).pin_memory();

    // Get input parameters
    void *dataset_data = dataset.data_ptr();
    void *mask_data = mask.data_ptr();
    void *bitval_data = bitval.data_ptr();
    int32_t *clusters_data = clusters.data_ptr<int32_t>();
    // Call preprocessing function with appropriate template arg
    if(data_size == 4) {
        // Use int32 for 4-byte types
        ibp::preproc_kmeans<int32_t>((int32_t*)dataset_data, num_vecs, vec_size, 
            (int32_t**)&mask_data, (int32_t**)&bitval_data, num_clusters, &clusters_data, threshold);
    } else if(data_size == 8) {
        // Use ull for 8-byte types
        ibp::preproc_kmeans<ull>((ull*)dataset_data, num_vecs, vec_size,
            (ull**)&mask_data, (ull**)&bitval_data, num_clusters, &clusters_data, threshold);
    }

    // Return output tensors
    return std::make_tuple(mask, bitval, clusters);
}