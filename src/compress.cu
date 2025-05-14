#include <torch/python.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "c10/cuda/CUDAStream.h"
#include "ibp_helpers.cuh"
#include "compress/ibp_compress_host.cuh"
#include "misc/ibp_misc_kernels.cuh"

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
    TORCH_CHECK(dataset.is_contiguous(), "Input tensor must be contiguous");

    TORCH_CHECK(mask.device().type() == c10::kCUDA || mask.is_pinned(),
        "Mask tensor must accessible by CUDA device (GPU or pinned memory)");
    TORCH_CHECK(bitval.device().type() == c10::kCUDA || bitval.is_pinned(),
        "Bitval tensor must accessible by CUDA device (GPU or pinned memory)");

    size_t data_size = torch::elementSize(torch::typeMetaToScalarType(dataset.dtype()));
    TORCH_CHECK(data_size == 1 || data_size == 2 || data_size == 4 || data_size == 8,
        "Input tensor must be 1, 2, 4, or 8-byte datatype");

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
    } else if(data_size == 2) {
        if(index_array != nullptr) {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<int16_t><<<num_vecs, 256, 0, stream>>>(
                    (int16_t*)dataset_data, num_vecs, vec_size, (int16_t*)mask_data, (int16_t*)bitval_data,
                    comp_sizes_data, index_array, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<int16_t><<<num_vecs, 256, 0, stream>>>(
                    (int16_t*)dataset_data, num_vecs, vec_size, (int16_t*)mask_data, (int16_t*)bitval_data,
                    comp_sizes_data, index_array);
        }
        else {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<int16_t><<<num_vecs, 256, 0, stream>>>(
                    (int16_t*)dataset_data, num_vecs, vec_size, (int16_t*)mask_data, (int16_t*)bitval_data,
                    comp_sizes_data, (void*)nullptr, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<int16_t><<<num_vecs, 256, 0, stream>>>(
                    (int16_t*)dataset_data, num_vecs, vec_size, (int16_t*)mask_data, (int16_t*)bitval_data,
                    comp_sizes_data);
        }
    } else if(data_size == 1) {
        if(index_array != nullptr) {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<int8_t><<<num_vecs, 256, 0, stream>>>(
                    (int8_t*)dataset_data, num_vecs, vec_size, (int8_t*)mask_data, (int8_t*)bitval_data,
                    comp_sizes_data, index_array, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<int8_t><<<num_vecs, 256, 0, stream>>>(
                    (int8_t*)dataset_data, num_vecs, vec_size, (int8_t*)mask_data, (int8_t*)bitval_data,
                    comp_sizes_data, index_array);
        }
        else {
            if(compress_ctr != nullptr)
                ibp::check_compress_size_kernel<int8_t><<<num_vecs, 256, 0, stream>>>(
                    (int8_t*)dataset_data, num_vecs, vec_size, (int8_t*)mask_data, (int8_t*)bitval_data,
                    comp_sizes_data, (void*)nullptr, (ull*)compress_ctr);
            else
                ibp::check_compress_size_kernel<int8_t><<<num_vecs, 256, 0, stream>>>(
                    (int8_t*)dataset_data, num_vecs, vec_size, (int8_t*)mask_data, (int8_t*)bitval_data,
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
    TORCH_CHECK(dataset.is_contiguous(), "Input tensor must be contiguous");

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
    TORCH_CHECK(dataset.is_contiguous(), "Input tensor must be contiguous");

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