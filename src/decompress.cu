#include <torch/python.h>

#include "c10/cuda/CUDAStream.h"
#include <ATen/cuda/CUDAContext.h>
#define IBP_DEBUG_PRINT
#include "ibp_helpers.cuh"
#include "decompress/ibp_decompress_host.cuh"
#include "misc/ibp_misc_kernels.cuh"

at::Tensor decompress_fetch(const at::Tensor &comp_dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const at::Tensor &bitmask, const torch::Device out_device, 
    const c10::optional<int> &comp_len_, const c10::optional<at::Tensor> &index_array_,
    const c10::optional<int> nblks_, const c10::optional<int> nthreads_, c10::optional<int> impl_)
{
    // Tested for V100, A100. Adjust as needed for your GPU
    auto dprops = at::cuda::getCurrentDeviceProperties();
    // Default to all SMs
    int NBLKS = dprops->multiProcessorCount * 2;

    if(nblks_.has_value())
        NBLKS = nblks_.value();

    int NTHREADS = 512;
    if(nthreads_.has_value())
        NTHREADS = nthreads_.value();

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

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Get index array if provided
    int64_t *index_array = nullptr, *offset_array = nullptr;
    // Needed for dynamic memory allocation to be correct
    at::Tensor index_tensor, offset_tensor;
    if (index_array_.has_value()) {
        TORCH_CHECK(index_array_.value().device().type() == c10::kCUDA, 
            "Index array must accessible by CUDA device (GPU or pinned memory)");
        TORCH_CHECK(index_array_.value().dim() == 1, "Index array should be 1D");
        index_array = index_array_.value().to(torch::kInt64).data_ptr<int64_t>();
        num_vecs = index_array_.value().size(0);

        // Sort index array to improve memory access locality
        int max_num_elems = comp_dataset.dim() == 1 ? 1 : comp_dataset.size(0);
        int64_t *d_keys_in = index_array, *d_values_in;
        int64_t *d_keys_out, *d_values_out;
        int end_bit = int(log2(max_num_elems)) + 1;

        at::Tensor values_tensor = torch::empty({(long)num_vecs}, torch::TensorOptions().device(c10::kCUDA).dtype(torch::kInt64));
        d_values_in = values_tensor.data_ptr<int64_t>();
        ibp::range_kernel<<<16, 1024, 0, stream>>>(d_values_in, 0, num_vecs);
        cudaCheckError();

        index_tensor = torch::empty({(long)num_vecs}, torch::TensorOptions().device(c10::kCUDA).dtype(torch::kInt64));
        offset_tensor = torch::empty({(long)num_vecs}, torch::TensorOptions().device(c10::kCUDA).dtype(torch::kInt64));

        d_keys_out = index_tensor.data_ptr<int64_t>();
        d_values_out = offset_tensor.data_ptr<int64_t>();
        cudaCheckError();

        void     *d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_vecs, 
            0, end_bit, stream);
        cudaCheckError();

        // Allocate temporary storage
        at::Tensor temp_tensor = torch::empty({(long)temp_storage_bytes}, torch::TensorOptions().device(c10::kCUDA).dtype(torch::kInt8));
        d_temp_storage = (void*)temp_tensor.data_ptr<int8_t>();

        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_vecs,
            0, end_bit, stream);
        index_array = d_keys_out;
        offset_array = d_values_out;
    }

    // Default: warp-parallel
    int impl = 0;
    if(impl_.has_value())
        impl = impl_.value();

    int comp_len = vec_size;
    if (comp_len_.has_value() && comp_len_.value() > 0)
        comp_len = comp_len_.value();
    
    // Now generate output decompressed tensor
    auto options = torch::TensorOptions().device(out_device).dtype(comp_dataset.dtype());
    at::Tensor decomp_dataset = torch::zeros({(long)num_vecs, (long)vec_size}, options);
    if(out_device.type() != c10::kCUDA)
        decomp_dataset = decomp_dataset.pin_memory();
    
    // Get input parameters
    void *decomp_data = decomp_dataset.data_ptr();
    void *comp_data = comp_dataset.data_ptr();
    void *mask_data = mask.data_ptr();
    void *bitval_data = bitval.data_ptr();
    int32_t *bitmask_data = bitmask.data_ptr<int32_t>();
    // Perform out-of-place compression with appropriate template args
    if(data_size == 4) {
        if(index_array == nullptr)
            ibp::decompress_fetch((int*)decomp_data, (int*)comp_data, num_vecs, vec_size, 
                (int*)mask_data, (int*)bitval_data, bitmask_data, comp_len, 
                stream, NBLKS, NTHREADS, impl);
        else
            ibp::decompress_fetch((int*)decomp_data, (int*)comp_data, num_vecs, vec_size, 
                (int*)mask_data, (int*)bitval_data, bitmask_data, comp_len,
                stream, NBLKS, NTHREADS, impl, index_array, offset_array);
    } else if(data_size == 8) {
        if(index_array == nullptr)
            ibp::decompress_fetch((ull*)decomp_data, (ull*)comp_data, num_vecs, vec_size, 
                (ull*)mask_data, (ull*)bitval_data, bitmask_data, comp_len,
                stream, NBLKS, NTHREADS, impl);
        else
            ibp::decompress_fetch((ull*)decomp_data, (ull*)comp_data, num_vecs, vec_size, 
                (ull*)mask_data, (ull*)bitval_data, bitmask_data, comp_len,
                stream, NBLKS, NTHREADS, impl, index_array, offset_array);
    }
    return decomp_dataset;
}