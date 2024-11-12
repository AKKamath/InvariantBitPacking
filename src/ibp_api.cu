#include <torch/python.h>

#include "c10/cuda/CUDAStream.h"
#define IBP_DEBUG_PRINT
#include "ibp_helpers.cuh"

bool ibp_print_debug = false;

void print_debug_msg(bool print)
{
    ibp_print_debug = print;
}

void cuda_deleter(void *ptr)
{
    cudaFree(ptr);
}

// Preprocessing functions
std::tuple<at::Tensor, at::Tensor> preprocess(const at::Tensor &dataset, 
    c10::optional<float> threshold_);
std::tuple<at::Tensor, at::Tensor, at::Tensor> preprocess_kmeans(const at::Tensor &dataset, 
    int num_clusters, c10::optional<float> threshold_);

// Compression functions
at::Tensor get_compress_size(const at::Tensor &dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const c10::optional<at::Tensor> &index_array_,
    const c10::optional<at::Tensor> &compress_ctr_);
at::Tensor compress_inplace(const at::Tensor &dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const c10::optional<at::Tensor> &index_array_);
std::tuple<at::Tensor, at::Tensor> compress_condensed(const at::Tensor &dataset, 
    const at::Tensor &mask, const at::Tensor &bitval, 
    const c10::optional<at::Tensor> &index_array_);

// Decompression functions
at::Tensor decompress_fetch(const at::Tensor &comp_dataset, const at::Tensor &mask, 
    const at::Tensor &bitval, const at::Tensor &bitmask, const torch::Device out_device, 
    const c10::optional<int> &comp_len_, const c10::optional<at::Tensor> &index_array_,
    const c10::optional<int> nblks_, const c10::optional<int> nthreads_, c10::optional<int> impl_);

#include <ATen/Functions.h>
#include <ATen/MapAllocator.h>
#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/TensorOptions.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
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
    m.def("preprocess_kmeans", &preprocess_kmeans, "Generate mask and bitval for input dataset");
    m.def("get_compress_size", &get_compress_size, "Get compressed size of input dataset");
    m.def("compress_inplace", &compress_inplace, "Convert input dataset into compressed form (occupies same space)");
    m.def("compress_condensed", &compress_condensed, "Return compressed form of input and byte offsets");
    m.def("decompress_fetch", &decompress_fetch, "Decompress dataset into new tensor");
    m.def("read_shared", &read_shared, "Read tensor from file into shared memory (avoids excessive mem usage)");
    //m.def("decompress", &decompress, "Decompressed form of input dataset and byte offsets");
}