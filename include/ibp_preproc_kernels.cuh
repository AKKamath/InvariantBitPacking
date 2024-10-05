#include "ibp_helpers.cuh"
// TODO: Potential optimization: tiled count
// Count number of bits set and unset in the input array

template<typename T>
__global__ void count_bit_kernel(T *input_arr, ull num_elems, ull elem_size, int32_t *bit_count) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8, 
        "Data type must be castable to integer");
    for(ull i = blockIdx.x; i < num_elems; i += gridDim.x) {
        for(ull j = threadIdx.x; j < elem_size; j += blockDim.x) {
            T val = ((T*)input_arr)[i * elem_size + j];
            for(ull bit = 0; bit < sizeof(T) * 8; ++bit) {
                if(val & (1ull << bit)) \
                    atomicAdd(&bit_count[j * sizeof(T) * 8 + bit], 1);
            }
        }
    }
}

template<typename T>
__global__ void create_mask(int32_t *bit_count, T *mask, T *vals, ull num_elems, ull elem_size, float threshold) {
    assert(gridDim.x == 1);
    for(ull i = threadIdx.x; i < elem_size; i += blockDim.x) {
        // Construct bitval and mask
        T val = 0;
        T masker = 0;
        for(ull j = 0; j < sizeof(T) * 8; ++j) {
            if(bit_count[i * sizeof(T) * 8 + j] > threshold * num_elems) {
                val |= (1ull << j);
                masker |= (1ull << j);
            } else if(bit_count[i * sizeof(T) * 8 + j] < (1.0 - threshold) * num_elems) {
                masker |= (1ull << j);
            }
        }
        vals[i] = val;
        mask[i] = masker;
    }
}

template<typename T>
__global__ void check_feats(T *input_arr, ull num_elems, ull elem_size, 
                            T *mask, T *vals, ull *bits_saved) {
    // Store running bits saved count
    __shared__ ull bit_ctr;
    bit_ctr = 0;
    __syncthreads();
    for(ull i = blockIdx.x; i < num_elems; i += gridDim.x) {
        // For each element, find bits saved
        for(ull j = threadIdx.x; j < elem_size; j += blockDim.x) {
            T val = input_arr[i * elem_size + j];
            if((val & mask[j]) == vals[j]) {
                int count = 0;
                POPC(count, mask[j]);
                atomicAdd(&bit_ctr, count);
            }
        }
        __syncthreads();
        // Need elem_size bits of metadata, so only worth if at least that many saved
        if(bit_ctr > elem_size && threadIdx.x == 0) {
            atomicAdd(bits_saved, bit_ctr - elem_size);
        }
        __syncthreads();
        bit_ctr = 0;
        __syncthreads();
    }
}