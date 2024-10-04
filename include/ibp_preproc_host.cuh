#include "ibp_preproc_kernels.cuh"

// TODO: Variable chunk size
/*
    Input: 
        input_arr - 2D array of input data to be compressed [num_elems x elem_size]
        num_elems - Number of elements in input_arr
        elem_size - Size of each element in input_arr 
        chunk_size - Size of each chunk
    Output:
        comp_mask - Returned mask for compression. Size is elem_size.
        comp_bitval - Returned bit values for compression. Size is elem_size.
    Optional:
        chunk_size [TODO] - Size of each chunk
        threshold - Fixed threshold for mask/bitval calculation. 
                    Default is -1, which means we will find the best

    Returns: Average compressed length of each element

*/
template<typename T>
int ibp_preproc_data(T *input_arr, ull num_elems, ull elem_size, T **comp_mask, 
    T **comp_bitval, float threshold = -1.0, int chunk_size = 4)
{
    // Init data final mask and bitval for future use
    cudaMalloc(comp_mask, elem_size * sizeof(T));
    cudaMalloc(comp_bitval, elem_size * sizeof(T));

    // One mask and bitval for entire dataset
    T *d_mask, *d_bitval;
    T *h_mask;
    cudaMalloc(&d_mask, elem_size * sizeof(T));
    cudaMalloc(&d_bitval, elem_size * sizeof(T));
    cudaMallocHost(&h_mask, elem_size * sizeof(T));

    ull *d_bits_saved;
    ull *h_bits_saved;
    cudaMalloc(&d_bits_saved, sizeof(ull));
    cudaMallocHost(&h_bits_saved, sizeof(ull));
    cudaCheckError();

    // Each counter is an int per bit of input element, so we need
    // elem_size inputs, 8 bits per byte, sizeof(T) bytes per input
    int *d_num_bits;
    cudaMalloc(&d_num_bits, elem_size * 8 * sizeof(T) * sizeof(int));
    cudaMemset(d_num_bits, 0, elem_size * 8 * sizeof(T) * sizeof(int));
    count_bit_kernel<<<32, 512>>> (input_arr, num_elems, elem_size, d_num_bits);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Use provided threshold if available, otherwise sweep 0.7 - 1.0
    float min_thresh = threshold < 0 ? 0.7 : threshold;
    float max_thresh = threshold < 0 ? 1.0 : threshold;

    double max_saved = 0;
    int avg_comp_size = 0;
    printf("Num elems: %d\n", num_elems);
    for(threshold = min_thresh; threshold <= max_thresh; threshold += 0.05) {
        // Construct mask and bitval based on threshold
        cudaMemset(d_mask, 0, elem_size * sizeof(T));
        cudaMemset(d_bitval, 0, elem_size * sizeof(T));
        create_mask<<<1, 512>>> (d_num_bits, d_mask, d_bitval, num_elems, elem_size, threshold);
        // Find number of bits set in mask (theoretical bit savings)
        cudaMemcpy(h_mask, d_mask, elem_size * sizeof(T), cudaMemcpyDeviceToHost);
        cudaCheckError();
        int popc = 0;
        // TODO: builtin_popcount is only for int
        for(ull i = 0; i < elem_size; ++i)
            popc += __builtin_popcount(h_mask[i]);
        printf("Saved %d bits of %d (%.2f%%)\n", popc, elem_size * sizeof(T) * 8, 
            (double)(popc) * 100.0 / ((double)elem_size * sizeof(T) * 8.0));
        
        // Count real bits saved in the dataset
        cudaMemset(d_bits_saved, 0, sizeof(long long unsigned));
        check_feats<<<32, 512>>> (input_arr, num_elems, elem_size, d_mask, d_bitval, d_bits_saved);
        cudaMemcpy(h_bits_saved, d_bits_saved, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
        cudaCheckError();
        printf("Threshold %f: Saved bits per element: %llu (Total %ld, %.3f%%)\n", threshold, *h_bits_saved, 
            num_elems * elem_size * sizeof(T) * 8, (double)*h_bits_saved * 100.0 / (num_elems * elem_size * sizeof(T) * 8.0));
        // Store the mask/value for max compressed format
        if(*h_bits_saved > max_saved) {
            cudaMemcpy(*comp_mask, d_mask, elem_size * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(*comp_bitval, d_bitval, elem_size * sizeof(float), cudaMemcpyDeviceToDevice);
            max_saved = *h_bits_saved;
            avg_comp_size = elem_size * (num_elems * elem_size * sizeof(T) * 8.0 - *h_bits_saved) / (num_elems * elem_size * sizeof(T) * 8) + 1;

            printf("Selected threshold %f; compress_len %d\n", threshold, avg_comp_size);
        }
    }
    // Free all memory needed for compression
    cudaFree(d_num_bits);
    cudaFree(d_mask);
    cudaFree(d_bitval);
    cudaFreeHost(h_mask);
    cudaFree(d_bits_saved);
    cudaFreeHost(h_bits_saved);
    return avg_comp_size;
}