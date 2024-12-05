#ifndef IBP_PREPROC_HOST
#define IBP_PREPROC_HOST
#include <stdio.h>
#include <algorithm>
#include "ibp_preproc_kernels.cuh"
#include "ibp_preproc_kmeans.cuh"
namespace ibp {
// TODO: Variable chunk size
/**
 * @brief Compresses input data using a specified chunk size and threshold.
 *
 * @param input_arr 2D array of input data to be compressed [num_vecs x vec_size].
 *                  Should be accessible by GPU (in GPU memory or pinned memory).
 * @param num_vecs Number of elements in input_arr.
 * @param vec_size Size of each element in input_arr.
 * @param chunk_size [Unused] Size of each chunk.
 * @param comp_mask Returned mask for compression. Size is vec_size.
 * @param comp_bitval Returned bit values for compression. Size is vec_size.
 * @param threshold Fixed threshold for mask/bitval calculation. Default is -1, which means the best threshold will be found.
 * 
 * @return Average compressed length of each element.
 */
template<typename T>
int preproc_data(T *input_arr, ull num_vecs, ull vec_size, T **comp_mask, 
    T **comp_bitval, float threshold = -1.0, cudaStream_t stream = 0)
{
    // Init data final mask and bitval for future use
    if(*comp_mask == nullptr)
        cudaMalloc(comp_mask, vec_size * sizeof(T));
    if(*comp_bitval == nullptr)
        cudaMalloc(comp_bitval, vec_size * sizeof(T));

    // One mask and bitval for entire dataset
    T *d_mask, *d_bitval;
    T *h_mask;
    cudaMalloc(&d_mask, vec_size * sizeof(T));
    cudaMalloc(&d_bitval, vec_size * sizeof(T));
    cudaMallocHost(&h_mask, vec_size * sizeof(T));

    ull *d_bits_saved;
    ull *h_bits_saved;
    cudaMalloc(&d_bits_saved, sizeof(ull));
    cudaMallocHost(&h_bits_saved, sizeof(ull));
    cudaCheckError();

    // Each counter is an int per bit of input element, so we need
    // vec_size inputs, 8 bits per byte, sizeof(T) bytes per input
    int *d_num_bits;
    cudaMalloc(&d_num_bits, vec_size * 8 * sizeof(T) * sizeof(int));
    cudaMemset(d_num_bits, 0, vec_size * 8 * sizeof(T) * sizeof(int));
    count_bit_kernel<<<320, 128, 0, stream>>> (input_arr, num_vecs, vec_size, d_num_bits);
    cudaCheckError();

    // Use provided threshold if available, otherwise sweep 0.7 - 1.0
    float min_thresh = threshold < 0 ? 0.7 : threshold;
    float max_thresh = threshold < 0 ? 1.0 : threshold;

    double max_saved = 0;
    int avg_comp_size = 0;
    DPRINTF("Num elems: %d\n", num_vecs);
    for(threshold = min_thresh; threshold <= max_thresh; threshold += 0.05) {
        // Construct mask and bitval based on threshold
        cudaMemset(d_mask, 0, vec_size * sizeof(T));
        cudaMemset(d_bitval, 0, vec_size * sizeof(T));
        create_mask<<<1, 512, 0, stream>>> (d_num_bits, d_mask, d_bitval, num_vecs, vec_size, threshold);
        // Find number of bits set in mask (theoretical bit savings)
        cudaMemcpy(h_mask, d_mask, vec_size * sizeof(T), cudaMemcpyDeviceToHost);
        cudaCheckError();
        int popc = 0;
        for(ull i = 0; i < vec_size; ++i) {
            popc += POPC(h_mask[i]);
        }
        //DPRINTF("Saved %d bits of %d (%.2f%%)\n", popc, vec_size * sizeof(T) * 8, 
        //    (double)(popc) * 100.0 / ((double)vec_size * sizeof(T) * 8.0));
        
        // Count real bits saved in the dataset
        cudaMemset(d_bits_saved, 0, sizeof(long long unsigned));
        check_feats<<<320, 512, 0, stream>>> (input_arr, num_vecs, vec_size, d_mask, d_bitval, d_bits_saved);
        cudaMemcpy(h_bits_saved, d_bits_saved, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
        cudaCheckError();
        DPRINTF("Threshold %.2f: Saved bits per element: %llu (Total %ld, %.3f%%)\n", threshold, *h_bits_saved, 
            num_vecs * vec_size * sizeof(T) * 8, (double)*h_bits_saved * 100.0 / (num_vecs * vec_size * sizeof(T) * 8.0));
        // Store the mask/value for max compressed format
        if(*h_bits_saved > max_saved) {
            cudaMemcpy(*comp_mask, d_mask, vec_size * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(*comp_bitval, d_bitval, vec_size * sizeof(T), cudaMemcpyDeviceToDevice);
            max_saved = *h_bits_saved;
            avg_comp_size = vec_size * (num_vecs * vec_size * sizeof(T) * 8.0 - *h_bits_saved) / (num_vecs * vec_size * sizeof(T) * 8) + 1;

            DPRINTF("Selected threshold %.2f; compress_len %d\n", threshold, avg_comp_size);
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

// TODO: This
template<typename T>
void preproc_kmeans(T *input_arr, ull num_vecs, ull vec_size, T **comp_mask, 
    T **comp_bitval, unsigned num_centroids, int32_t **dev_cluster, float threshold = -1.0)
{
    int32_t *dev_centroids_count;
    int *host_centroid_count;
    //int32_t *dev_cluster;
    //T *masks, *multivals;
    int32_t *dist_vector, *host_vector;
    int32_t *host_pick, *dev_pick;
    cudaMalloc(&dev_pick, sizeof(int32_t));
    cudaMallocHost(&host_pick, sizeof(int32_t));
    T *h_mask;
    ull *d_counter, *h_counter;
    T *dev_centroids;
    cudaMalloc(&dev_centroids, num_centroids * vec_size * sizeof(T));
    cudaMalloc(&d_counter, sizeof(ull));
    cudaMallocHost(&h_counter, sizeof(ull));
    cudaMallocHost(&h_mask, vec_size * sizeof(T));
    cudaMalloc(&dist_vector, num_vecs * sizeof(int32_t));
    cudaMallocHost(&host_vector, num_vecs * sizeof(int32_t));
    if(*dev_cluster == nullptr)
        cudaMalloc(dev_cluster, num_vecs * sizeof(int32_t));
    cudaMalloc(&dev_centroids_count, num_centroids * sizeof(int32_t));
    cudaMallocHost(&host_centroid_count, num_centroids * sizeof(int));
    if(*comp_mask == nullptr)
        cudaMalloc(comp_mask, num_centroids * vec_size * sizeof(T));
    if(*comp_bitval == nullptr)
        cudaMalloc(comp_bitval, num_centroids * vec_size * sizeof(T));
    cudaMemset(*dev_cluster, 0, num_vecs * sizeof(int32_t));
    //cudaMemcpy(index_arr, index_array, num_vecs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(*comp_mask, -1, num_centroids * vec_size * sizeof(T));
    cudaMemcpy(*comp_bitval, input_arr, vec_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_centroids, input_arr, vec_size * sizeof(T), cudaMemcpyHostToDevice);
    int *d_bits_set;
    cudaMalloc(&d_bits_set, num_centroids * vec_size * 8 * sizeof(T) * sizeof(int));
    cudaCheckError();

    T *masks = *comp_mask, *multivals = *comp_bitval;
    
    // K-means++ initialization algorithm
    /*for(int i = 1; i < min((ull)num_centroids, num_vecs); ++i) {
        GPUMemset<<<32, 512>>>(dist_vector, INT_MAX, num_vecs);
        calc_distances<<<32, 512>>>(input_arr, num_vecs, vec_size, multivals, i, dist_vector);
        pick_max_distance<<<1, 512>>>(dist_vector, num_vecs, dev_pick);
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaMemcpy(host_pick, dev_pick, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_vector, dist_vector, num_vecs * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaCheckError();
        DPRINTF("%d: Picked %d; dist %d\n", i, *host_pick, host_vector[*host_pick]);
        int64_t nodeId = //*host_pick;
        cudaMemcpy(&multivals[i * vec_size], &input_arr[nodeId * vec_size], vec_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(&dev_centroids[i * vec_size], &input_arr[nodeId * vec_size], vec_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaCheckError();
    }*/

    // K-means initialization algorithm (Better compression than K-means++)
    {
        std::vector<int> indices(num_vecs);
        for(int i = 0; i < num_vecs; ++i)
            indices[i] = i;
        std::random_shuffle(indices.begin(), indices.end());
        // K-means initialization algorithm
        for(int i = 1; i < min((ull)num_centroids, num_vecs); ++i) {
            int64_t vec_id = indices[i];
            cudaMemcpy(&multivals[i * vec_size], &input_arr[vec_id * vec_size], vec_size * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(&dev_centroids[i * vec_size], &input_arr[vec_id * vec_size], vec_size * sizeof(T), cudaMemcpyHostToDevice);
            cudaCheckError();
        }
    }
    cudaCheckError();

    const int ITERS = 10;
    DPRINTF("Iter ");
    for(int i = 0; i < ITERS; ++i) {
        DPRINTF("%d, ", i);
        if(i % 20 == 0 && i > 0)
            DPRINTF("\n");
        cudaMemset(dev_centroids_count, 0, num_centroids * sizeof(int32_t));
        cudaMemset(d_bits_set, 0, num_centroids * vec_size * 8 * sizeof(T) * sizeof(int));
        cudaDeviceSynchronize();
        cudaCheckError();
        cluster_vecs<<<320, 256>>>(input_arr, num_vecs, vec_size, masks, multivals, num_centroids, *dev_cluster, dev_centroids);
        compute_new_centroids<<<500, 256>>>(input_arr, num_vecs, vec_size, dev_centroids, num_centroids, dev_centroids_count, d_bits_set, *dev_cluster);
        construct_bitmasks<<<500, 256>>>(input_arr, num_vecs, vec_size, dev_centroids, num_centroids, dev_centroids_count, d_bits_set, *dev_cluster);
        //create_mask_many<<<50, 512>>>(input_arr, vec_size, num_vecs, masks, multivals, num_centroids, dev_centroids_count, dev_cluster, 0.9);
        cudaDeviceSynchronize();
        /*for(int centroid = 0; centroid < num_centroids; ++centroid) {
            cudaMemcpy(h_mask, &dev_centroids[centroid * vec_size], vec_size * sizeof(T), cudaMemcpyDeviceToHost);
            DPRINTF("Centroid %d: ", centroid);
            for(int j = 0; j < vec_size; ++j) {
                DPRINTF("%x ", h_mask[j]);
            }
            DPRINTF("\n");
        }*/
        cudaCheckError();
        /*if(i == 0) {
            create_mask_many<<<100, 256>>>(input_arr, vec_size, num_vecs, masks, multivals, num_centroids, dev_centroids_count, *dev_cluster, 0.9);
            cudaMemcpy(host_centroid_count, dev_centroids_count, num_centroids * sizeof(int), cudaMemcpyDeviceToHost);
            cudaCheckError();
            //DPRINTF("Before clustering\n");
            for(int i = 0; i < num_centroids; ++i) {
                cudaMemcpy(h_mask, &masks[i * vec_size], vec_size * sizeof(T), cudaMemcpyDeviceToHost);
                cudaCheckError();
                int popc = 0;
                for(int maskId = 0; maskId < vec_size; ++maskId) {
                    //DPRINTF("%x ", h_mask[maskId]);
                    popc += POPC(h_mask[maskId]);
                }
                if(host_centroid_count[i])
                    DPRINTF("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
                        vec_size * 32, (double)(popc) * 100.0 / ((double)vec_size * 32.0), host_centroid_count[i]);
            }
        }*/
    }
    DPRINTF("Finished clustering\n");
    create_mask_many<<<100, 256>>>(input_arr, vec_size, num_vecs, masks, multivals, num_centroids, dev_centroids_count, *dev_cluster, 0.9);
    cudaMemcpy(host_centroid_count, dev_centroids_count, num_centroids * sizeof(int), cudaMemcpyDeviceToHost);
    int used_centroids = 0;
    for(int i = 0; i < num_centroids; ++i) {
        if(host_centroid_count[i])
            used_centroids++;
    }
    cudaCheckError();
    /*for(int i = 0; i < num_centroids; ++i) {
        cudaMemcpy(h_mask, &masks[i * vec_size], vec_size * sizeof(T), cudaMemcpyDeviceToHost);
        int popc = 0;
        for(int maskId = 0; maskId < vec_size; ++maskId) {
            //DPRINTF("%x ", h_mask[maskId]);
            popc += POPC(h_mask[maskId]);
        }
        if(host_centroid_count[i])
            DPRINTF("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
                vec_size * 32, (double)(popc) * 100.0 / ((double)vec_size * 32.0), host_centroid_count[i]);
    }*/

   float max_comp = 0;
    DPRINTF("Num nodes: %ld, num_centroids %d, used centroids %d\n", num_vecs, num_centroids, used_centroids);
    for(float threshold = 0.7; threshold <= 1.0; threshold += 0.05) {
        cudaMemset(d_counter, 0, sizeof(long long unsigned));
        create_mask_many<<<160, 256>>>(input_arr, vec_size, num_vecs, masks, multivals, num_centroids, dev_centroids_count, *dev_cluster, threshold);
        check_feats_many<<<50, 512>>>(input_arr, vec_size, num_vecs, masks, multivals, *dev_cluster, d_counter);
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaMemcpy(h_counter, d_counter, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
        float total_size = num_vecs * vec_size * sizeof(T) * 8.0;
        DPRINTF("KMeans %f: counts %llu (Total %ld, %.3f%%)\n", threshold, *h_counter, 
            num_vecs * vec_size * sizeof(T) * 8, (double)*h_counter * 100 / total_size);
        max_comp = max(max_comp, (float)*h_counter / total_size);
    }
    DPRINTF("Centroids %d, compressed: %f\n", used_centroids, max_comp);
    cudaFree(d_bits_set);
    cudaFree(dev_centroids);
    cudaFree(dev_pick);
    cudaFreeHost(host_pick);
    cudaFree(d_counter);
    cudaFreeHost(h_counter);
    //cudaFree(masks);
    //cudaFree(multivals);
    //cudaFree(dev_cluster);
    cudaFree(dev_centroids_count);
    cudaFreeHost(host_centroid_count);
    cudaFree(dist_vector);
    cudaFreeHost(host_vector);
    cudaFreeHost(h_mask);
}
}
#endif // IBP_PREPROC_HOST