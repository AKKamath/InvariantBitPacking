#include <stdio.h>
#include "ibp_preproc_kernels.cuh"
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
    T **comp_bitval, float threshold = -1.0, int chunk_size = 4)
{
    // Init data final mask and bitval for future use
    cudaMalloc(comp_mask, vec_size * sizeof(T));
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
    count_bit_kernel<<<320, 128>>> (input_arr, num_vecs, vec_size, d_num_bits);
    cudaDeviceSynchronize();
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
        create_mask<<<1, 512>>> (d_num_bits, d_mask, d_bitval, num_vecs, vec_size, threshold);
        // Find number of bits set in mask (theoretical bit savings)
        cudaMemcpy(h_mask, d_mask, vec_size * sizeof(T), cudaMemcpyDeviceToHost);
        cudaCheckError();
        int popc = 0, count = 0;
        for(ull i = 0; i < vec_size; ++i) {
            POPC(count, h_mask[i]);
            popc += count;
        }
        DPRINTF("Saved %d bits of %d (%.2f%%)\n", popc, vec_size * sizeof(T) * 8, 
            (double)(popc) * 100.0 / ((double)vec_size * sizeof(T) * 8.0));
        
        // Count real bits saved in the dataset
        cudaMemset(d_bits_saved, 0, sizeof(long long unsigned));
        check_feats<<<320, 512>>> (input_arr, num_vecs, vec_size, d_mask, d_bitval, d_bits_saved);
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
int preproc_kmeans(T *input_arr, ull num_vecs, ull vec_size, T **comp_mask, 
    T **comp_bitval, float threshold = -1.0, int chunk_size = 4)
{
    return 0;
/*
    int32_t *dev_centroids_count;
    int *host_centroid_count;
    int32_t *dev_cluster;
    int32_t *masks, *multivals;
    int32_t *dist_vector, *host_vector;
    int32_t *host_pick, *dev_pick;
    unsigned num_centroids = 100;
    cudaMalloc(&dist_vector, nodes_per_gpu * sizeof(int32_t));
    cudaMallocHost(&host_vector, nodes_per_gpu * sizeof(int32_t));
    cudaMalloc(&dev_cluster, nodes_per_gpu * sizeof(int32_t));
    cudaMalloc(&dev_centroids_count, num_centroids * sizeof(int32_t));
    cudaMallocHost(&host_centroid_count, num_centroids * sizeof(int));
    cudaMalloc(&masks, num_centroids * feature_len * sizeof(int32_t));
    cudaMalloc(&multivals, num_centroids * feature_len * sizeof(int32_t));
    cudaMemset(dev_cluster, 0, nodes_per_gpu * sizeof(int32_t));
    cudaMemcpy(index_arr, index_array, nodes_per_gpu * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(masks, -1, num_centroids * feature_len * sizeof(int32_t));
    cudaMemcpy(multivals, &typecast_feats[index_arr[0] * feature_len], feature_len * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // K-means++ initialization algorithm
    for(int i = 1; i < num_centroids; ++i) {
        calc_distances<<<32, 512>>>(multivals, i, typecast_feats, index_array, nodes_per_gpu, dist_vector, feature_len);
        pick_max_distance<<<1, 512>>>(dist_vector, nodes_per_gpu, dev_pick);
        cudaMemcpy(host_pick, dev_pick, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_vector, dist_vector, nodes_per_gpu * sizeof(int32_t), cudaMemcpyDeviceToHost);
        //DPRINTF("Picked %d; dist %d\n", *host_pick, host_vector[*host_pick]);
        int64_t nodeId = index_arr[*host_pick];
        cudaMemcpy(&multivals[i * feature_len], &typecast_feats[nodeId * feature_len], feature_len * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
    cudaCheckError();

    const int ITERS = 00;
    DPRINTF("Iter ");
    for(int i = 0; i < ITERS; ++i) {
        DPRINTF("%d, ", i);
        if(i % 20 == 0 && i > 0)
            DPRINTF("\n");
        cudaMemset(dev_centroids_count, 0, num_centroids * sizeof(int32_t));
        cudaDeviceSynchronize();
        classify_nodes<<<32, 512>>>(masks, multivals, num_centroids, typecast_feats, index_array, nodes_per_gpu, dev_cluster, feature_len);
        create_mask_many<<<50, 512>>> (masks, multivals, num_centroids, dev_centroids_count, typecast_feats, index_array, dev_cluster, feature_len, nodes_per_gpu, 0.9);
        cudaDeviceSynchronize();
        cudaCheckError();
        if(i == 0) {
            cudaMemcpy(host_centroid_count, dev_centroids_count, num_centroids * sizeof(int), cudaMemcpyDeviceToHost);
            DPRINTF("Before clustering\n");
            for(int i = 0; i < num_centroids; ++i) {
                cudaMemcpy(h_mask, &masks[i * feature_len], feature_len * sizeof(int), cudaMemcpyDeviceToHost);
                int popc = 0;
                for(int maskId = 0; maskId < feature_len; ++maskId) {
                    //DPRINTF("%x ", h_mask[maskId]);
                    popc += POPC(h_mask[maskId]);
                }
                if(host_centroid_count[i])
                    DPRINTF("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
                        feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0), host_centroid_count[i]);
            }
        }
    }
    DPRINTF("Finished clustering\n");

    cudaMemcpy(host_centroid_count, dev_centroids_count, num_centroids * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_centroids; ++i) {
        cudaMemcpy(h_mask, &masks[i * feature_len], feature_len * sizeof(int), cudaMemcpyDeviceToHost);
        int popc = 0;
        for(int maskId = 0; maskId < feature_len; ++maskId) {
            //DPRINTF("%x ", h_mask[maskId]);
            popc += POPC(h_mask[maskId]);
        }
        if(host_centroid_count[i])
            DPRINTF("Centroid %d: Set bits %d of %d (%f%%) for %d nodes\n", i, popc, 
                feature_len * 32, (double)(popc) * 100.0 / ((double)feature_len * 32.0), host_centroid_count[i]);
    }

    DPRINTF("Num nodes: %ld, num_centroids %d\n", nodes_per_gpu, num_centroids);
    for(float threshold = 0.7; threshold <= 1.0; threshold += 0.05) {
        cudaMemset(count_stuff, 0, sizeof(long long unsigned));
        create_mask_many<<<50, 512>>> (masks, multivals, num_centroids, dev_centroids_count, typecast_feats, index_array, dev_cluster, feature_len, nodes_per_gpu, threshold);
        check_feats_many<<<50, 512>>>(typecast_feats, index_array, nodes_per_gpu, dev_cluster, feature_len, masks, multivals, count_stuff, false);
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaMemcpy(host_count, count_stuff, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
        DPRINTF("KMeans %f: counts %llu (Total %ld, %.3f%%)\n", threshold, *host_count, 
            nodes_per_gpu * feature_len * 32, (double)*host_count * 100 / (nodes_per_gpu * feature_len * 32.0));
    }
    cudaFree(masks);
    cudaFree(multivals);
    cudaFree(dev_cluster);
    cudaFree(dev_centroids_count);
    cudaFreeHost(host_centroid_count);
    cudaFree(dist_vector);
    cudaFreeHost(host_vector);
    */
}
}