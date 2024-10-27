#ifndef IBP_DECOMPRESS_DEV
#define IBP_DECOMPRESS_DEV
#include "ibp_helpers.cuh"

#define SHM_META 128
#define SHM_WORK 256

namespace ibp {
// Read one iteration of data from CPU to shared memory
template<typename T>
__inline__ __device__ int read_one_iter(const T *cpu_src, T *shm_meta, T *shm_working, 
    int min_elems, int max_elems, int bitmask_offset, int start_offset = 0) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    const int offset = ((DWARP_SIZE * sizeof(T) - (((uint64_t)(cpu_src + start_offset)) % (DWARP_SIZE * sizeof(T)))) % (DWARP_SIZE * sizeof(T))) / sizeof(T);
    const int onset  = ((DWARP_SIZE * sizeof(T) - (((uint64_t)min_elems * sizeof(T)) % (DWARP_SIZE * sizeof(T)))) % (DWARP_SIZE * sizeof(T))) / sizeof(T);

    const int elems_per_32B = 32 / sizeof(T);
    // Round up to nearest 32-byte boundary
    int sub_val = (offset + elems_per_32B - 1) / elems_per_32B * elems_per_32B;
    for(int k = threadId + (offset - sub_val); k < offset; k += DWARP_SIZE) {
        T *dest_element;
        if(k + start_offset < bitmask_offset / sizeof(T)) {
            // ASSUMPTION: Shared memory working region size is 128B
            dest_element = (T*)shm_meta + (k + start_offset) % (128 / sizeof(T));
        } else {
            // ASSUMPTION: Shared memory working region size is 256B
            dest_element = (T*)shm_working + (k + start_offset - bitmask_offset / sizeof(T)) % (256 / sizeof(T));
        }
        // Combined read, to improve PCIe util.
        T src_data = *((cpu_src + start_offset) + k);
        // Selective write to GPU memory
        if(k >= 0 && k < max_elems) {
            *dest_element = src_data;
            
            /*unsigned int mask = __activemask();
            if(k % 32 == 0) {
                printf("1. Write %d bytes; num_threads %d; %p; min %d max %d\n", 
                    (int)(__popc(mask) * sizeof(int32_t)), sub_val, 
                    cpu_src + start_offset, min_elems, max_elems);
            }*/
        }
    }
    if(offset >= min_elems)
        return start_offset + offset;
    // Read can be completed with less than 32 threads if min_elems is small
    if(max_elems - offset <= 24) {
        // Round up to nearest 32-byte boundary
        int add_val = (min_elems - offset + elems_per_32B - 1) / elems_per_32B * elems_per_32B; 
        for(int k = threadId + offset; k < offset + add_val; k += DWARP_SIZE) {
            T *dest_element;
            if(k + start_offset < bitmask_offset / sizeof(T)) {
                // ASSUMPTION: Shared memory working region size is 128B
                dest_element = (T*)shm_meta + (k + start_offset) % (SHM_META / sizeof(T));
            } else {
                // ASSUMPTION: Shared memory working region size is 256B/64 elements
                dest_element = (T*)shm_working + (k + start_offset - bitmask_offset / sizeof(T)) % (SHM_WORK / sizeof(T));
            }
            // Combined read, to improve PCIe util.
            T src_data = *((cpu_src + start_offset) + k);
            // Selective write to GPU memory
            if(k >= 0 && k < max_elems) {
                *dest_element = src_data;
                
                /*unsigned int mask = __activemask();
                if(k % 32 == 0) {
                    printf("1b. Write %d bytes; num_threads %d; %p; min %d max %d\n", 
                        (int)(__popc(mask) * sizeof(int32_t)), add_val, 
                        cpu_src + start_offset, min_elems, max_elems);
                }*/
            }
        }
        __syncwarp();
        return start_offset + min(max_elems, offset + add_val);
    }
    int k;
    for(k = threadId + offset; k < min_elems + offset + onset; k += DWARP_SIZE) {
        __syncwarp();
        T *dest_element;
        if(k + start_offset < bitmask_offset / sizeof(T)) {
            // ASSUMPTION: Shared memory working region size is 128B
            dest_element = (T*)shm_meta + (k + start_offset) % (SHM_META / sizeof(T));
        } else {
            // ASSUMPTION: Shared memory working region size is 256B
            dest_element = (T*)shm_working + (k + start_offset - bitmask_offset / sizeof(T)) % (SHM_WORK / sizeof(T));
        }
        // Combined read, to improve PCIe util.
        T src_data = *((cpu_src + start_offset) + k);
        // Selective write to GPU memory
        if(k < max_elems) {
            *dest_element = src_data;
            
            /*unsigned int mask = __activemask();
            if(k % 32 == 0) {
                printf("2. Write %d bytes; offset %d; min+onset %d; %p; min %d max %d\n", 
                (int)(__popc(mask) * sizeof(int32_t)), offset, min_elems + onset, 
                cpu_src + start_offset, min_elems, max_elems);
            }*/
        }
    }
    __syncwarp();

    return start_offset + min(max_elems, min_elems + offset + onset);
}

// Function to decompress and write vectors; optimized for src in CPU memory
template<bool FITS_SHMEM, typename T>
__inline__ __device__ void decompress_fetch_cpu(T *dest, const T *src, 
    T *shm_mask, T *shm_bitval, const int32_t vec_size, const int32_t compressed_len, T *workspace,
    const T *dev_mask = nullptr, const T *dev_bitval = nullptr, int shmem_elems = 0) {
    int laneId = threadIdx.x % DWARP_SIZE;
    int64_t bitmask_offset = BITS_TO_BYTES(vec_size);
    // Datatype align
    bitmask_offset = (bitmask_offset + sizeof(T) - 1) / sizeof(T) * sizeof(T);

    // Shared memory buffers
    T *metadata = workspace;
    T *working_data = workspace + SHM_META / sizeof(T);
    int metadata_offset = 0, working_offset = 0;
    // Read up to 64elems/256B from src buffer
    int offset = read_one_iter(src, metadata, working_data, 1, 
        min(SHM_META / sizeof(T), (unsigned long)vec_size), bitmask_offset);

    if(offset > bitmask_offset / sizeof(T)) {
        metadata_offset = bitmask_offset / sizeof(T);
        working_offset = offset;
        /*if(laneId == 0)
        printf("Offset: %d, Metadata offset = %d, working offset = %d\n", 
            offset, metadata_offset, working_offset);*/
    } else {
        metadata_offset = offset;
        // Only read metadata so far, so read working data now
        working_offset = bitmask_offset / sizeof(T);
        working_offset = read_one_iter(src, metadata, working_data, 
                            min(SHM_META / sizeof(T), (unsigned long)vec_size), 
                            min(SHM_WORK / sizeof(T), (unsigned long)vec_size), 
                            bitmask_offset, working_offset);
        /*if(laneId == 0)
        printf("Offset: %d, Metadata offset = %d, working offset = %d\n", 
            offset, metadata_offset, working_offset);*/
    }

    // Code to decompress
    int32_t bitshift = 0;
    for(int i = laneId; i < (vec_size + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; i += DWARP_SIZE) {
        int read_metadata = false;
        // This thread needs next set of metadata
        if(i / (8 * sizeof(T)) >= metadata_offset && i < vec_size) {
            read_metadata = true;
        }
        read_metadata = __ballot_sync(FULL_MASK, read_metadata);
        if(read_metadata) {
            // Read next 128B of metadata
            metadata_offset = read_one_iter(src, metadata, working_data, 
                min(SHM_META / sizeof(T), bitmask_offset / sizeof(T) - metadata_offset), 
                min(SHM_META / sizeof(T), bitmask_offset / sizeof(T) - metadata_offset), 
                bitmask_offset, metadata_offset);
            /*if(laneId == 0)
                printf("2. [Added] Metadata offset = %d, working offset = %d\n", 
                    metadata_offset, working_offset);*/
        }
        int32_t cur_bitshift = 0;
        bool compressed_feat = false;
        if(i < vec_size) {
            // Check if this feature is in compressed or uncompressed format
            // i / 32 converts bits to words, % 32 because shmem buffer holds 32 words
            int32_t elem_offset = (i / (8 * sizeof(T))) % (SHM_META / sizeof(T));
            int32_t bit_offset =  i % (8 * sizeof(T));
            compressed_feat = (metadata[elem_offset] & (1ull << bit_offset));
            // Default 32 bits per thread, less if compressed
            cur_bitshift = sizeof(T) * 8;
            if(compressed_feat) {
                T mask;
                if constexpr(!FITS_SHMEM) {
                    if(i < shmem_elems)
                        mask = shm_mask[i];
                    else
                        mask = dev_mask[i];
                } else
                    mask = shm_mask[i];
                cur_bitshift -= POPC(mask);
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        // See whether last thread's bitshift exceeds current data in shared memory
        int lastbit_read = __shfl_sync(FULL_MASK, (bitshift + cur_bitshift) / (sizeof(T) * 8), DWARP_SIZE - 1);
        if(lastbit_read >= working_offset - bitmask_offset / sizeof(T)) {
            // Read next 128B of metadata
            working_offset = read_one_iter(src, metadata, working_data, 
                min(SHM_META / sizeof(T), (unsigned long)
                    max((int32_t) (1 + lastbit_read - working_offset + bitmask_offset / sizeof(T)), 
                        compressed_len - working_offset)), 
                min(SHM_WORK / sizeof(T), (unsigned long)vec_size - working_offset), 
                bitmask_offset, working_offset);
            /*if(laneId == 0)
                printf("3. Metadata offset = %d, [Added] working offset = %d\n", 
                    metadata_offset, working_offset);*/
        }
        
        // Now decompress
        if(i < vec_size) {
            int32_t num_bits = cur_bitshift;
            int32_t temp_read_size;
            T local_mask;
            if constexpr(!FITS_SHMEM) {
                if(i < shmem_elems)
                    local_mask = shm_mask[i];
                else
                    local_mask = dev_mask[i];
            } else
                local_mask = shm_mask[i];
            T temp_dest = 0;
            // Start from bitshift and insert current compressed feature
            if(compressed_feat) {
                if constexpr(!FITS_SHMEM) {
                    if(i < shmem_elems)
                        temp_dest = shm_bitval[i];
                    else
                        temp_dest = dev_bitval[i];
                } else
                    temp_dest = shm_bitval[i];
                temp_read_size = min(num_bits, CLZ(local_mask));
            } else {
                temp_dest = 0;
                temp_read_size = num_bits;
            }

            int32_t fin_read_bits = 0;
            // Number of compressed bits being inserted
            while(num_bits > 0) {
                // Perform actual read
                // bitshift / 32 converts bits to words, % 64 because shmem buffer holds 64 words
                int32_t elem_offset = (bitshift / (8 * sizeof(T))) % 64;
                int32_t bit_offset = bitshift % (8 * sizeof(T));
                int32_t read_bits = min(temp_read_size, (int32_t)(8 * sizeof(T)) - bit_offset);
                temp_dest |= ((working_data[elem_offset] << bit_offset) & (((1L << read_bits) - 1L) << ((8 * sizeof(T)) - read_bits))) >> fin_read_bits;
                fin_read_bits += read_bits;
                num_bits -= read_bits;
                bitshift += read_bits;
                // If the feature was compressed, do some bitshifts before next iteration
                if(compressed_feat) {
                    // Shift by inserted bits
                    local_mask <<= read_bits;
                    // Shift by masked bits
                    T shift = CLZ(~local_mask);
                    local_mask <<= shift;
                    fin_read_bits += shift;
                    // Get new insert size
                    temp_read_size = min(CLZ(local_mask), num_bits);
                } else
                    temp_read_size = num_bits;
            }
            dest[i] = temp_dest;
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}
} // namespace ibp
#endif // IBP_DECOMPRESS_DEV