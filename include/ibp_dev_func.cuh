#include "ibp_helpers.cuh"
// Function to compress and write features
__inline__ __device__ void compress_and_write(int32_t *dest, int32_t *src, 
    int32_t *mask, int32_t *bitval, int32_t feature_len, int32_t chunk_size = 4)
{
    int laneId = threadIdx.x % DWARP_SIZE;
    // First determine metadata bits
    int64_t bitmask_offset = BITS_TO_BYTES((feature_len * sizeof(float) + chunk_size - 1) / chunk_size);
    // 4-byte align
    bitmask_offset = (bitmask_offset + 3) / 4 * 4;
    int32_t bitshift = 0;
    for(int j = laneId; j < (feature_len + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; j += DWARP_SIZE) {
        int32_t cur_bitshift = 0;
        int32_t val = 0;
        if(j < feature_len) {
            int32_t local_mask = mask[j];
            // Read WORD from feature
            val = src[j];
            // Init with bits in this float
            cur_bitshift = 32;
            // If this chunk is compressable, subtract the bits saved
            if((val & local_mask) == bitval[j]) {
                cur_bitshift -= __popc(local_mask);
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        //printf("BS %p %d: %d %d\n", dest, j, cur_bitshift, bitshift);
        if(j < feature_len) {
            int32_t local_mask = mask[j];
            int32_t insert_size = 32 - __popc(local_mask);
            int32_t compressed_val = 0;
            // Start from bitshift and insert current compressed feature
            if((val & local_mask) == bitval[j]) {
                //uncomp = true;
                // Number of compressed bits being inserted
                int32_t num_bits = insert_size;
                int32_t temp_insert_size = __clz(local_mask);
                int32_t num_inserted = 0;
                while(num_bits > 0) {
                    num_inserted += temp_insert_size;
                    compressed_val |= (val & (int32_t)(((1L << temp_insert_size) - 1L) << (32 - num_inserted)));
                    //printf("IP %p %d: %x %x (%d, %d)\n", dest, j, compressed_val, val, temp_insert_size, num_inserted);
                    num_bits -= temp_insert_size;
                    // Shift by inserted bits
                    local_mask <<= temp_insert_size;
                    // Shift by masked bits
                    int32_t shift = __clz(~local_mask);
                    local_mask <<= shift;
                    val <<= shift;
                    // Get new insert size
                    temp_insert_size = min(__clz(local_mask), num_bits);
                }
                //printf("COMP %p %d: %x %x %x (%d)\n", dest, j, compressed_val, src[j], mask[j], insert_size);
                // Set bitmask bit
                int32_t word_offset = j / 32;
                int32_t bit_offset = j % 32;
                atomicOr(dest + word_offset, 1 << bit_offset);
            } else {
                // Ignore mask and insert entire chunk
                insert_size = 32;
                compressed_val = val;
                // Unset bitmask bit
                int32_t word_offset = j / 32;
                int32_t bit_offset =  j % 32;
                atomicAnd(dest + word_offset, ~(1 << (bit_offset)));
                //printf("VAL %p %d: %x (%p)\n", dest, j, compressed_val, src + j);
            }
            int32_t fin_ins_bits = 0;
            while(insert_size > 0) {
                // Now perform actual insertion
                int32_t word_offset = (bitmask_offset + bitshift / 8) / 4;
                int32_t bit_offset = (bitmask_offset * 8 + bitshift) % 32;
                int32_t insert_bits = min(insert_size, 32 - bit_offset);
                int32_t insert = ((compressed_val << fin_ins_bits) & (((1L << insert_bits) - 1L) << (32 - insert_bits))) >> bit_offset;

                //if(uncomp)
                //    printf("PREV %p %d (%d): %x\n", dest, j, word_offset, atomicAdd(dest + word_offset, 0));
                // Insert into cache
                atomicOr(dest + word_offset, insert);
                //printf("%p %d: %x %x %x (%d, %d) (%d, %d)\n", dest, j, compressed_val, insert, 
                //    atomicAdd(dest + word_offset, 0), bit_offset, word_offset, insert_bits, fin_ins_bits);
                // Shift appropriately
                //compressed_val <<= insert_bits;
                fin_ins_bits += insert_bits;
                insert_size -= insert_bits;
                bitshift += insert_bits;
            }
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}

// Function to decompress and write features
__inline__ __device__ void decompress_and_write(int32_t *dest, int32_t *src, 
    int32_t *shm_mask, int32_t *shm_bitval, int32_t feature_len, int32_t chunk_size = 4) {
    int laneId = threadIdx.x % DWARP_SIZE;
    int64_t bitmask_offset = BITS_TO_BYTES((feature_len * sizeof(float) + chunk_size - 1) / chunk_size);
    // 4-byte align
    bitmask_offset = (bitmask_offset + 3) / 4 * 4;
    // Code to decompress
    int32_t bitshift = 0;
    for(int i = laneId; i < (feature_len + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; i += DWARP_SIZE) {
        int32_t cur_bitshift = 0;
        bool compressed_feat = false;
        if(i < feature_len) {
            // Check if this feature is in compressed or uncompressed format
            int32_t word_offset = i / 32;
            int32_t bit_offset =  i % 32;
            compressed_feat = (src[word_offset] & (1 << bit_offset));
            // Default 32 bits per thread, less if compressed
            cur_bitshift = 32;
            if(compressed_feat) {
                cur_bitshift -= __popc(shm_mask[i]);
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        // Now decompress
        if(i < feature_len) {
            int32_t num_bits = cur_bitshift;
            int32_t temp_read_size;
            int32_t local_mask = shm_mask[i];
            int32_t temp_dest = 0;
            // Start from bitshift and insert current compressed feature
            if(compressed_feat) {
                temp_dest = shm_bitval[i];
                temp_read_size = min(num_bits, __clz(local_mask));
            } else {
                temp_dest = 0;
                temp_read_size = num_bits;
            }

            int32_t fin_read_bits = 0;
            // Number of compressed bits being inserted
            while(num_bits > 0) {
                // Perform actual read
                int32_t word_offset = (bitmask_offset + bitshift / 8) / 4;
                int32_t bit_offset = (bitmask_offset * 8 + bitshift) % 32;
                int32_t read_bits = min(temp_read_size, 32 - bit_offset);
                temp_dest |= ((src[word_offset] << bit_offset) & (((1L << read_bits) - 1L) << (32 - read_bits))) >> fin_read_bits;
                fin_read_bits += read_bits;
                num_bits -= read_bits;
                bitshift += read_bits;
                // If the feature was compressed, do some bitshifts before next iteration
                if(compressed_feat) {
                    // Shift by inserted bits
                    local_mask <<= read_bits;
                    // Shift by masked bits
                    int32_t shift = __clz(~local_mask);
                    local_mask <<= shift;
                    fin_read_bits += shift;
                    // Get new insert size
                    temp_read_size = min(__clz(local_mask), num_bits);
                } else
                    temp_read_size = num_bits;
            }
            dest[i] = temp_dest;
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}

__inline__ __device__ int read_one_iter(const int32_t *cpu_src, int32_t *shm_meta, int32_t *shm_working, 
    int min_elems, int max_elems, int bitmask_offset, int start_offset = 0) 
{
    int threadId = threadIdx.x % DWARP_SIZE;
    const int offset = ((GPU_CL_SIZE - (((uint64_t)(cpu_src + start_offset)) % GPU_CL_SIZE)) % GPU_CL_SIZE) / sizeof(int32_t);
    const int onset  = ((GPU_CL_SIZE - (((uint64_t)min_elems * sizeof(int32_t)) % GPU_CL_SIZE)) % GPU_CL_SIZE) / sizeof(int32_t);

    int sub_val = (offset + 7) / 8 * 8; // Round up to nearest multiple of 8
    for(int k = threadId + (offset - sub_val); k < offset; k += DWARP_SIZE) {
        int32_t *dest_element;
        if(k + start_offset < bitmask_offset / sizeof(int32_t)) {
            // ASSUMPTION: Shared memory working region size is 128B/32 elements
            dest_element = (int32_t*)shm_meta + (k + start_offset) % 32;
        } else {
            // ASSUMPTION: Shared memory working region size is 256B/64 elements
            dest_element = (int32_t*)shm_working + (k + start_offset - bitmask_offset / sizeof(int32_t)) % 64;
        }
        // Combined read, to improve PCIe util.
        int32_t src_data = *((cpu_src + start_offset) + k);
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
        int add_val = (min_elems - offset + 7) / 8 * 8; // Round up to nearest multiple of 8
        for(int k = threadId + offset; k < offset + add_val; k += DWARP_SIZE) {
            int32_t *dest_element;
            if(k + start_offset < bitmask_offset / sizeof(int32_t)) {
                // ASSUMPTION: Shared memory working region size is 128B/32 elements
                dest_element = (int32_t*)shm_meta + (k + start_offset) % 32;
            } else {
                // ASSUMPTION: Shared memory working region size is 256B/64 elements
                dest_element = (int32_t*)shm_working + (k + start_offset - bitmask_offset / sizeof(int32_t)) % 64;
            }
            // Combined read, to improve PCIe util.
            int32_t src_data = *((cpu_src + start_offset) + k);
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
        int32_t *dest_element;
        if(k + start_offset < bitmask_offset / sizeof(int32_t)) {
            // ASSUMPTION: Shared memory working region size is 128B/32 elements
            dest_element = (int32_t*)shm_meta + (k + start_offset) % 32;
        } else {
            // ASSUMPTION: Shared memory working region size is 256B/64 elements
            dest_element = (int32_t*)shm_working + (k + start_offset - bitmask_offset / sizeof(int32_t)) % 64;
        }
        // Combined read, to improve PCIe util.
        int32_t src_data = *((cpu_src + start_offset) + k);
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

// Function to decompress and write features
__inline__ __device__ void decompress_and_write_cpu(int32_t *dest, const int32_t *src, 
    int32_t *shm_mask, int32_t *shm_bitval, const int32_t feature_len, const int32_t compressed_len, int32_t *workspace,
    const int32_t *dev_mask, const int32_t *dev_bitval, int shmem_elems, int32_t chunk_size = 4) {
    int laneId = threadIdx.x % DWARP_SIZE;
    int64_t bitmask_offset = BITS_TO_BYTES((feature_len * sizeof(float) + chunk_size - 1) / chunk_size);
    // 4-byte align
    bitmask_offset = (bitmask_offset + 3) / 4 * 4;

    // Shared memory buffers
    int32_t *metadata = workspace;
    int32_t *working_data = workspace + 32;
    int metadata_offset = 0, working_offset = 0;
    // Read up to 64elems/256B from src buffer
    int offset = read_one_iter(src, metadata, working_data, 1, 
                               min(32, feature_len), bitmask_offset);

    if(offset > bitmask_offset / sizeof(int32_t)) {
        metadata_offset = bitmask_offset / sizeof(int32_t);
        working_offset = offset;
        /*if(laneId == 0)
        printf("Offset: %d, Metadata offset = %d, working offset = %d\n", 
            offset, metadata_offset, working_offset);*/
    } else {
        metadata_offset = offset;
        // Only read metadata so far, so read working data now
        working_offset = bitmask_offset / sizeof(int32_t);
        working_offset = read_one_iter(src, metadata, working_data, min(32, feature_len), 
                                min(64, feature_len), bitmask_offset, working_offset);
        /*if(laneId == 0)
        printf("Offset: %d, Metadata offset = %d, working offset = %d\n", 
            offset, metadata_offset, working_offset);*/
    }

    // Code to decompress
    int32_t bitshift = 0;
    for(int i = laneId; i < (feature_len + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; i += DWARP_SIZE) {
        int read_metadata = false;
        // This thread needs next set of metadata
        if(i / 32 >= metadata_offset && i < feature_len) {
            read_metadata = true;
        }
        read_metadata = __ballot_sync(FULL_MASK, read_metadata);
        if(read_metadata) {
            // Read next 128B of metadata
            metadata_offset = read_one_iter(src, metadata, working_data, 
                min(32UL, bitmask_offset / sizeof(int32_t) - metadata_offset), 
                min(32UL, bitmask_offset / sizeof(int32_t) - metadata_offset), bitmask_offset, metadata_offset);
            /*if(laneId == 0)
                printf("2. [Added] Metadata offset = %d, working offset = %d\n", 
                    metadata_offset, working_offset);*/
        }
        int32_t cur_bitshift = 0;
        bool compressed_feat = false;
        if(i < feature_len) {
            // Check if this feature is in compressed or uncompressed format
            // i / 32 converts bits to words, % 32 because shmem buffer holds 32 words
            int32_t word_offset = (i / 32) % 32;
            int32_t bit_offset =  i % 32;
            compressed_feat = (metadata[word_offset] & (1 << bit_offset));
            // Default 32 bits per thread, less if compressed
            cur_bitshift = 32;
            if(compressed_feat) {
                int32_t mask;
                if(i < shmem_elems)
                    mask = shm_mask[i];
                else
                    mask = dev_mask[i];
                cur_bitshift -= __popc(mask);
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        //int read_data = false;
        // This thread needs next set of working data
        //if((bitshift + cur_bitshift) / 32 >= working_offset - bitmask_offset / sizeof(int32_t) && i < feature_len) {
        //    read_data = true;
        //}
        //read_data = __ballot_sync(FULL_MASK, read_data);
        // See whether last thread's bitshift exceeds current data in shared memory
        int lastbit_read = __shfl_sync(FULL_MASK, (bitshift + cur_bitshift) / 32, DWARP_SIZE - 1);
        if(lastbit_read >= working_offset - bitmask_offset / sizeof(int32_t)) {
            // Read next 128B of metadata
            working_offset = read_one_iter(src, metadata, working_data, 
                min(32, max((int32_t) (1 + lastbit_read - working_offset + bitmask_offset / sizeof(int32_t)), 
                    compressed_len - working_offset)), 
                min(64, feature_len - working_offset), bitmask_offset, working_offset);
            /*if(laneId == 0)
                printf("3. Metadata offset = %d, [Added] working offset = %d\n", 
                    metadata_offset, working_offset);*/
        }
        
        // Now decompress
        if(i < feature_len) {
            int32_t num_bits = cur_bitshift;
            int32_t temp_read_size;
            int32_t local_mask;
            if(i < shmem_elems)
                local_mask = shm_mask[i];
            else
                local_mask = dev_mask[i];
            int32_t temp_dest = 0;
            // Start from bitshift and insert current compressed feature
            if(compressed_feat) {
                if(i < shmem_elems)
                    temp_dest = shm_bitval[i];
                else
                    temp_dest = dev_bitval[i];
                temp_read_size = min(num_bits, __clz(local_mask));
            } else {
                temp_dest = 0;
                temp_read_size = num_bits;
            }

            int32_t fin_read_bits = 0;
            // Number of compressed bits being inserted
            while(num_bits > 0) {
                // Perform actual read
                // bitshift / 32 converts bits to words, % 64 because shmem buffer holds 64 words
                int32_t word_offset = (bitshift / 32) % 64;
                int32_t bit_offset = bitshift % 32;
                int32_t read_bits = min(temp_read_size, 32 - bit_offset);
                temp_dest |= ((working_data[word_offset] << bit_offset) & (((1L << read_bits) - 1L) << (32L - read_bits))) >> fin_read_bits;
                fin_read_bits += read_bits;
                num_bits -= read_bits;
                bitshift += read_bits;
                // If the feature was compressed, do some bitshifts before next iteration
                if(compressed_feat) {
                    // Shift by inserted bits
                    local_mask <<= read_bits;
                    // Shift by masked bits
                    int32_t shift = __clz(~local_mask);
                    local_mask <<= shift;
                    fin_read_bits += shift;
                    // Get new insert size
                    temp_read_size = min(__clz(local_mask), num_bits);
                } else
                    temp_read_size = num_bits;
            }
            dest[i] = temp_dest;
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}