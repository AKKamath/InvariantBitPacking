#include "ibp_helpers.cuh"

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