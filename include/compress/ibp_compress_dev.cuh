#ifndef IBP_COMPRESS_DEV
#define IBP_COMPRESS_DEV
#include "ibp_helpers.cuh"

namespace ibp {
// Function to compress and write features
template <typename T>
__inline__ __device__ void compress_and_write(T *dest, T *src, ull vec_size, 
    T *mask, T *bitval)
{
    int laneId = threadIdx.x % DWARP_SIZE;
    // First determine metadata bits
    int bitmask_offset = BITS_TO_BYTES(vec_size);
    // Datatype align
    bitmask_offset = (bitmask_offset + sizeof(T) - 1) / sizeof(T) * sizeof(T);
    int bitshift = 0;
    for(int j = laneId; j < (vec_size + DWARP_SIZE - 1) / DWARP_SIZE * DWARP_SIZE; j += DWARP_SIZE) {
        int cur_bitshift = 0;
        T val = 0;
        if(j < vec_size) {
            T local_mask = mask[j];
            // Read element from vector
            val = src[j];
            // Init with bits in this element
            cur_bitshift = 8 * sizeof(T);
            // If this chunk is compressable, subtract the bits saved
            if((val & local_mask) == bitval[j]) {
                int count = 0;
                POPC(count, local_mask);
                cur_bitshift -= count;
            }
        }
        // Perform scan to obtain starting bit for this thread
        bitshift += warpExclusiveScanSync(FULL_MASK, cur_bitshift);
        //printf("BS %p %d: %d %d\n", dest, j, cur_bitshift, bitshift);
        if(j < vec_size) {
            T local_mask = mask[j];
            int count = 0;
            POPC(count, local_mask);
            int insert_size = 8 * sizeof(T) - count;
            T compressed_val = 0;
            // Start from bitshift and insert current compressed feature
            if((val & local_mask) == bitval[j]) {
                //uncomp = true;
                // Number of compressed bits being inserted
                int num_bits = insert_size;
                int temp_insert_size = CLZ(local_mask);
                int num_inserted = 0;
                while(num_bits > 0) {
                    num_inserted += temp_insert_size;
                    compressed_val |= (val & (T)(((1ll << temp_insert_size) - 1ll) << (sizeof(T) * 8 - num_inserted)));
                    //printf("IP %p %d: %x %x (%d, %d)\n", dest, j, compressed_val, val, temp_insert_size, num_inserted);
                    num_bits -= temp_insert_size;
                    // Shift by inserted bits
                    local_mask <<= temp_insert_size;
                    // Shift by masked bits
                    T shift = CLZ(~local_mask);
                    local_mask <<= shift;
                    val <<= shift;
                    // Get new insert size
                    temp_insert_size = min(CLZ(local_mask), (int)num_bits);
                }
                //printf("COMP %p %d: %x %x %x (%d)\n", dest, j, compressed_val, src[j], mask[j], insert_size);
                // Set bitmask bit
                T elem_offset = j / (sizeof(T) * 8);
                T bit_offset = j % (sizeof(T) * 8);
                atomicOr(dest + elem_offset, 1ll << bit_offset);
            } else {
                // Ignore mask and insert entire chunk
                insert_size = (sizeof(T) * 8);
                compressed_val = val;
                // Unset bitmask bit
                T elem_offset = j / (sizeof(T) * 8);
                T bit_offset =  j % (sizeof(T) * 8);
                atomicAnd(dest + elem_offset, ~(1ll << (bit_offset)));
                //printf("VAL %p %d: %x (%p)\n", dest, j, compressed_val, src + j);
            }
            T fin_ins_bits = 0;
            while(insert_size > 0) {
                // Now perform actual insertion
                T elem_offset = (bitmask_offset + bitshift / 8) / sizeof(T);
                T bit_offset = (bitmask_offset * 8 + bitshift) % (sizeof(T) * 8);
                T insert_bits = min(insert_size, (int)(sizeof(T) * 8 - bit_offset));
                T insert = ((compressed_val << fin_ins_bits) & (((1ll << insert_bits) - 1ll) << ((sizeof(T) * 8) - insert_bits))) >> bit_offset;

                //if(uncomp)
                //    printf("PREV %p %d (%d): %x\n", dest, j, elem_offset, atomicAdd(dest + elem_offset, 0));
                // Insert into cache
                atomicOr(dest + elem_offset, insert);
                //printf("%p %d: %x %x %x (%d, %d) (%d, %d)\n", dest, j, compressed_val, insert, 
                //    atomicAdd(dest + elem_offset, 0), bit_offset, elem_offset, insert_bits, fin_ins_bits);
                // Shift appropriately
                fin_ins_bits += insert_bits;
                insert_size -= insert_bits;
                bitshift += insert_bits;
            }
        }
        // Get starting bitshift from previous iteration
        bitshift = __shfl_sync(FULL_MASK, bitshift, DWARP_SIZE - 1);
    }
}
} // namespace ibp
#endif // IBP_COMPRESS_DEV