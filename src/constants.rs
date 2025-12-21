//! Internal constants and helper functions for nibblerun encoding.

/// Base epoch for timestamp compression (reduces storage by ~4 bytes)
pub const EPOCH_BASE: u64 = 1_760_000_000;

// Precomputed delta encoding table: (bits, num_bits) for deltas -10 to +10
//
// New encoding scheme (optimized based on real-world data analysis):
// - 0       = zero delta (1 bit)
// - 100     = +1 delta (3 bits)
// - 101     = -1 delta (3 bits)
// - 110     = single-interval gap (3 bits) - handled separately
// - 11100   = +2 delta (5 bits)
// - 11101   = -2 delta (5 bits)
// - 1111110xxxx = ±3-10 delta (11 bits)
// - 11111110xxxxxxxxxxx = large delta (19 bits)
// - 11111111xxxxxx = gap 2-65 intervals (14 bits)
#[allow(clippy::unusual_byte_groupings)]
pub const DELTA_ENCODE: [(u32, u8); 21] = [
    (0b1111110_0000, 11), // -10
    (0b1111110_0001, 11), // -9
    (0b1111110_0010, 11), // -8
    (0b1111110_0011, 11), // -7
    (0b1111110_0100, 11), // -6
    (0b1111110_0101, 11), // -5
    (0b1111110_0110, 11), // -4
    (0b1111110_0111, 11), // -3
    (0b11101, 5),         // -2 (1110 + sign=1)
    (0b101, 3),           // -1
    (0, 0),               // 0 (unused - handled by zero run)
    (0b100, 3),           // +1
    (0b11100, 5),         // +2 (1110 + sign=0)
    (0b1111110_1000, 11), // +3
    (0b1111110_1001, 11), // +4
    (0b1111110_1010, 11), // +5
    (0b1111110_1011, 11), // +6
    (0b1111110_1100, 11), // +7
    (0b1111110_1101, 11), // +8
    (0b1111110_1110, 11), // +9
    (0b1111110_1111, 11), // +10
];

// Branch hints using #[cold] attribute (stable Rust)
#[cold]
#[inline(never)]
pub const fn cold_gap_handler() {}

/// Division by interval
#[inline]
pub fn div_by_interval(x: u64, interval: u16) -> u64 {
    x / u64::from(interval)
}

/// Encode a zero run, returning (bits, `num_bits`, consumed)
///
/// Optimization: For runs of 1-7, individual zeros (1 bit each) are more efficient
/// than run-length encoding. Only use run encoding for n >= 8.
///
/// New encoding (shifted by 1 bit due to single-gap at 110):
/// - 1-7 zeros: individual 0 bits (1 bit each)
/// - 8-21 zeros: 11110xxxx (9 bits) - prefix 11110 + 4-bit length
/// - 22-149 zeros: 111110xxxxxxx (13 bits) - prefix 111110 + 7-bit length
/// - 150+ zeros: multiple 13-bit encodings
///
/// | Run length | Individual zeros | Run encoding | Winner |
/// |------------|------------------|--------------|--------|
/// | 1          | 1 bit            | -            | individual |
/// | 2-7        | 2-7 bits         | 9 bits       | individual |
/// | 8          | 8 bits           | 9 bits       | individual (but close) |
/// | 9-21       | 9-21 bits        | 9 bits       | run    |
/// | 22+        | 22+ bits         | 13 bits      | run    |
#[inline]
pub const fn encode_zero_run(n: u32) -> (u32, u32, u32) {
    if n <= 7 {
        // Individual zeros are more efficient for small runs
        (0, 1, 1)
    } else if n <= 21 {
        // 8-21 zeros: use the 9-bit encoding (prefix 11110 + 4-bit length)
        ((0b11110 << 4) | (n - 8), 9, n)
    } else if n <= 149 {
        // 22-149 zeros: use the 13-bit encoding (prefix 111110 + 7-bit length)
        ((0b11_1110 << 7) | (n - 22), 13, n)
    } else {
        // 150+ zeros: encode 149 at a time
        ((0b11_1110 << 7) | 127, 13, 149)
    }
}
