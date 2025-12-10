//! Internal constants and helper functions for nibblerun encoding.

/// Base epoch for timestamp compression (reduces storage by ~4 bytes)
pub(crate) const EPOCH_BASE: u64 = 1_760_000_000;

/// Header size in bytes (4 + 2 + 2 + 4 + 2 = 14)
pub(crate) const HEADER_SIZE: usize = 14;

// Precomputed delta encoding table: (bits, num_bits) for deltas -10 to +10
#[allow(clippy::unusual_byte_groupings)]
pub(crate) const DELTA_ENCODE: [(u32, u8); 21] = [
    (0b1111110_0000, 11), // -10
    (0b1111110_0001, 11), // -9
    (0b1111110_0010, 11), // -8
    (0b1111110_0011, 11), // -7
    (0b1111110_0100, 11), // -6
    (0b1111110_0101, 11), // -5
    (0b1111110_0110, 11), // -4
    (0b1111110_0111, 11), // -3
    (0b111_1101, 7),      // -2
    (0b101, 3),           // -1
    (0, 0),               // 0 (unused - handled by zero run)
    (0b100, 3),           // +1
    (0b111_1100, 7),      // +2
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
pub(crate) fn cold_gap_handler() {}

/// Division by interval
#[inline]
pub(crate) fn div_by_interval(x: u64, interval: u16) -> u64 {
    x / u64::from(interval)
}

/// Compute average with proper rounding (round half away from zero)
/// This ensures the average is always within [min, max] of the input values
#[inline]
pub(crate) fn rounded_avg(sum: i32, count: u16) -> i32 {
    if count <= 1 {
        return sum;
    }
    let c = i32::from(count);
    if sum >= 0 {
        (sum + c / 2) / c
    } else {
        (sum - c / 2) / c
    }
}

/// Pack pending averaging state into `pending_state` (u64)
/// - Bits 0-5: actual bit accumulator count (0-63)
/// - Bits 6-15: `pending_count` (0-1023)
/// - Bits 16-47: `pending_sum` as i32 (32 bits, stored as u32)
#[inline]
pub(crate) fn pack_pending(bits: u32, count: u16, sum: i32) -> u64 {
    (u64::from(sum as u32) << 16) | ((u64::from(count) & 0x3FF) << 6) | (u64::from(bits) & 0x3F)
}

/// Unpack pending averaging state from `pending_state`
/// Returns (`bit_accum_count`, `pending_count`, `pending_sum`)
#[inline]
pub(crate) fn unpack_pending(packed: u64) -> (u32, u16, i32) {
    let bits = (packed & 0x3F) as u32;
    let count = ((packed >> 6) & 0x3FF) as u16;
    let sum = (packed >> 16) as u32 as i32;
    (bits, count, sum)
}

/// Encode a zero run, returning (bits, `num_bits`, consumed)
#[inline]
pub(crate) fn encode_zero_run(n: u32) -> (u32, u32, u32) {
    if n == 1 {
        (0, 1, 1)
    } else if n <= 5 {
        ((0b110 << 2) | (n - 2), 5, n)
    } else if n <= 21 {
        ((0b1110 << 4) | (n - 6), 8, n)
    } else if n <= 149 {
        ((0b11110 << 7) | (n - 22), 12, n)
    } else {
        ((0b11110 << 7) | 127, 12, 149)
    }
}
