//! Appendable memory format for nibblerun time series compression.
//!
//! This module provides stateless functions that operate directly on byte buffers,
//! allowing compressed time series data to be stored as raw bytes and appended to
//! without reconstructing an Encoder struct.

use crate::constants::{cold_gap_handler, div_by_interval, encode_zero_run, DELTA_ENCODE, EPOCH_BASE};
use crate::error::AppendError;
use crate::reading::Reading;
use crate::value::Value;

/// Header size for i8 values (used as default for HEADER_SIZE constant)
pub const HEADER_SIZE: usize = header_size_for_value_bytes(1);

/// Compute header size based on value byte size
/// Header layout:
///   base_ts_offset: 4 bytes
///   count: 2 bytes
///   prev_logical_idx: 2 bytes
///   first_value: V::BYTES
///   prev_value: V::BYTES (delta reference - last encoded interval's value)
///   current_value: V::BYTES (current interval's value, may have keep-last updates)
///   zero_run: 1 byte
///   bit_count: 1 byte
///   bit_accum: 1 byte
pub const fn header_size_for_value_bytes(value_bytes: usize) -> usize {
    4 + 2 + 2 + value_bytes + value_bytes + value_bytes + 1 + 1 + 1
}

// Header field offsets (fixed portion)
const OFF_BASE_TS_OFFSET: usize = 0;
const OFF_COUNT: usize = 4;
const OFF_PREV_LOGICAL_IDX: usize = 6;
const OFF_FIRST_VALUE: usize = 8;

// Variable offsets depend on V::BYTES
#[inline]
const fn off_prev_value(value_bytes: usize) -> usize {
    8 + value_bytes
}

#[inline]
const fn off_current_value(value_bytes: usize) -> usize {
    8 + 2 * value_bytes
}

#[inline]
const fn off_zero_run(value_bytes: usize) -> usize {
    8 + 3 * value_bytes
}

#[inline]
const fn off_bit_count(value_bytes: usize) -> usize {
    8 + 3 * value_bytes + 1
}

#[inline]
const fn off_bit_accum(value_bytes: usize) -> usize {
    8 + 3 * value_bytes + 2
}

// ============================================================================
// BitSpan types for visualization
// ============================================================================

/// Information about a span of bits in the encoded data (for visualization)
#[derive(Debug, Clone)]
pub struct BitSpan {
    /// Unique identifier for this span
    pub id: usize,
    /// Starting bit position in the encoded data
    pub start_bit: usize,
    /// Number of bits in this span
    pub length: usize,
    /// What this span represents
    pub kind: BitSpanKind,
}

/// The kind of data a bit span represents
#[derive(Debug, Clone)]
pub enum BitSpanKind {
    /// Header: base timestamp (reconstructed)
    HeaderBaseTs(u64),
    /// Header: reading count
    HeaderCount(u16),
    /// Header: previous logical index
    HeaderPrevLogicalIdx(u16),
    /// Header: first value
    HeaderFirstValue(i32),
    /// Header: previous value (delta reference)
    HeaderPrevValue(i32),
    /// Header: current value (current interval)
    HeaderCurrentValue(i32),
    /// Header: zero run counter
    HeaderZeroRun(u8),
    /// Header: bit count
    HeaderBitCount(u8),
    /// Header: bit accumulator
    HeaderBitAccum(u8),
    /// Zero delta (unchanged value)
    Zero,
    /// Zero run 8-21 values
    ZeroRun8_21(u32),
    /// Zero run 22-149 values
    ZeroRun22_149(u32),
    /// Delta ±1
    Delta1(i32),
    /// Delta ±2
    Delta2(i32),
    /// Delta ±3 to ±10
    Delta3_10(i32),
    /// Large delta ±11 to ±1023
    LargeDelta(i32),
    /// Single-interval gap
    SingleGap,
    /// Multi-interval gap (2-65 intervals)
    Gap(u32),
}

// ============================================================================
// Header read/write helpers (inline for performance)
// ============================================================================

#[inline]
fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
}

#[inline]
fn write_u16_le(buf: &mut [u8], offset: usize, value: u16) {
    let bytes = value.to_le_bytes();
    buf[offset] = bytes[0];
    buf[offset + 1] = bytes[1];
}

#[inline]
fn write_u32_le(buf: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    buf[offset] = bytes[0];
    buf[offset + 1] = bytes[1];
    buf[offset + 2] = bytes[2];
    buf[offset + 3] = bytes[3];
}

// ============================================================================
// Bit writing helpers
// ============================================================================

/// Write bits to buffer, flushing complete bytes
#[inline]
fn write_bits(buf: &mut Vec<u8>, bit_accum: &mut u32, bit_count: &mut u8, value: u32, num_bits: u32) {
    *bit_accum = (*bit_accum << num_bits) | value;
    *bit_count += num_bits as u8;

    // Flush complete bytes
    while *bit_count >= 8 {
        *bit_count -= 8;
        buf.push((*bit_accum >> *bit_count) as u8);
    }
}

/// Flush pending zeros to buffer
#[inline]
fn flush_zeros(buf: &mut Vec<u8>, bit_accum: &mut u32, bit_count: &mut u8, zero_run: &mut u8) {
    while *zero_run > 0 {
        let (bits, num_bits, consumed) = encode_zero_run(u32::from(*zero_run));
        write_bits(buf, bit_accum, bit_count, bits, num_bits);
        *zero_run -= consumed as u8;
    }
}

/// Encode a delta value
#[inline]
fn encode_delta(buf: &mut Vec<u8>, bit_accum: &mut u32, bit_count: &mut u8, delta: i32) {
    let idx = (delta + 10) as usize;
    if idx <= 20 {
        let (bits, num_bits) = DELTA_ENCODE[idx];
        if num_bits > 0 {
            write_bits(buf, bit_accum, bit_count, bits, u32::from(num_bits));
            return;
        }
    }
    // Large delta: 11111110 (8-bit prefix) + 11-bit signed value = 19 bits
    let bits = (0b1111_1110_u32 << 11) | ((delta as u32) & 0x7FF);
    write_bits(buf, bit_accum, bit_count, bits, 19);
}

/// Write gap markers
#[inline]
fn write_gaps(buf: &mut Vec<u8>, bit_accum: &mut u32, bit_count: &mut u8, mut count: u32) {
    while count > 0 {
        if count == 1 {
            // Single-interval gap: 110 (3 bits)
            write_bits(buf, bit_accum, bit_count, 0b110, 3);
            count = 0;
        } else {
            // Multi-interval gap: 2-65 intervals
            let g = count.min(65);
            write_bits(buf, bit_accum, bit_count, (0xFF << 6) | (g - 2), 14);
            count -= g;
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Create a new appendable buffer with the first reading.
///
/// # Arguments
/// * `ts` - Unix timestamp in seconds
/// * `value` - First sensor value
///
/// # Returns
/// A new `Vec<u8>` containing the initialized buffer with header and no data yet.
#[must_use]
pub fn create<V: Value, const INTERVAL: u16>(ts: u64, value: V) -> Vec<u8> {
    let header_size = header_size_for_value_bytes(V::BYTES);
    let mut buf = vec![0u8; header_size];

    // Write header
    let base_ts_offset = ts.wrapping_sub(EPOCH_BASE) as u32;
    write_u32_le(&mut buf, OFF_BASE_TS_OFFSET, base_ts_offset);
    write_u16_le(&mut buf, OFF_COUNT, 1);
    write_u16_le(&mut buf, OFF_PREV_LOGICAL_IDX, 0);

    let val_i32 = value.to_i32();
    V::from_i32(val_i32).write_le(&mut buf[OFF_FIRST_VALUE..]);
    V::from_i32(val_i32).write_le(&mut buf[off_prev_value(V::BYTES)..]);
    V::from_i32(val_i32).write_le(&mut buf[off_current_value(V::BYTES)..]);

    // Initialize zero_run, bit_count, bit_accum to 0
    buf[off_zero_run(V::BYTES)] = 0;
    buf[off_bit_count(V::BYTES)] = 0;
    buf[off_bit_accum(V::BYTES)] = 0;

    buf
}

/// Append a reading to an existing buffer.
///
/// # Arguments
/// * `buf` - Mutable reference to the buffer
/// * `ts` - Unix timestamp in seconds
/// * `value` - Sensor value
///
/// # Errors
/// Returns an error if:
/// - Buffer is too short or has wrong format
/// - Timestamp is before the base timestamp
/// - Timestamp is out of order
/// - Too many total readings
/// - Value delta exceeds encodable range
///
/// # Same-interval behavior
/// If multiple readings arrive in the same interval, only the last value is kept.
#[inline]
pub fn append<V: Value, const INTERVAL: u16>(
    buf: &mut Vec<u8>,
    ts: u64,
    value: V,
) -> Result<(), AppendError> {
    let header_size = header_size_for_value_bytes(V::BYTES);
    if buf.len() < header_size {
        return Err(AppendError::CountOverflow); // TODO: better error
    }

    let value_i32 = value.to_i32();

    // Read state from header
    let base_ts_offset = read_u32_le(buf, OFF_BASE_TS_OFFSET);
    let base_ts = EPOCH_BASE + u64::from(base_ts_offset);
    let count = read_u16_le(buf, OFF_COUNT);
    let prev_logical_idx = read_u16_le(buf, OFF_PREV_LOGICAL_IDX);
    let prev_value = V::read_le(&buf[off_prev_value(V::BYTES)..]).to_i32();
    let current_value = V::read_le(&buf[off_current_value(V::BYTES)..]).to_i32();
    let mut zero_run = buf[off_zero_run(V::BYTES)];
    let mut bit_accum = u32::from(buf[off_bit_accum(V::BYTES)]);
    let mut bit_count = buf[off_bit_count(V::BYTES)];

    // Reject out-of-order readings (ts before base_ts)
    if ts < base_ts {
        return Err(AppendError::TimestampBeforeBase { ts, base_ts });
    }

    // Calculate logical index
    let logical_idx_u32 = div_by_interval(ts - base_ts, INTERVAL) as u32;

    // Check for time span overflow
    if logical_idx_u32 > u32::from(u16::MAX) {
        return Err(AppendError::TimeSpanOverflow {
            ts,
            base_ts,
            max_intervals: u32::from(u16::MAX),
        });
    }
    let logical_idx = logical_idx_u32 as u16;

    // Reject readings that go backwards in time
    if logical_idx < prev_logical_idx {
        return Err(AppendError::OutOfOrder {
            ts,
            logical_idx: u32::from(logical_idx),
            prev_logical_idx: u32::from(prev_logical_idx),
        });
    }

    // Same interval - "keep last" semantics: just update current_value
    if logical_idx == prev_logical_idx {
        V::from_i32(value_i32).write_le(&mut buf[off_current_value(V::BYTES)..]);
        return Ok(());
    }

    // New interval - check for potential errors before committing

    // Check count overflow
    if count == u16::MAX {
        return Err(AppendError::CountOverflow);
    }

    // Check delta overflow (only for count > 1, since first interval has no delta)
    if count > 1 {
        let delta = current_value - prev_value;
        if !(-1024..=1023).contains(&delta) {
            return Err(AppendError::DeltaOverflow {
                delta,
                prev_value,
                new_value: current_value,
            });
        }
    }

    let index_gap = u32::from(logical_idx) - u32::from(prev_logical_idx);

    // Finalize current interval: encode delta if count > 1
    if count == 1 {
        // First interval is finalized - update first_value with keep-last result
        V::from_i32(current_value).write_le(&mut buf[OFF_FIRST_VALUE..]);
    } else {
        // Encode delta from prev_value to current_value
        let delta = current_value - prev_value;
        if delta == 0 {
            zero_run = zero_run.saturating_add(1);
            // Flush if zero_run reaches 149 (max for efficient encoding)
            if zero_run >= 149 {
                flush_zeros(buf, &mut bit_accum, &mut bit_count, &mut zero_run);
            }
        } else {
            flush_zeros(buf, &mut bit_accum, &mut bit_count, &mut zero_run);
            encode_delta(buf, &mut bit_accum, &mut bit_count, delta);
        }
    }

    // Gap handling (rare)
    if index_gap > 1 {
        cold_gap_handler();
        flush_zeros(buf, &mut bit_accum, &mut bit_count, &mut zero_run);
        write_gaps(buf, &mut bit_accum, &mut bit_count, index_gap - 1);
    }

    // Update header for new interval
    write_u16_le(buf, OFF_COUNT, count + 1);
    write_u16_le(buf, OFF_PREV_LOGICAL_IDX, logical_idx);
    V::from_i32(current_value).write_le(&mut buf[off_prev_value(V::BYTES)..]);
    V::from_i32(value_i32).write_le(&mut buf[off_current_value(V::BYTES)..]);
    buf[off_zero_run(V::BYTES)] = zero_run;
    buf[off_bit_count(V::BYTES)] = bit_count;
    buf[off_bit_accum(V::BYTES)] = bit_accum as u8;

    Ok(())
}

/// Decode buffer to readings.
///
/// # Arguments
/// * `buf` - Buffer to decode
///
/// # Returns
/// Vector of decoded readings. Returns empty vector if buffer is invalid.
#[must_use]
pub fn decode<V: Value, const INTERVAL: u16>(buf: &[u8]) -> Vec<Reading<V>> {
    let header_size = header_size_for_value_bytes(V::BYTES);
    if buf.len() < header_size {
        return Vec::new();
    }

    // Read header
    let base_ts_offset = read_u32_le(buf, OFF_BASE_TS_OFFSET);
    let base_ts = EPOCH_BASE.wrapping_add(u64::from(base_ts_offset));
    let count = read_u16_le(buf, OFF_COUNT) as usize;
    let first_value = V::read_le(&buf[OFF_FIRST_VALUE..]).to_i32();
    let prev_value = V::read_le(&buf[off_prev_value(V::BYTES)..]).to_i32();
    let current_value = V::read_le(&buf[off_current_value(V::BYTES)..]).to_i32();
    let zero_run = buf[off_zero_run(V::BYTES)];
    let bit_count = buf[off_bit_count(V::BYTES)];
    let bit_accum = u32::from(buf[off_bit_accum(V::BYTES)]);

    if count == 0 {
        return Vec::new();
    }

    // For single interval, use current_value (which has keep-last applied)
    // For multiple intervals, first_value was finalized when we moved to interval 2
    let first_val = if count == 1 {
        V::from_i32(current_value)
    } else {
        V::from_i32(first_value)
    };

    let mut decoded = Vec::with_capacity(count);
    decoded.push(Reading {
        ts: base_ts,
        value: first_val,
    });

    if count == 1 {
        return decoded;
    }

    // Build finalized bit data by encoding the pending current_value delta
    let data = &buf[header_size..];
    let mut final_data = data.to_vec();
    let mut accum = bit_accum;
    // Sanitize bit_count to valid range [0, 7] since we only store partial bytes
    let mut bits = u32::from(bit_count) & 7;
    let mut run_zeros = u32::from(zero_run);

    // Helper to flush complete bytes from accumulator
    let flush_bytes = |accum: &mut u32, bits: &mut u32, out: &mut Vec<u8>| {
        while *bits >= 8 {
            *bits -= 8;
            out.push((*accum >> *bits) as u8);
        }
    };

    // Encode the final interval's delta (current_value - prev_value)
    let delta = current_value.wrapping_sub(prev_value);
    if delta == 0 {
        run_zeros = run_zeros.saturating_add(1);
    } else {
        // First flush pending zeros (with overflow protection)
        let max_iterations = 1000u32;
        let mut iterations = 0u32;
        while run_zeros > 0 && iterations < max_iterations {
            let (b, n, c) = encode_zero_run(run_zeros);
            accum = (accum << n) | b;
            bits = bits.saturating_add(n);
            run_zeros = run_zeros.saturating_sub(c);
            flush_bytes(&mut accum, &mut bits, &mut final_data);
            iterations += 1;
        }
        // Encode the delta
        let (delta_bits, delta_num_bits) = encode_delta_value(delta);
        accum = (accum << delta_num_bits) | delta_bits;
        bits = bits.saturating_add(delta_num_bits);
        flush_bytes(&mut accum, &mut bits, &mut final_data);
    }

    // Flush remaining zeros (with overflow protection)
    let max_iterations = 1000u32;
    let mut iterations = 0u32;
    while run_zeros > 0 && iterations < max_iterations {
        let (b, n, c) = encode_zero_run(run_zeros);
        accum = (accum << n) | b;
        bits = bits.saturating_add(n);
        run_zeros = run_zeros.saturating_sub(c);
        flush_bytes(&mut accum, &mut bits, &mut final_data);
        iterations += 1;
    }

    // Flush any remaining complete bytes
    flush_bytes(&mut accum, &mut bits, &mut final_data);

    if bits > 0 {
        final_data.push((accum << (8 - bits)) as u8);
    }

    // Now decode using BitReader
    let mut reader = BitReader::new(&final_data);
    let mut prev_val = first_value;
    let mut idx = 1u64;
    let interval = u64::from(INTERVAL);

    while decoded.len() < count && reader.has_more() {
        if reader.read_bits(1) == 0 {
            // 0 = zero delta
            decoded.push(Reading {
                ts: base_ts.wrapping_add(idx.wrapping_mul(interval)),
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }
        if reader.read_bits(1) == 0 {
            // 10x = ±1 delta
            prev_val = prev_val.wrapping_add(if reader.read_bits(1) == 0 { 1 } else { -1 });
            decoded.push(Reading {
                ts: base_ts.wrapping_add(idx.wrapping_mul(interval)),
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }
        // 11...
        if reader.read_bits(1) == 0 {
            // 110 = single-interval gap
            idx += 1;
            continue;
        }
        // 111...
        if reader.read_bits(1) == 0 {
            // 1110x = ±2 delta
            prev_val = prev_val.wrapping_add(if reader.read_bits(1) == 0 { 2 } else { -2 });
            decoded.push(Reading {
                ts: base_ts.wrapping_add(idx.wrapping_mul(interval)),
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }
        // 1111...
        if reader.read_bits(1) == 0 {
            // 11110xxxx = zero run 8-21
            for _ in 0..reader.read_bits(4) + 8 {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: base_ts.wrapping_add(idx.wrapping_mul(interval)),
                    value: V::from_i32(prev_val),
                });
                idx += 1;
            }
            continue;
        }
        // 11111...
        if reader.read_bits(1) == 0 {
            // 111110xxxxxxx = zero run 22-149
            for _ in 0..reader.read_bits(7) + 22 {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: base_ts.wrapping_add(idx.wrapping_mul(interval)),
                    value: V::from_i32(prev_val),
                });
                idx += 1;
            }
            continue;
        }
        // 111111...
        if reader.read_bits(1) == 0 {
            // 1111110xxxx = ±3-10 delta
            let e = reader.read_bits(4) as i32;
            prev_val = prev_val.wrapping_add(if e < 8 { e - 10 } else { e - 5 });
            decoded.push(Reading {
                ts: base_ts.wrapping_add(idx.wrapping_mul(interval)),
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }
        // 1111111...
        if reader.read_bits(1) == 0 {
            // 11111110xxxxxxxxxxx = large delta
            let raw = reader.read_bits(11);
            prev_val = prev_val.wrapping_add(if raw & 0x400 != 0 {
                (raw | 0xFFFF_F800) as i32
            } else {
                raw as i32
            });
            decoded.push(Reading {
                ts: base_ts.wrapping_add(idx.wrapping_mul(interval)),
                value: V::from_i32(prev_val),
            });
            idx += 1;
        } else {
            // 11111111xxxxxx = gap 2-65 intervals
            idx = idx.wrapping_add(u64::from(reader.read_bits(6) + 2));
        }
    }

    decoded
}

/// Encode a delta value, returning (bits, num_bits)
#[inline]
fn encode_delta_value(delta: i32) -> (u32, u32) {
    // Use wrapping add to handle extreme values gracefully
    let idx = delta.wrapping_add(10) as usize;
    if idx <= 20 {
        let (bits, num_bits) = DELTA_ENCODE[idx];
        if num_bits > 0 {
            return (bits, u32::from(num_bits));
        }
    }
    // Large delta
    let clamped = delta.clamp(-1024, 1023);
    let bits = (0b1111_1110_u32 << 11) | ((clamped as u32) & 0x7FF);
    (bits, 19)
}

/// Get the count of readings in the buffer.
#[must_use]
pub fn count(buf: &[u8]) -> Option<u16> {
    if buf.len() < HEADER_SIZE {
        return None;
    }
    Some(read_u16_le(buf, OFF_COUNT))
}

/// Check if a buffer is empty (no readings).
#[must_use]
pub fn is_empty(buf: &[u8]) -> bool {
    count(buf).is_none_or(|c| c == 0)
}

/// Decode buffer to readings with bit span information for visualization.
///
/// Returns both the decoded readings and information about what each bit span
/// in the encoded data represents.
///
/// # Arguments
/// * `buf` - Buffer to decode
///
/// # Returns
/// Tuple of (readings, header_spans, data_spans). Returns empty vectors if buffer is invalid.
#[must_use]
pub fn decode_with_spans<V: Value, const INTERVAL: u16>(
    buf: &[u8],
) -> (Vec<Reading<V>>, Vec<BitSpan>, Vec<BitSpan>) {
    let header_size = header_size_for_value_bytes(V::BYTES);
    let value_bits = V::BYTES * 8;
    if buf.len() < header_size {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut header_spans = Vec::new();
    let mut data_spans = Vec::new();
    let mut span_id = 0;

    // Parse header and create header spans
    // Header is byte-aligned, so we track bit positions as byte_offset * 8

    // Base timestamp offset (4 bytes = 32 bits)
    let base_ts_offset = read_u32_le(buf, OFF_BASE_TS_OFFSET);
    let base_ts = EPOCH_BASE.wrapping_add(u64::from(base_ts_offset));
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_BASE_TS_OFFSET * 8,
        length: 32,
        kind: BitSpanKind::HeaderBaseTs(base_ts),
    });
    span_id += 1;

    // Count (2 bytes = 16 bits)
    let count = read_u16_le(buf, OFF_COUNT) as usize;
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_COUNT * 8,
        length: 16,
        kind: BitSpanKind::HeaderCount(count as u16),
    });
    span_id += 1;

    // Previous logical index (2 bytes = 16 bits)
    let prev_logical_idx = read_u16_le(buf, OFF_PREV_LOGICAL_IDX);
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_PREV_LOGICAL_IDX * 8,
        length: 16,
        kind: BitSpanKind::HeaderPrevLogicalIdx(prev_logical_idx),
    });
    span_id += 1;

    // First value (V::BYTES)
    let first_value = V::read_le(&buf[OFF_FIRST_VALUE..]).to_i32();
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_FIRST_VALUE * 8,
        length: value_bits,
        kind: BitSpanKind::HeaderFirstValue(first_value),
    });
    span_id += 1;

    // Previous value (V::BYTES)
    let prev_value = V::read_le(&buf[off_prev_value(V::BYTES)..]).to_i32();
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_prev_value(V::BYTES) * 8,
        length: value_bits,
        kind: BitSpanKind::HeaderPrevValue(prev_value),
    });
    span_id += 1;

    // Current value (V::BYTES)
    let current_value = V::read_le(&buf[off_current_value(V::BYTES)..]).to_i32();
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_current_value(V::BYTES) * 8,
        length: value_bits,
        kind: BitSpanKind::HeaderCurrentValue(current_value),
    });
    span_id += 1;

    // Zero run (1 byte = 8 bits)
    let zero_run = buf[off_zero_run(V::BYTES)];
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_zero_run(V::BYTES) * 8,
        length: 8,
        kind: BitSpanKind::HeaderZeroRun(zero_run),
    });
    span_id += 1;

    // Bit count (1 byte = 8 bits)
    let bit_count = buf[off_bit_count(V::BYTES)];
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_bit_count(V::BYTES) * 8,
        length: 8,
        kind: BitSpanKind::HeaderBitCount(bit_count),
    });
    span_id += 1;

    // Bit accumulator (1 byte = 8 bits)
    let bit_accum = u32::from(buf[off_bit_accum(V::BYTES)]);
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_bit_accum(V::BYTES) * 8,
        length: 8,
        kind: BitSpanKind::HeaderBitAccum(bit_accum as u8),
    });
    span_id += 1;

    if count == 0 {
        return (Vec::new(), header_spans, data_spans);
    }

    // For single interval, use current_value (which has keep-last applied)
    // For multiple intervals, first_value was finalized when we moved to interval 2
    let first_val = if count == 1 {
        V::from_i32(current_value)
    } else {
        V::from_i32(first_value)
    };

    let mut decoded = Vec::with_capacity(count);
    decoded.push(Reading {
        ts: base_ts,
        value: first_val,
    });

    if count == 1 {
        return (decoded, header_spans, data_spans);
    }

    // Build finalized bit data (same as decode())
    let data = &buf[header_size..];
    let mut final_data = data.to_vec();
    let mut accum = bit_accum;
    let mut bits = u32::from(bit_count) & 7;
    let mut run_zeros = u32::from(zero_run);

    let flush_bytes = |accum: &mut u32, bits: &mut u32, out: &mut Vec<u8>| {
        while *bits >= 8 {
            *bits -= 8;
            out.push((*accum >> *bits) as u8);
        }
    };

    // Encode final delta (current_value - prev_value)
    let delta = current_value.wrapping_sub(prev_value);
    if delta == 0 {
        run_zeros = run_zeros.saturating_add(1);
    } else {
        let max_iterations = 1000u32;
        let mut iterations = 0u32;
        while run_zeros > 0 && iterations < max_iterations {
            let (b, n, c) = encode_zero_run(run_zeros);
            accum = (accum << n) | b;
            bits = bits.saturating_add(n);
            run_zeros = run_zeros.saturating_sub(c);
            flush_bytes(&mut accum, &mut bits, &mut final_data);
            iterations += 1;
        }
        let (delta_bits, delta_num_bits) = encode_delta_value(delta);
        accum = (accum << delta_num_bits) | delta_bits;
        bits = bits.saturating_add(delta_num_bits);
        flush_bytes(&mut accum, &mut bits, &mut final_data);
    }

    let max_iterations = 1000u32;
    let mut iterations = 0u32;
    while run_zeros > 0 && iterations < max_iterations {
        let (b, n, c) = encode_zero_run(run_zeros);
        accum = (accum << n) | b;
        bits = bits.saturating_add(n);
        run_zeros = run_zeros.saturating_sub(c);
        flush_bytes(&mut accum, &mut bits, &mut final_data);
        iterations += 1;
    }

    flush_bytes(&mut accum, &mut bits, &mut final_data);
    if bits > 0 {
        final_data.push((accum << (8 - bits)) as u8);
    }

    // Now decode with span tracking
    let mut reader = BitReaderWithPos::new(&final_data);
    let mut prev_val = first_value;
    let mut idx = 1u64;
    let interval = u64::from(INTERVAL);
    let header_bits = header_size * 8;

    while decoded.len() < count && reader.has_more() {
        let start_bit = reader.bit_pos();

        if reader.read_bits(1) == 0 {
            // 0 = zero delta
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 1,
                kind: BitSpanKind::Zero,
            });
            span_id += 1;
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        if reader.read_bits(1) == 0 {
            // 10x = ±1 delta
            let sign = reader.read_bits(1);
            let delta = if sign == 0 { 1 } else { -1 };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 3,
                kind: BitSpanKind::Delta1(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        // 11...
        if reader.read_bits(1) == 0 {
            // 110 = single-interval gap
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 3,
                kind: BitSpanKind::SingleGap,
            });
            span_id += 1;
            idx += 1;
            continue;
        }

        // 111...
        if reader.read_bits(1) == 0 {
            // 1110x = ±2 delta
            let sign = reader.read_bits(1);
            let delta = if sign == 0 { 2 } else { -2 };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 5,
                kind: BitSpanKind::Delta2(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        // 1111...
        if reader.read_bits(1) == 0 {
            // 11110xxxx = zero run 8-21
            let n = reader.read_bits(4) + 8;
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 9,
                kind: BitSpanKind::ZeroRun8_21(n),
            });
            span_id += 1;
            for _ in 0..n {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: base_ts + idx * interval,
                    value: V::from_i32(prev_val),
                });
                idx += 1;
            }
            continue;
        }

        // 11111...
        if reader.read_bits(1) == 0 {
            // 111110xxxxxxx = zero run 22-149
            let n = reader.read_bits(7) + 22;
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 13,
                kind: BitSpanKind::ZeroRun22_149(n),
            });
            span_id += 1;
            for _ in 0..n {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: base_ts + idx * interval,
                    value: V::from_i32(prev_val),
                });
                idx += 1;
            }
            continue;
        }

        // 111111...
        if reader.read_bits(1) == 0 {
            // 1111110xxxx = ±3-10 delta
            let e = reader.read_bits(4) as i32;
            let delta = if e < 8 { e - 10 } else { e - 5 };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 11,
                kind: BitSpanKind::Delta3_10(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        // 1111111...
        if reader.read_bits(1) == 0 {
            // 11111110xxxxxxxxxxx = large delta
            let raw = reader.read_bits(11);
            let delta = if raw & 0x400 != 0 {
                (raw | 0xFFFF_F800) as i32
            } else {
                raw as i32
            };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 19,
                kind: BitSpanKind::LargeDelta(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
        } else {
            // 11111111xxxxxx = gap 2-65 intervals
            let n = reader.read_bits(6) + 2;
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 14,
                kind: BitSpanKind::Gap(n),
            });
            span_id += 1;
            idx += u64::from(n);
        }
    }

    (decoded, header_spans, data_spans)
}

// ============================================================================
// BitReader (internal)
// ============================================================================

/// BitReader with position tracking for span calculation
struct BitReaderWithPos<'a> {
    buf: &'a [u8],
    byte_pos: usize,
    bit_offset: usize, // 0-7, bits consumed from current logical position
    bits: u64,
    left: u32,
}

impl<'a> BitReaderWithPos<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut r = BitReaderWithPos {
            buf,
            byte_pos: 0,
            bit_offset: 0,
            bits: 0,
            left: 0,
        };
        r.refill();
        r
    }

    fn refill(&mut self) {
        while self.left <= 56 && self.byte_pos < self.buf.len() {
            self.bits = (self.bits << 8) | u64::from(self.buf[self.byte_pos]);
            self.byte_pos += 1;
            self.left += 8;
        }
    }

    fn read_bits(&mut self, n: u32) -> u32 {
        if self.left < n {
            self.refill();
        }
        if self.left < n {
            return 0;
        }
        self.left -= n;
        self.bit_offset += n as usize;
        ((self.bits >> self.left) & ((1 << n) - 1)) as u32
    }

    /// Get current bit position in the stream
    fn bit_pos(&self) -> usize {
        // byte_pos is how many bytes we've loaded into the buffer
        // left is how many bits remain in the buffer
        // So actual position = (byte_pos * 8) - left
        (self.byte_pos * 8).saturating_sub(self.left as usize)
    }

    const fn has_more(&self) -> bool {
        self.left > 0 || self.byte_pos < self.buf.len()
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    bits: u64,
    left: u32,
}

impl<'a> BitReader<'a> {
    #[inline]
    fn new(buf: &'a [u8]) -> Self {
        let mut r = BitReader {
            buf,
            pos: 0,
            bits: 0,
            left: 0,
        };
        r.refill();
        r
    }

    #[inline]
    fn refill(&mut self) {
        while self.left <= 56 && self.pos < self.buf.len() {
            self.bits = (self.bits << 8) | u64::from(self.buf[self.pos]);
            self.pos += 1;
            self.left += 8;
        }
    }

    #[inline]
    fn read_bits(&mut self, n: u32) -> u32 {
        if self.left < n {
            self.refill();
        }
        if self.left < n {
            return 0;
        }
        self.left -= n;
        ((self.bits >> self.left) & ((1 << n) - 1)) as u32
    }

    #[inline]
    const fn has_more(&self) -> bool {
        self.left > 0 || self.pos < self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_decode_single() {
        let buf = create::<i32, 300>(1_761_000_000, 22);
        assert_eq!(buf.len(), header_size_for_value_bytes(i32::BYTES));

        let readings = decode::<i32, 300>(&buf);
        assert_eq!(readings.len(), 1);
        assert_eq!(readings[0].ts, 1_761_000_000);
        assert_eq!(readings[0].value, 22);
    }

    #[test]
    fn test_append_same_interval() {
        let mut buf = create::<i32, 300>(1_761_000_000, 20);
        // Append in same interval - keep last semantics
        append::<i32, 300>(&mut buf, 1_761_000_100, 24).unwrap();

        let readings = decode::<i32, 300>(&buf);
        assert_eq!(readings.len(), 1);
        assert_eq!(readings[0].value, 24); // keep last: 24
    }

    #[test]
    fn test_append_new_interval() {
        let mut buf = create::<i32, 300>(1_761_000_000, 22);
        append::<i32, 300>(&mut buf, 1_761_000_300, 22).unwrap(); // Same value
        append::<i32, 300>(&mut buf, 1_761_000_600, 23).unwrap(); // +1

        let readings = decode::<i32, 300>(&buf);
        assert_eq!(readings.len(), 3);
        assert_eq!(readings[0].value, 22);
        assert_eq!(readings[1].value, 22);
        assert_eq!(readings[2].value, 23);
    }

    #[test]
    fn test_roundtrip_many() {
        let base = 1_761_000_000u64;
        let mut buf = create::<i32, 300>(base, 20);

        for i in 1..100 {
            let ts = base + i * 300;
            let value = 20 + ((i % 5) as i32);
            append::<i32, 300>(&mut buf, ts, value).unwrap();
        }

        let readings = decode::<i32, 300>(&buf);
        assert_eq!(readings.len(), 100);

        // Verify first and pattern
        assert_eq!(readings[0].value, 20);
        for i in 1..100 {
            assert_eq!(readings[i].ts, base + i as u64 * 300);
            assert_eq!(readings[i].value, 20 + ((i % 5) as i32));
        }
    }
}
