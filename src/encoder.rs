//! Encoder for nibblerun time series compression.

use serde::{Deserialize, Serialize};

use crate::constants::{div_by_interval, encode_zero_run, DELTA_ENCODE, MAX_GAP_PER_MARKER, MAX_ZERO_RUN_TIER2};
use crate::decoder::{decode_bitstream, BitReader};
use crate::error::{AppendError, DecodeError};
use crate::reading::Reading;
use crate::value::Value;

/// Encoder for `NibbleRun` format
///
/// Accumulates sensor readings and produces compressed output.
/// Generic over value type V (i8, i16, or i32) and interval INTERVAL (compile-time constant).
#[derive(Clone, Serialize, Deserialize)]
pub struct Encoder<V: Value, const INTERVAL: u16 = 300> {
    base_ts: u32,
    first_value: V,
    count: u16,
    prev_logical_idx: u16,
    prev_value: V,
    current_value: V,
    zero_run: u8,
    bit_count: u8,
    bit_accum: u8,
    data: Vec<u8>,
}

impl<V: Value, const INTERVAL: u16> Encoder<V, INTERVAL> {
    /// Create a new encoder
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_ts: 0,
            first_value: V::default(),
            count: 0,
            prev_logical_idx: 0,
            prev_value: V::default(),
            current_value: V::default(),
            zero_run: 0,
            bit_count: 0,
            bit_accum: 0,
            data: Vec::new(),
        }
    }

    /// Get the interval in seconds
    #[inline]
    #[must_use]
    pub const fn interval() -> u16 {
        INTERVAL
    }

    /// Header size for frozen format
    #[inline]
    #[must_use]
    pub const fn header_size() -> usize {
        4 + 2 + V::BYTES
    }

    /// Append a sensor reading
    #[inline]
    pub fn append(&mut self, ts: u32, value: V) -> Result<(), AppendError<V>> {
        if self.count == 0 {
            self.base_ts = ts;
            self.first_value = value;
            self.count = 1;
            self.prev_logical_idx = 0;
            self.prev_value = value;
            self.current_value = value;
            return Ok(());
        }

        if ts < self.base_ts {
            return Err(AppendError::TimestampBeforeBase { ts, base_ts: self.base_ts });
        }

        let logical_idx = div_by_interval(u64::from(ts - self.base_ts), INTERVAL) as u32;

        if logical_idx > u32::from(u16::MAX) {
            return Err(AppendError::TimeSpanOverflow {
                ts,
                base_ts: self.base_ts,
                max_intervals: u32::from(u16::MAX),
            });
        }
        let logical_idx = logical_idx as u16;

        if logical_idx < self.prev_logical_idx {
            return Err(AppendError::OutOfOrder {
                ts,
                logical_idx: u32::from(logical_idx),
                prev_logical_idx: u32::from(self.prev_logical_idx),
            });
        }

        // Same interval - keep last
        if logical_idx == self.prev_logical_idx {
            self.current_value = value;
            return Ok(());
        }

        if self.count == u16::MAX {
            return Err(AppendError::CountOverflow);
        }

        // Early check: validate the delta for THIS new value (fail fast)
        let new_delta = value.to_i32() - self.current_value.to_i32();
        if !(-1024..=1023).contains(&new_delta) {
            return Err(AppendError::DeltaOverflow {
                delta: new_delta,
                current_value: self.current_value,
                new_value: value,
            });
        }

        // Finalize previous interval
        if self.count == 1 {
            self.first_value = self.current_value;
        } else {
            let delta = self.current_value.to_i32() - self.prev_value.to_i32();
            // Note: delta check already passed when this value was appended
            if delta == 0 {
                self.zero_run = self.zero_run.saturating_add(1);
                if u32::from(self.zero_run) >= MAX_ZERO_RUN_TIER2 {
                    self.flush_zeros();
                }
            } else {
                self.flush_zeros();
                self.encode_delta(delta);
            }
        }

        // Handle gaps
        let gap = logical_idx - self.prev_logical_idx;
        if gap > 1 {
            self.flush_zeros();
            self.write_gaps(u32::from(gap - 1));
        }

        // Update state
        self.count += 1;
        self.prev_logical_idx = logical_idx;
        self.prev_value = self.current_value;
        self.current_value = value;

        Ok(())
    }

    /// Get the current size in bytes of the appendable format
    ///
    /// This is the size of `to_bytes()` output, which includes header and data.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        if self.count == 0 { 0 } else { Self::wire_header_size() + self.data.len() }
    }

    /// Get the number of readings stored
    #[inline]
    #[must_use]
    pub fn count(&self) -> usize {
        self.count as usize
    }

    /// Check if the encoder contains no readings
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Decode all readings from this encoder
    ///
    /// Returns all readings that have been appended to this encoder.
    /// This includes finalizing any pending state (zero runs, partial bytes).
    ///
    /// # Errors
    /// Returns `DecodeError::MalformedData` if the internal data is corrupted
    /// (should not happen with properly constructed encoders).
    #[must_use = "decoding returns readings that should be used"]
    pub fn decode(&self) -> Result<Vec<Reading<V>>, DecodeError> {
        if self.count == 0 {
            return Ok(Vec::new());
        }

        let base_ts = self.base_ts;
        let interval = u32::from(INTERVAL);
        let count = self.count as usize;

        let first_val = if count == 1 { self.current_value } else { self.first_value };

        let mut result = Vec::with_capacity(count);
        result.push(Reading { ts: base_ts, value: first_val });

        if count == 1 {
            return Ok(result);
        }

        // Finalize bit data
        let mut final_data = self.data.clone();
        let mut accum = u32::from(self.bit_accum);
        // Sanitize bit_count - valid range is 0-7 (partial byte)
        let mut bits = u32::from(self.bit_count) & 0x07;
        let mut zeros = u32::from(self.zero_run);

        let delta = self.current_value.to_i32() - self.prev_value.to_i32();
        if delta == 0 {
            zeros += 1;
        } else {
            flush_pending_zeros(&mut accum, &mut bits, &mut zeros, &mut final_data);
            write_delta(&mut accum, &mut bits, delta, &mut final_data);
        }
        flush_pending_zeros(&mut accum, &mut bits, &mut zeros, &mut final_data);
        flush_remaining_bits(&mut accum, &mut bits, &mut final_data);

        // Decode bit stream using shared helper
        let mut reader = BitReader::new(&final_data);
        decode_bitstream::<V>(&mut reader, &mut result, base_ts, interval, count, first_val.to_i32())?;

        Ok(result)
    }

    /// Serialize the encoder to bytes (appendable format)
    ///
    /// The appendable format preserves all internal state, allowing the encoder
    /// to be restored with `from_bytes()` and continue appending readings.
    ///
    /// For a more compact read-only format, use `freeze()` instead.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        if self.count == 0 {
            return Vec::new();
        }
        let header_size = Self::wire_header_size();
        let mut buf = vec![0u8; header_size + self.data.len()];

        write_u32_le(&mut buf, 0, self.base_ts);
        write_u16_le(&mut buf, 4, self.count);
        write_u16_le(&mut buf, 6, self.prev_logical_idx);
        self.first_value.write_le(&mut buf[8..]);
        self.prev_value.write_le(&mut buf[8 + V::BYTES..]);
        self.current_value.write_le(&mut buf[8 + 2 * V::BYTES..]);
        buf[8 + 3 * V::BYTES] = self.zero_run;
        buf[8 + 3 * V::BYTES + 1] = self.bit_count;
        buf[8 + 3 * V::BYTES + 2] = self.bit_accum;
        buf[header_size..].copy_from_slice(&self.data);
        buf
    }

    /// Restore an encoder from bytes (appendable format)
    ///
    /// The bytes must have been created by `to_bytes()`. After restoration,
    /// you can continue appending readings to the encoder.
    ///
    /// # Errors
    /// Returns `AppendError::BufferTooShort` if the buffer is too short to contain a valid header.
    /// Returns `AppendError::MalformedData` if the header contains invalid values.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AppendError<V>> {
        if bytes.is_empty() {
            return Ok(Self::new());
        }
        let header_size = Self::wire_header_size();
        if bytes.len() < header_size {
            return Err(AppendError::BufferTooShort { expected: header_size, actual: bytes.len() });
        }

        let first_value = V::read_le(&bytes[8..]);
        let prev_value = V::read_le(&bytes[8 + V::BYTES..]);
        let current_value = V::read_le(&bytes[8 + 2 * V::BYTES..]);
        let bit_count = bytes[8 + 3 * V::BYTES + 1];

        // Validate bit_count is in valid range (0-7 for partial byte)
        if bit_count > 7 {
            return Err(AppendError::MalformedData);
        }

        // Validate that value deltas won't overflow when computing
        // The encoder only produces deltas in range [-1024, 1023], so values
        // that would produce larger deltas indicate corrupted data
        let first_i64 = i64::from(first_value.to_i32());
        let prev_i64 = i64::from(prev_value.to_i32());
        let current_i64 = i64::from(current_value.to_i32());

        // Check current - prev won't overflow i32
        let delta = current_i64 - prev_i64;
        if delta < i64::from(i32::MIN) || delta > i64::from(i32::MAX) {
            return Err(AppendError::MalformedData);
        }

        // Check first - prev won't overflow (used in some code paths)
        let delta2 = first_i64 - prev_i64;
        if delta2 < i64::from(i32::MIN) || delta2 > i64::from(i32::MAX) {
            return Err(AppendError::MalformedData);
        }

        Ok(Self {
            base_ts: read_u32_le(bytes, 0),
            count: read_u16_le(bytes, 4),
            prev_logical_idx: read_u16_le(bytes, 6),
            first_value,
            prev_value,
            current_value,
            zero_run: bytes[8 + 3 * V::BYTES],
            bit_count,
            bit_accum: bytes[8 + 3 * V::BYTES + 2],
            data: bytes[header_size..].to_vec(),
        })
    }

    /// Create a compact, read-only snapshot of the encoded data (frozen format)
    ///
    /// The frozen format is more compact than `to_bytes()` but cannot be restored
    /// to an encoder. Use `decode_frozen()` to read the data back.
    ///
    /// This is the recommended format for long-term storage.
    #[must_use]
    pub fn freeze(&self) -> Vec<u8> {
        if self.count == 0 {
            return Vec::new();
        }

        let header_size = Self::frozen_header_size();

        if self.count == 1 {
            let mut buf = vec![0u8; header_size];
            write_u32_le(&mut buf, 0, self.base_ts);
            write_u16_le(&mut buf, 4, self.count);
            self.current_value.write_le(&mut buf[6..]);
            return buf;
        }

        // Finalize pending state
        let mut final_data = self.data.clone();
        let mut accum = u32::from(self.bit_accum);
        // Sanitize bit_count - valid range is 0-7 (partial byte)
        let mut bits = u32::from(self.bit_count) & 0x07;
        let mut zeros = u32::from(self.zero_run);

        let delta = self.current_value.to_i32() - self.prev_value.to_i32();
        if delta == 0 {
            zeros += 1;
        } else {
            flush_pending_zeros(&mut accum, &mut bits, &mut zeros, &mut final_data);
            write_delta(&mut accum, &mut bits, delta, &mut final_data);
        }
        flush_pending_zeros(&mut accum, &mut bits, &mut zeros, &mut final_data);
        flush_remaining_bits(&mut accum, &mut bits, &mut final_data);

        let mut buf = vec![0u8; header_size];
        write_u32_le(&mut buf, 0, self.base_ts);
        write_u16_le(&mut buf, 4, self.count);
        self.first_value.write_le(&mut buf[6..]);
        buf.extend_from_slice(&final_data);
        buf
    }

    #[inline]
    const fn wire_header_size() -> usize {
        4 + 2 + 2 + V::BYTES * 3 + 3
    }

    #[inline]
    const fn frozen_header_size() -> usize {
        4 + 2 + V::BYTES
    }

    #[inline]
    fn write_bits(&mut self, value: u32, num_bits: u32) {
        let mut accum = u32::from(self.bit_accum);
        let mut count = self.bit_count;
        accum = (accum << num_bits) | value;
        count += num_bits as u8;
        while count >= 8 {
            count -= 8;
            self.data.push((accum >> count) as u8);
        }
        self.bit_accum = accum as u8;
        self.bit_count = count;
    }

    #[inline]
    fn flush_zeros(&mut self) {
        while self.zero_run > 0 {
            let (bits, num_bits, consumed) = encode_zero_run(u32::from(self.zero_run));
            self.write_bits(bits, num_bits);
            self.zero_run -= consumed as u8;
        }
    }

    #[inline]
    fn encode_delta(&mut self, delta: i32) {
        let idx = (delta + 10) as usize;
        if idx <= 20 {
            let (bits, num_bits) = DELTA_ENCODE[idx];
            if num_bits > 0 {
                self.write_bits(bits, u32::from(num_bits));
                return;
            }
        }
        let bits = (0b1111_1110_u32 << 11) | ((delta as u32) & 0x7FF);
        self.write_bits(bits, 19);
    }

    #[inline]
    fn write_gaps(&mut self, mut count: u32) {
        while count > 0 {
            if count == 1 {
                self.write_bits(0b110, 3);
                count = 0;
            } else {
                let g = count.min(MAX_GAP_PER_MARKER);
                self.write_bits((0xFF << 6) | (g - 2), 14);
                count -= g;
            }
        }
    }
}

impl<V: Value, const INTERVAL: u16> Default for Encoder<V, INTERVAL> {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

fn flush_pending_zeros(accum: &mut u32, bits: &mut u32, zeros: &mut u32, out: &mut Vec<u8>) {
    // Limit iterations to prevent runaway loops from malformed input
    let mut iterations = 0u32;
    while *zeros > 0 && iterations < 1000 {
        iterations += 1;
        let (b, n, c) = encode_zero_run(*zeros);
        if n > 0 && n < 32 {
            *accum = accum.wrapping_shl(n) | b;
            *bits = bits.saturating_add(n);
        }
        *zeros = zeros.saturating_sub(c);
        while *bits >= 8 {
            *bits -= 8;
            if *bits < 32 {
                out.push((*accum >> *bits) as u8);
            }
        }
    }
}

fn write_delta(accum: &mut u32, bits: &mut u32, delta: i32, out: &mut Vec<u8>) {
    let idx = delta.saturating_add(10);
    let (b, n) = if (0..=20).contains(&idx) {
        let (b, n) = DELTA_ENCODE[idx as usize];
        if n > 0 { (b, u32::from(n)) } else { ((0b1111_1110_u32 << 11) | ((delta as u32) & 0x7FF), 19) }
    } else {
        ((0b1111_1110_u32 << 11) | ((delta as u32) & 0x7FF), 19)
    };
    if n > 0 && n < 32 {
        *accum = accum.wrapping_shl(n) | b;
        *bits = bits.saturating_add(n);
    }
    while *bits >= 8 {
        *bits -= 8;
        if *bits < 32 {
            out.push((*accum >> *bits) as u8);
        }
    }
}

fn flush_remaining_bits(accum: &mut u32, bits: &mut u32, out: &mut Vec<u8>) {
    if *bits > 0 && *bits < 8 {
        out.push((*accum << (8 - *bits)) as u8);
        *bits = 0;
    }
}

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
    buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

#[inline]
fn write_u32_le(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
