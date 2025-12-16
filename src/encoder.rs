//! Encoder for nibblerun time series compression.

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::constants::{
    cold_gap_handler, div_by_interval, encode_zero_run, pack_pending, rounded_avg, unpack_pending,
    DELTA_ENCODE, EPOCH_BASE,
};
use crate::error::AppendError;
use crate::reading::Reading;
use crate::value::Value;

/// Encoder for `NibbleRun` format
///
/// Accumulates sensor readings and produces compressed output.
/// Generic over value type V (i8, i16, or i32) for compile-time type safety.
#[repr(C)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Encoder<V: Value> {
    base_ts: u64,
    last_ts: u64,
    bit_accum: u64,
    /// Packed state: bits 0-5 = `bit_accum_count`, bits 6-15 = `pending_count` (0-1023), bits 16-47 = `pending_sum`
    pending_state: u64,
    data: Vec<u8>,
    prev_temp: i32,
    first_temp: i32,
    zero_run: u32,
    prev_logical_idx: u32,
    count: u16,
    interval: u16,
    #[serde(skip)]
    _marker: PhantomData<V>,
}

impl<V: Value> Encoder<V> {
    /// Create a new encoder with the specified interval
    ///
    /// # Arguments
    /// * `interval` - Interval between readings in seconds (e.g., 300 for 5 minutes)
    #[inline]
    #[must_use]
    pub fn new(interval: u16) -> Self {
        Encoder {
            base_ts: 0,
            last_ts: 0,
            bit_accum: 0,
            data: Vec::with_capacity(48),
            prev_temp: 0,
            first_temp: 0,
            zero_run: 0,
            pending_state: 0,
            prev_logical_idx: 0,
            count: 0,
            interval,
            _marker: PhantomData,
        }
    }

    /// Get the interval in seconds
    #[inline]
    #[must_use]
    pub fn interval(&self) -> u16 {
        self.interval
    }

    /// Header size for this encoder's value type
    #[inline]
    #[must_use]
    pub const fn header_size() -> usize {
        4 + 2 + V::BYTES // base_ts_offset (4) + count (2) + first_value (V::BYTES)
    }

    /// Append a sensor reading
    ///
    /// Multiple readings in the same interval are averaged.
    /// The value type V provides compile-time range checking.
    ///
    /// # Arguments
    /// * `ts` - Unix timestamp in seconds
    /// * `value` - Sensor value (type checked at compile time)
    ///
    /// # Errors
    /// Returns an error if:
    /// - Timestamp is before the base timestamp
    /// - Timestamp is out of order (earlier interval than previous)
    /// - Too many readings in the same interval (max 1023)
    /// - Too many total readings (max 65535)
    /// - Value delta exceeds encodable range [-1024, 1023]
    #[inline]
    pub fn append(&mut self, ts: u64, value: V) -> Result<(), AppendError> {
        let value = value.to_i32();

        // First reading - initialize pending state
        if self.count == 0 {
            self.base_ts = ts;
            self.last_ts = ts;
            self.first_temp = value;
            self.prev_temp = value;
            self.prev_logical_idx = 0;
            self.count = 1;
            // Initialize pending: count=1, sum=value
            self.pending_state = pack_pending(0, 1, value);
            return Ok(());
        }

        // Reject out-of-order readings (ts before base_ts)
        if ts < self.base_ts {
            return Err(AppendError::TimestampBeforeBase {
                ts,
                base_ts: self.base_ts,
            });
        }

        // Calculate logical index
        let logical_idx = div_by_interval(ts - self.base_ts, self.interval) as u32;

        // Reject readings that go backwards in time
        if logical_idx < self.prev_logical_idx {
            return Err(AppendError::OutOfOrder {
                ts,
                logical_idx,
                prev_logical_idx: self.prev_logical_idx,
            });
        }

        // Same interval - accumulate for averaging
        if logical_idx == self.prev_logical_idx {
            let (bits, count, sum) = unpack_pending(self.pending_state);
            if count >= 1023 {
                return Err(AppendError::IntervalOverflow { count });
            }
            self.pending_state = pack_pending(bits, count + 1, sum.saturating_add(value));
            self.last_ts = ts;
            return Ok(());
        }

        // New interval - check for potential errors before committing

        // Check count overflow
        if self.count == u16::MAX {
            return Err(AppendError::CountOverflow);
        }

        // Check delta overflow: compute what the delta would be after finalizing
        let (_, pending_count, pending_sum) = unpack_pending(self.pending_state);
        if pending_count > 0 && self.count > 1 {
            let avg = rounded_avg(pending_sum, pending_count);
            let delta = avg - self.prev_temp;
            if !(-1024..=1023).contains(&delta) {
                return Err(AppendError::DeltaOverflow {
                    delta,
                    prev_value: self.prev_temp,
                    new_value: avg,
                });
            }
        }

        // All checks passed - finalize previous interval
        self.finalize_pending_interval();

        let index_gap = logical_idx - self.prev_logical_idx;

        // Gap handling (rare)
        if index_gap > 1 {
            cold_gap_handler();
            self.flush_zeros();
            self.write_gaps(index_gap - 1);
        }

        // Start accumulating for new interval
        self.prev_logical_idx = logical_idx;
        self.last_ts = ts;
        self.count += 1;
        // Initialize pending for new interval: count=1, sum=value
        let (bits, _, _) = unpack_pending(self.pending_state);
        self.pending_state = pack_pending(bits, 1, value);

        Ok(())
    }

    /// Finalize the pending interval: compute average and encode the delta
    /// Called when crossing to a new interval or when serializing
    #[inline]
    fn finalize_pending_interval(&mut self) {
        let (_, count, sum) = unpack_pending(self.pending_state);
        if count == 0 {
            return;
        }

        // Compute average with proper rounding (round half away from zero)
        let avg = rounded_avg(sum, count);

        // For the first interval, update first_temp to the average
        if self.count == 1 {
            self.first_temp = avg;
            self.prev_temp = avg;
        } else {
            // Encode delta from previous interval's average
            let delta = avg - self.prev_temp;
            if delta == 0 {
                self.zero_run += 1;
            } else {
                self.flush_zeros();
                self.encode_delta(delta);
            }
            self.prev_temp = avg;
        }

        // Clear pending state (re-extract bits after any encoding that may have occurred)
        let (bits, _, _) = unpack_pending(self.pending_state);
        self.pending_state = u64::from(bits); // Clear pending count/sum, keep actual bits
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
        // Large delta: must fit in 11-bit signed range [-1024, 1023]
        // New encoding: 11111110 (8-bit prefix) + 11-bit signed value = 19 bits
        assert!(
            (-1024..=1023).contains(&delta),
            "Delta {delta} out of range [-1024, 1023]. Temperature swings this large are not supported."
        );
        let bits = (0b1111_1110_u32 << 11) | ((delta as u32) & 0x7FF);
        self.write_bits(bits, 19);
    }

    /// Get the encoded size in bytes
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        if self.count == 0 {
            return 0;
        }

        let header_size = Self::header_size();

        // Extract pending state
        let (actual_bits, pending_count, pending_sum) = unpack_pending(self.pending_state);

        // For single interval, no additional bits needed
        if self.count == 1 {
            return header_size;
        }

        // Calculate the final interval's delta and its encoding size
        let mut zeros = self.zero_run;
        let mut extra_bits = 0u32;

        if pending_count > 0 {
            let final_avg = rounded_avg(pending_sum, pending_count);
            let delta = final_avg - self.prev_temp;
            if delta == 0 {
                zeros += 1;
            } else {
                // Need to encode zeros first, then the delta
                extra_bits += Self::zero_run_bits(self.zero_run);
                zeros = 0; // zeros already accounted for
                extra_bits += Self::delta_bits(delta);
            }
        }

        // Calculate bits needed for pending zero run
        let zero_run_bits = Self::zero_run_bits(zeros);

        let total_bits = actual_bits + zero_run_bits + extra_bits;

        header_size + self.data.len() + (total_bits as usize).div_ceil(8)
    }

    /// Calculate the number of bits needed to encode a delta
    #[inline]
    fn delta_bits(delta: i32) -> u32 {
        let idx = (delta + 10) as usize;
        if idx <= 20 {
            let (_, num_bits) = DELTA_ENCODE[idx];
            if num_bits > 0 {
                return u32::from(num_bits);
            }
        }
        19 // Large delta encoding (11111110 + 11 bits)
    }

    /// Calculate the number of bits needed to encode a zero run
    ///
    /// Must match the logic in `encode_zero_run`:
    /// - n <= 7: individual zeros (1 bit each)
    /// - n 8-21: 9-bit run encoding (prefix 11110 + 4-bit length)
    /// - n 22-149: 13-bit run encoding (prefix 111110 + 7-bit length)
    /// - n 150+: multiple 13-bit encodings
    #[inline]
    fn zero_run_bits(mut n: u32) -> u32 {
        let mut bits = 0;
        while n > 0 {
            if n <= 7 {
                // Individual zeros: 1 bit each
                bits += n;
                n = 0;
            } else if n <= 21 {
                bits += 9;
                n = 0;
            } else if n <= 149 {
                bits += 13;
                n = 0;
            } else {
                bits += 13;
                n -= 149;
            }
        }
        bits
    }

    /// Get the number of readings encoded
    #[inline]
    #[must_use]
    pub fn count(&self) -> usize {
        self.count as usize
    }

    /// Decode the encoder's contents back to readings
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn decode(&self) -> Vec<Reading<V>> {
        if self.count == 0 {
            return Vec::new();
        }

        // Extract pending state
        let (actual_bits, pending_count, pending_sum) = unpack_pending(self.pending_state);

        // Compute the average for the final pending interval
        let final_avg = if pending_count > 0 {
            rounded_avg(pending_sum, pending_count)
        } else {
            self.prev_temp
        };

        // For single interval, first_temp is the average
        let first_temp = if self.count == 1 {
            final_avg
        } else {
            self.first_temp
        };

        let mut decoded = Vec::with_capacity(self.count as usize);
        decoded.push(Reading {
            ts: self.base_ts,
            value: V::from_i32(first_temp),
        });

        if self.count == 1 {
            return decoded;
        }

        // Build finalized bit data (same as to_bytes but just the data portion)
        let mut final_data = self.data.clone();
        let mut accum = self.bit_accum;
        let mut bits = actual_bits;
        let mut zeros = self.zero_run;

        // Helper to flush complete bytes from accumulator
        let flush_bytes = |accum: &mut u64, bits: &mut u32, out: &mut Vec<u8>| {
            while *bits >= 8 {
                *bits -= 8;
                out.push((*accum >> *bits) as u8);
            }
        };

        // For multi-interval encoders, encode the final interval's delta
        if pending_count > 0 {
            let delta = final_avg - self.prev_temp;
            if delta == 0 {
                zeros += 1;
            } else {
                // First flush pending zeros
                while zeros > 0 {
                    let (b, n, c) = encode_zero_run(zeros);
                    accum = (accum << n) | u64::from(b);
                    bits += n;
                    zeros -= c;
                    flush_bytes(&mut accum, &mut bits, &mut final_data);
                }
                // Encode the delta
                let (delta_bits, delta_num_bits) = Self::encode_delta_value(delta);
                accum = (accum << delta_num_bits) | u64::from(delta_bits);
                bits += delta_num_bits;
                flush_bytes(&mut accum, &mut bits, &mut final_data);
            }
        }

        // Flush remaining zeros
        while zeros > 0 {
            let (b, n, c) = encode_zero_run(zeros);
            accum = (accum << n) | u64::from(b);
            bits += n;
            zeros -= c;
            flush_bytes(&mut accum, &mut bits, &mut final_data);
        }

        // Flush any remaining complete bytes
        flush_bytes(&mut accum, &mut bits, &mut final_data);

        if bits > 0 {
            final_data.push((accum << (8 - bits)) as u8);
        }

        let mut reader = BitReader::new(&final_data);
        let mut prev_temp = first_temp; // Use computed first_temp (may be averaged)
        let mut idx = 1u64;
        let count = self.count as usize;

        let interval = u64::from(self.interval);
        // New encoding scheme:
        // 0       = zero delta
        // 100     = +1, 101 = -1
        // 110     = single-interval gap
        // 11100   = +2, 11101 = -2
        // 11110xxxx = zero run 8-21
        // 111110xxxxxxx = zero run 22-149
        // 1111110xxxx = ±3-10 delta
        // 11111110xxxxxxxxxxx = large delta
        // 11111111xxxxxx = gap 2-65
        while decoded.len() < count && reader.has_more() {
            if reader.read_bits(1) == 0 {
                // 0 = zero delta
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    value: V::from_i32(prev_temp),
                });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                // 10x = ±1 delta
                prev_temp += if reader.read_bits(1) == 0 { 1 } else { -1 };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    value: V::from_i32(prev_temp),
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
                prev_temp += if reader.read_bits(1) == 0 { 2 } else { -2 };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    value: V::from_i32(prev_temp),
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
                        ts: self.base_ts + idx * interval,
                        value: V::from_i32(prev_temp),
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
                        ts: self.base_ts + idx * interval,
                        value: V::from_i32(prev_temp),
                    });
                    idx += 1;
                }
                continue;
            }
            // 111111...
            if reader.read_bits(1) == 0 {
                // 1111110xxxx = ±3-10 delta
                let e = reader.read_bits(4) as i32;
                prev_temp += if e < 8 { e - 10 } else { e - 5 };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    value: V::from_i32(prev_temp),
                });
                idx += 1;
                continue;
            }
            // 1111111...
            if reader.read_bits(1) == 0 {
                // 11111110xxxxxxxxxxx = large delta (±11-1023)
                let raw = reader.read_bits(11);
                prev_temp += if raw & 0x400 != 0 {
                    (raw | 0xFFFF_F800) as i32
                } else {
                    raw as i32
                };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    value: V::from_i32(prev_temp),
                });
                idx += 1;
            } else {
                // 11111111xxxxxx = gap 2-65 intervals
                idx += u64::from(reader.read_bits(6) + 2);
            }
        }

        decoded
    }

    /// Finalize and return the encoded bytes
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        if self.count == 0 {
            return Vec::new();
        }

        let header_size = Self::header_size();

        // Extract pending state
        let (actual_bits, pending_count, pending_sum) = unpack_pending(self.pending_state);

        // Compute the average for the final pending interval
        let final_avg = if pending_count > 0 {
            rounded_avg(pending_sum, pending_count)
        } else {
            self.prev_temp
        };

        // For single interval, first_temp is the average
        let first_temp = if self.count == 1 {
            final_avg
        } else {
            self.first_temp
        };

        let mut result = Vec::with_capacity(header_size + self.data.len() + 4);

        // Header: base_ts_offset (4) + count (2) + first_value (V::BYTES)
        // Use wrapping_sub for consistent behavior in debug/release modes
        // Note: timestamps before EPOCH_BASE will produce incorrect data
        let base_ts_offset = self.base_ts.wrapping_sub(EPOCH_BASE) as u32;
        result.extend_from_slice(&base_ts_offset.to_le_bytes()); // offset 0: 4 bytes
        result.extend_from_slice(&self.count.to_le_bytes()); // offset 4: 2 bytes

        // Write first_value with appropriate size for V
        let first_value = V::from_i32(first_temp);
        let mut value_buf = [0u8; 4];
        first_value.write_le(&mut value_buf);
        result.extend_from_slice(&value_buf[..V::BYTES]); // offset 6: V::BYTES bytes

        // Data
        result.extend_from_slice(&self.data);

        // Finalize pending bits and zeros, plus the final interval's delta
        let mut accum = self.bit_accum;
        let mut bits = actual_bits;
        let mut zeros = self.zero_run;

        // Helper to flush complete bytes from accumulator
        let flush_bytes = |accum: &mut u64, bits: &mut u32, out: &mut Vec<u8>| {
            while *bits >= 8 {
                *bits -= 8;
                out.push((*accum >> *bits) as u8);
            }
        };

        // For multi-interval encoders, encode the final interval's delta
        if self.count > 1 && pending_count > 0 {
            let delta = final_avg - self.prev_temp;
            if delta == 0 {
                zeros += 1;
            } else {
                // First flush pending zeros
                while zeros > 0 {
                    let (b, n, c) = encode_zero_run(zeros);
                    accum = (accum << n) | u64::from(b);
                    bits += n;
                    zeros -= c;
                    flush_bytes(&mut accum, &mut bits, &mut result);
                }
                // Encode the delta
                let (delta_bits, delta_num_bits) = Self::encode_delta_value(delta);
                accum = (accum << delta_num_bits) | u64::from(delta_bits);
                bits += delta_num_bits;
                flush_bytes(&mut accum, &mut bits, &mut result);
            }
        }

        // Flush remaining zeros
        while zeros > 0 {
            let (b, n, c) = encode_zero_run(zeros);
            accum = (accum << n) | u64::from(b);
            bits += n;
            zeros -= c;
            flush_bytes(&mut accum, &mut bits, &mut result);
        }

        // Flush any remaining complete bytes
        flush_bytes(&mut accum, &mut bits, &mut result);

        // Pad remaining bits
        if bits > 0 {
            result.push((accum << (8 - bits)) as u8);
        }

        result
    }

    /// Encode a delta value, returning (bits, `num_bits`)
    #[inline]
    fn encode_delta_value(delta: i32) -> (u32, u32) {
        let idx = (delta + 10) as usize;
        if idx <= 20 {
            let (bits, num_bits) = DELTA_ENCODE[idx];
            if num_bits > 0 {
                return (bits, u32::from(num_bits));
            }
        }
        // Large delta: clamp to 11-bit signed range
        // New encoding: 11111110 (8-bit prefix) + 11-bit signed value = 19 bits
        let clamped = delta.clamp(-1024, 1023);
        let bits = (0b1111_1110_u32 << 11) | ((clamped as u32) & 0x7FF);
        (bits, 19)
    }

    #[inline]
    fn write_bits(&mut self, value: u32, num_bits: u32) {
        // Extract actual bits and pending state
        let (mut bits, pending_count, pending_sum) = unpack_pending(self.pending_state);

        self.bit_accum = (self.bit_accum << num_bits) | u64::from(value);
        bits += num_bits;

        // Flush complete bytes to prevent overflow
        while bits >= 8 {
            bits -= 8;
            self.data.push((self.bit_accum >> bits) as u8);
        }

        // Repack with pending state preserved
        self.pending_state = pack_pending(bits, pending_count, pending_sum);
    }

    #[inline]
    fn flush_zeros(&mut self) {
        while self.zero_run > 0 {
            let (bits, num_bits, consumed) = encode_zero_run(self.zero_run);
            self.write_bits(bits, num_bits);
            self.zero_run -= consumed;
        }
    }

    #[inline]
    fn write_gaps(&mut self, mut count: u32) {
        // New encoding:
        // - Single gap (1 interval): 110 (3 bits)
        // - Multi gap (2-65 intervals): 11111111xxxxxx (14 bits, 6-bit count for 2-65)
        while count > 0 {
            if count == 1 {
                // Single-interval gap: 110 (3 bits)
                self.write_bits(0b110, 3);
                count = 0;
            } else {
                // Multi-interval gap: 2-65 intervals
                // Encoding: 11111111 (8-bit prefix) + 6-bit value (0-63 maps to 2-65)
                let g = count.min(65);
                self.write_bits((0xFF << 6) | (g - 2), 14);
                count -= g;
            }
        }
    }
}


/// Bit reader for decoding variable-length bit sequences (internal to `Encoder::decode`)
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
    fn has_more(&self) -> bool {
        self.left > 0 || self.pos < self.buf.len()
    }
}
