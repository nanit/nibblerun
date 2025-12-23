//! Encoder for nibblerun time series compression.

use serde::{Deserialize, Serialize};

use crate::constants::{div_by_interval, encode_zero_run, DELTA_ENCODE};
use crate::error::AppendError;
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
    pub fn append(&mut self, ts: u32, value: V) -> Result<(), AppendError> {
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

        // Finalize previous interval
        if self.count == 1 {
            self.first_value = self.current_value;
        } else {
            let delta = self.current_value.to_i32() - self.prev_value.to_i32();
            if !(-1024..=1023).contains(&delta) {
                return Err(AppendError::DeltaOverflow {
                    delta,
                    prev_value: self.prev_value.to_i32(),
                    new_value: self.current_value.to_i32(),
                });
            }
            if delta == 0 {
                self.zero_run = self.zero_run.saturating_add(1);
                if self.zero_run >= 149 {
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

    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        if self.count == 0 { 0 } else { Self::wire_header_size() + self.data.len() }
    }

    #[inline]
    #[must_use]
    pub fn count(&self) -> usize {
        self.count as usize
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[must_use]
    pub fn decode(&self) -> Vec<Reading<V>> {
        if self.count == 0 {
            return Vec::new();
        }

        let base_ts = self.base_ts;
        let interval = u32::from(INTERVAL);
        let count = self.count as usize;

        let first_val = if count == 1 { self.current_value } else { self.first_value };

        let mut result = Vec::with_capacity(count);
        result.push(Reading { ts: base_ts, value: first_val });

        if count == 1 {
            return result;
        }

        // Finalize bit data
        let mut final_data = self.data.clone();
        let mut accum = u32::from(self.bit_accum);
        let mut bits = u32::from(self.bit_count);
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

        // Decode bit stream
        let mut reader = BitReader::new(&final_data);
        let mut prev_val = first_val.to_i32();
        let mut idx = 1u32;

        while result.len() < count && reader.has_more() {
            if reader.read_bits(1) == 0 {
                result.push(Reading { ts: base_ts + idx * interval, value: V::from_i32(prev_val) });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                prev_val += if reader.read_bits(1) == 0 { 1 } else { -1 };
                result.push(Reading { ts: base_ts + idx * interval, value: V::from_i32(prev_val) });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                prev_val += if reader.read_bits(1) == 0 { 2 } else { -2 };
                result.push(Reading { ts: base_ts + idx * interval, value: V::from_i32(prev_val) });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                for _ in 0..reader.read_bits(4) + 8 {
                    if result.len() >= count { break; }
                    result.push(Reading { ts: base_ts + idx * interval, value: V::from_i32(prev_val) });
                    idx += 1;
                }
                continue;
            }
            if reader.read_bits(1) == 0 {
                for _ in 0..reader.read_bits(7) + 22 {
                    if result.len() >= count { break; }
                    result.push(Reading { ts: base_ts + idx * interval, value: V::from_i32(prev_val) });
                    idx += 1;
                }
                continue;
            }
            if reader.read_bits(1) == 0 {
                let e = reader.read_bits(4) as i32;
                prev_val += if e < 8 { e - 10 } else { e - 5 };
                result.push(Reading { ts: base_ts + idx * interval, value: V::from_i32(prev_val) });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                let raw = reader.read_bits(11);
                prev_val += if raw & 0x400 != 0 { (raw | 0xFFFF_F800) as i32 } else { raw as i32 };
                result.push(Reading { ts: base_ts + idx * interval, value: V::from_i32(prev_val) });
                idx += 1;
            } else {
                idx += reader.read_bits(6) + 2;
            }
        }

        result
    }

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

    #[must_use]
    pub fn as_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, AppendError> {
        if bytes.is_empty() {
            return Ok(Self::new());
        }
        let header_size = Self::wire_header_size();
        if bytes.len() < header_size {
            return Err(AppendError::CountOverflow);
        }
        Ok(Self {
            base_ts: read_u32_le(&bytes, 0),
            count: read_u16_le(&bytes, 4),
            prev_logical_idx: read_u16_le(&bytes, 6),
            first_value: V::read_le(&bytes[8..]),
            prev_value: V::read_le(&bytes[8 + V::BYTES..]),
            current_value: V::read_le(&bytes[8 + 2 * V::BYTES..]),
            zero_run: bytes[8 + 3 * V::BYTES],
            bit_count: bytes[8 + 3 * V::BYTES + 1],
            bit_accum: bytes[8 + 3 * V::BYTES + 2],
            data: bytes[header_size..].to_vec(),
        })
    }

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
        let mut bits = u32::from(self.bit_count);
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
                let g = count.min(65);
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
    while *zeros > 0 {
        let (b, n, c) = encode_zero_run(*zeros);
        *accum = (*accum << n) | b;
        *bits += n;
        *zeros -= c;
        while *bits >= 8 {
            *bits -= 8;
            out.push((*accum >> *bits) as u8);
        }
    }
}

fn write_delta(accum: &mut u32, bits: &mut u32, delta: i32, out: &mut Vec<u8>) {
    let idx = (delta + 10) as usize;
    let (b, n) = if idx <= 20 {
        let (b, n) = DELTA_ENCODE[idx];
        if n > 0 { (b, u32::from(n)) } else { ((0b1111_1110_u32 << 11) | ((delta as u32) & 0x7FF), 19) }
    } else {
        ((0b1111_1110_u32 << 11) | ((delta as u32) & 0x7FF), 19)
    };
    *accum = (*accum << n) | b;
    *bits += n;
    while *bits >= 8 {
        *bits -= 8;
        out.push((*accum >> *bits) as u8);
    }
}

fn flush_remaining_bits(accum: &mut u32, bits: &mut u32, out: &mut Vec<u8>) {
    if *bits > 0 {
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

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    bits: u64,
    left: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut r = Self { buf, pos: 0, bits: 0, left: 0 };
        r.refill();
        r
    }

    fn refill(&mut self) {
        while self.left <= 56 && self.pos < self.buf.len() {
            self.bits = (self.bits << 8) | u64::from(self.buf[self.pos]);
            self.pos += 1;
            self.left += 8;
        }
    }

    fn read_bits(&mut self, n: u32) -> u32 {
        if self.left < n { self.refill(); }
        if self.left < n { return 0; }
        self.left -= n;
        ((self.bits >> self.left) & ((1 << n) - 1)) as u32
    }

    fn has_more(&self) -> bool {
        self.left > 0 || self.pos < self.buf.len()
    }
}
