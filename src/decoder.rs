//! Decoding functionality for nibblerun encoded data.

use crate::constants::{encode_zero_run, DELTA_ENCODE};
use crate::reading::Reading;
use crate::value::Value;

/// Decode frozen format bytes back to readings
///
/// # Type Parameters
/// * `V` - Value type (i8, i16, or i32). Must match the type used during encoding.
/// * `INTERVAL` - The interval in seconds (must match encoder's interval)
///
/// # Arguments
/// * `bytes` - Frozen format bytes from `Encoder::freeze()`
///
/// # Returns
/// Vector of decoded readings. Returns an empty vector if bytes is too short.
#[must_use]
pub fn decode<V: Value, const INTERVAL: u16>(bytes: &[u8]) -> Vec<Reading<V>> {
    decode_frozen::<V, INTERVAL>(bytes)
}

/// Decode frozen format bytes
#[must_use]
pub fn decode_frozen<V: Value, const INTERVAL: u16>(buf: &[u8]) -> Vec<Reading<V>> {
    let header_size = 4 + 2 + V::BYTES;
    if buf.len() < header_size {
        return Vec::new();
    }

    let base_ts = read_u32_le(buf, 0);
    let count = read_u16_le(buf, 4) as usize;
    let first_value = V::read_le(&buf[6..]).to_i32();

    if count == 0 {
        return Vec::new();
    }

    let interval = u32::from(INTERVAL);
    let mut result = Vec::with_capacity(count);
    result.push(Reading { ts: base_ts, value: V::from_i32(first_value) });

    if count == 1 {
        return result;
    }

    let data = &buf[header_size..];
    let mut reader = BitReader::new(data);
    let mut prev_val = first_value;
    let mut idx = 1u32;

    while result.len() < count && reader.has_more() {
        let ts = base_ts.wrapping_add(idx.wrapping_mul(interval));
        if reader.read_bits(1) == 0 {
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            prev_val = prev_val.wrapping_add(if reader.read_bits(1) == 0 { 1 } else { -1 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            prev_val = prev_val.wrapping_add(if reader.read_bits(1) == 0 { 2 } else { -2 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            for _ in 0..reader.read_bits(4) + 8 {
                if result.len() >= count { break; }
                let ts = base_ts.wrapping_add(idx.wrapping_mul(interval));
                result.push(Reading { ts, value: V::from_i32(prev_val) });
                idx = idx.wrapping_add(1);
            }
            continue;
        }
        if reader.read_bits(1) == 0 {
            for _ in 0..reader.read_bits(7) + 22 {
                if result.len() >= count { break; }
                let ts = base_ts.wrapping_add(idx.wrapping_mul(interval));
                result.push(Reading { ts, value: V::from_i32(prev_val) });
                idx = idx.wrapping_add(1);
            }
            continue;
        }
        if reader.read_bits(1) == 0 {
            let e = reader.read_bits(4) as i32;
            prev_val = prev_val.wrapping_add(if e < 8 { e - 10 } else { e - 5 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            let raw = reader.read_bits(11);
            prev_val = prev_val.wrapping_add(if raw & 0x400 != 0 { (raw | 0xFFFF_F800) as i32 } else { raw as i32 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
        } else {
            idx = idx.wrapping_add(reader.read_bits(6) + 2);
        }
    }

    result
}

/// Decode appendable format bytes (with pending state)
#[must_use]
pub fn decode_appendable<V: Value, const INTERVAL: u16>(buf: &[u8]) -> Vec<Reading<V>> {
    let header_size = 4 + 2 + 2 + V::BYTES * 3 + 3;
    if buf.len() < header_size {
        return Vec::new();
    }

    let base_ts = read_u32_le(buf, 0);
    let count = read_u16_le(buf, 4) as usize;
    let first_value = V::read_le(&buf[8..]).to_i32();
    let prev_value = V::read_le(&buf[8 + V::BYTES..]).to_i32();
    let current_value = V::read_le(&buf[8 + 2 * V::BYTES..]).to_i32();
    let zero_run = buf[8 + 3 * V::BYTES];
    let bit_count = buf[8 + 3 * V::BYTES + 1];
    let bit_accum = buf[8 + 3 * V::BYTES + 2];

    if count == 0 {
        return Vec::new();
    }

    let interval = u32::from(INTERVAL);
    let first_val = if count == 1 { current_value } else { first_value };

    let mut result = Vec::with_capacity(count);
    result.push(Reading { ts: base_ts, value: V::from_i32(first_val) });

    if count == 1 {
        return result;
    }

    // Finalize bit data
    let mut final_data = buf[header_size..].to_vec();
    let mut accum = u32::from(bit_accum);
    let mut bits = u32::from(bit_count);
    let mut zeros = u32::from(zero_run);

    let delta = current_value.wrapping_sub(prev_value);
    if delta == 0 {
        zeros = zeros.saturating_add(1);
    } else {
        flush_pending_zeros(&mut accum, &mut bits, &mut zeros, &mut final_data);
        write_delta(&mut accum, &mut bits, delta, &mut final_data);
    }
    flush_pending_zeros(&mut accum, &mut bits, &mut zeros, &mut final_data);
    if bits > 0 && bits < 8 {
        final_data.push((accum << (8 - bits)) as u8);
    }

    let mut reader = BitReader::new(&final_data);
    let mut prev_val = first_val;
    let mut idx = 1u32;

    while result.len() < count && reader.has_more() {
        let ts = base_ts.wrapping_add(idx.wrapping_mul(interval));
        if reader.read_bits(1) == 0 {
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            prev_val = prev_val.wrapping_add(if reader.read_bits(1) == 0 { 1 } else { -1 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            prev_val = prev_val.wrapping_add(if reader.read_bits(1) == 0 { 2 } else { -2 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            for _ in 0..reader.read_bits(4) + 8 {
                if result.len() >= count { break; }
                let ts = base_ts.wrapping_add(idx.wrapping_mul(interval));
                result.push(Reading { ts, value: V::from_i32(prev_val) });
                idx = idx.wrapping_add(1);
            }
            continue;
        }
        if reader.read_bits(1) == 0 {
            for _ in 0..reader.read_bits(7) + 22 {
                if result.len() >= count { break; }
                let ts = base_ts.wrapping_add(idx.wrapping_mul(interval));
                result.push(Reading { ts, value: V::from_i32(prev_val) });
                idx = idx.wrapping_add(1);
            }
            continue;
        }
        if reader.read_bits(1) == 0 {
            let e = reader.read_bits(4) as i32;
            prev_val = prev_val.wrapping_add(if e < 8 { e - 10 } else { e - 5 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
            continue;
        }
        if reader.read_bits(1) == 0 {
            let raw = reader.read_bits(11);
            prev_val = prev_val.wrapping_add(if raw & 0x400 != 0 { (raw | 0xFFFF_F800) as i32 } else { raw as i32 });
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.wrapping_add(1);
        } else {
            idx = idx.wrapping_add(reader.read_bits(6) + 2);
        }
    }

    result
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
    let idx = delta.wrapping_add(10) as usize;
    let (b, n) = if idx <= 20 {
        let (b, n) = DELTA_ENCODE[idx];
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

#[inline]
fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
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
