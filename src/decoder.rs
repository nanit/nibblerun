//! Decoding functionality for nibblerun encoded data.

use crate::constants::EPOCH_BASE;
use crate::reading::Reading;
use crate::value::Value;

/// Push multiple readings with the same value (zero-run decoding)
#[inline]
fn push_zero_run<V: Value>(
    decoded: &mut Vec<Reading<V>>,
    count: usize,
    start_ts: u64,
    interval: u64,
    temp: V,
    idx: &mut u64,
    run_len: u32,
) {
    for _ in 0..run_len {
        if decoded.len() >= count {
            break;
        }
        decoded.push(Reading {
            ts: start_ts + *idx * interval,
            value: temp,
        });
        *idx += 1;
    }
}

/// Decode `NibbleRun` bytes back to readings
///
/// # Type Parameters
/// * `V` - Value type (i8, i16, or i32). Must match the type used during encoding.
///
/// # Arguments
/// * `bytes` - Encoded bytes from `Encoder::to_bytes()`
/// * `interval` - The interval in seconds used when encoding (must match encoder's interval)
///
/// # Returns
/// Vector of decoded readings. Returns an empty vector if bytes is too short
/// or contains no readings.
#[must_use]
pub fn decode<V: Value>(bytes: &[u8], interval: u64) -> Vec<Reading<V>> {
    let header_size = 4 + 2 + V::BYTES; // base_ts_offset + count + first_value

    if bytes.len() < header_size {
        return Vec::new();
    }

    // Header layout:
    // [0-3]: base_ts_offset (4 bytes)
    // [4-5]: count (2 bytes)
    // [6..6+V::BYTES]: first_value (V::BYTES bytes)
    let base_ts_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let start_ts = EPOCH_BASE + u64::from(base_ts_offset);
    let count = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
    let first_temp = V::read_le(&bytes[6..6 + V::BYTES]);

    let mut decoded = Vec::with_capacity(count);
    if count == 0 {
        return decoded;
    }

    decoded.push(Reading {
        ts: start_ts,
        value: first_temp,
    });
    if count == 1 || bytes.len() <= header_size {
        return decoded;
    }

    let mut reader = BitReader::new(&bytes[header_size..]);
    let mut prev_temp = first_temp.to_i32();
    let mut idx = 1u64;

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
                ts: start_ts + idx * interval,
                value: V::from_i32(prev_temp),
            });
            idx += 1;
            continue;
        }
        if reader.read_bits(1) == 0 {
            // 10x = ±1 delta
            prev_temp = prev_temp.wrapping_add(if reader.read_bits(1) == 0 { 1 } else { -1 });
            decoded.push(Reading {
                ts: start_ts + idx * interval,
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
            prev_temp = prev_temp.wrapping_add(if reader.read_bits(1) == 0 { 2 } else { -2 });
            decoded.push(Reading {
                ts: start_ts + idx * interval,
                value: V::from_i32(prev_temp),
            });
            idx += 1;
            continue;
        }
        // 1111...
        if reader.read_bits(1) == 0 {
            // 11110xxxx = zero run 8-21
            push_zero_run(&mut decoded, count, start_ts, interval, V::from_i32(prev_temp), &mut idx, reader.read_bits(4) + 8);
            continue;
        }
        // 11111...
        if reader.read_bits(1) == 0 {
            // 111110xxxxxxx = zero run 22-149
            push_zero_run(&mut decoded, count, start_ts, interval, V::from_i32(prev_temp), &mut idx, reader.read_bits(7) + 22);
            continue;
        }
        // 111111...
        if reader.read_bits(1) == 0 {
            // 1111110xxxx = ±3-10 delta
            let e = reader.read_bits(4) as i32;
            prev_temp = prev_temp.wrapping_add(if e < 8 { e - 10 } else { e - 5 });
            decoded.push(Reading {
                ts: start_ts + idx * interval,
                value: V::from_i32(prev_temp),
            });
            idx += 1;
            continue;
        }
        // 1111111...
        if reader.read_bits(1) == 0 {
            // 11111110xxxxxxxxxxx = large delta (±11-1023)
            let raw = reader.read_bits(11);
            let delta = if raw & 0x400 != 0 {
                (raw | 0xFFFF_F800) as i32
            } else {
                raw as i32
            };
            prev_temp = prev_temp.wrapping_add(delta);
            decoded.push(Reading {
                ts: start_ts + idx * interval,
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

/// Bit reader for decoding variable-length bit sequences
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
