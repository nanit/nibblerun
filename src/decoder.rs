//! Decoding functionality for nibblerun encoded data.

use crate::constants::EPOCH_BASE;
use crate::reading::Reading;
use crate::value::Value;

/// Push multiple readings with the same value (zero-run decoding)
/// Optimized to avoid per-iteration bounds checks and use running timestamp
#[inline]
fn push_zero_run<V: Value, const INTERVAL: u16>(
    decoded: &mut Vec<Reading<V>>,
    count: usize,
    temp: V,
    ts: &mut u64,
    run_len: u32,
) {
    // Calculate how many we can actually push
    let remaining = count.saturating_sub(decoded.len());
    let actual_len = (run_len as usize).min(remaining);

    // Reserve capacity once
    decoded.reserve(actual_len);

    let interval = u64::from(INTERVAL);
    // Push without per-iteration capacity checks
    for _ in 0..actual_len {
        decoded.push(Reading {
            ts: *ts,
            value: temp,
        });
        *ts += interval;
    }
}

/// Decode `NibbleRun` bytes back to readings
///
/// # Type Parameters
/// * `V` - Value type (i8, i16, or i32). Must match the type used during encoding.
/// * `INTERVAL` - The interval in seconds (must match encoder's interval)
///
/// # Arguments
/// * `bytes` - Encoded bytes from `Encoder::to_bytes()`
///
/// # Returns
/// Vector of decoded readings. Returns an empty vector if bytes is too short
/// or contains no readings.
#[must_use]
pub fn decode<V: Value, const INTERVAL: u16>(bytes: &[u8]) -> Vec<Reading<V>> {
    let interval = u64::from(INTERVAL);
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
    // Use running timestamp to avoid multiplication on every iteration
    let mut ts = start_ts + interval;

    // Encoding scheme:
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
            // 0 = zero delta (most common case - ~89% of data)
            decoded.push(Reading {
                ts,
                value: V::from_i32(prev_temp),
            });
            ts += interval;
            continue;
        }
        if reader.read_bits(1) == 0 {
            // 10x = ±1 delta
            prev_temp = prev_temp.wrapping_add(if reader.read_bits(1) == 0 { 1 } else { -1 });
            decoded.push(Reading {
                ts,
                value: V::from_i32(prev_temp),
            });
            ts += interval;
            continue;
        }
        // 11...
        if reader.read_bits(1) == 0 {
            // 110 = single-interval gap
            ts += interval;
            continue;
        }
        // 111...
        if reader.read_bits(1) == 0 {
            // 1110x = ±2 delta
            prev_temp = prev_temp.wrapping_add(if reader.read_bits(1) == 0 { 2 } else { -2 });
            decoded.push(Reading {
                ts,
                value: V::from_i32(prev_temp),
            });
            ts += interval;
            continue;
        }
        // 1111...
        if reader.read_bits(1) == 0 {
            // 11110xxxx = zero run 8-21
            push_zero_run::<V, INTERVAL>(&mut decoded, count, V::from_i32(prev_temp), &mut ts, reader.read_bits(4) + 8);
            continue;
        }
        // 11111...
        if reader.read_bits(1) == 0 {
            // 111110xxxxxxx = zero run 22-149
            push_zero_run::<V, INTERVAL>(&mut decoded, count, V::from_i32(prev_temp), &mut ts, reader.read_bits(7) + 22);
            continue;
        }
        // 111111...
        if reader.read_bits(1) == 0 {
            // 1111110xxxx = ±3-10 delta
            let e = reader.read_bits(4) as i32;
            prev_temp = prev_temp.wrapping_add(if e < 8 { e - 10 } else { e - 5 });
            decoded.push(Reading {
                ts,
                value: V::from_i32(prev_temp),
            });
            ts += interval;
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
                ts,
                value: V::from_i32(prev_temp),
            });
            ts += interval;
        } else {
            // 11111111xxxxxx = gap 2-65 intervals
            let gap = u64::from(reader.read_bits(6) + 2);
            ts += gap * interval;
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
    const fn has_more(&self) -> bool {
        self.left > 0 || self.pos < self.buf.len()
    }
}
