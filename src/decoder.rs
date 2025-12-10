//! Decoding functionality for nibblerun encoded data.

use crate::constants::{EPOCH_BASE, HEADER_SIZE};
use crate::reading::Reading;

/// Decode `NibbleRun` bytes back to readings
///
/// # Arguments
/// * `bytes` - Encoded bytes from `Encoder::to_bytes()`
///
/// # Returns
/// Vector of decoded readings. Returns an empty vector if bytes is too short
/// (less than 14 bytes) or contains no readings.
#[must_use]
pub fn decode(bytes: &[u8]) -> Vec<Reading> {
    if bytes.len() < HEADER_SIZE {
        return Vec::new();
    }

    let base_ts_offset = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let start_ts = EPOCH_BASE + u64::from(base_ts_offset);
    let count = u16::from_le_bytes(bytes[6..8].try_into().unwrap()) as usize;
    let first_temp = i32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let interval = u64::from(u16::from_le_bytes(bytes[12..14].try_into().unwrap()));

    let mut decoded = Vec::with_capacity(count);
    if count == 0 {
        return decoded;
    }

    decoded.push(Reading {
        ts: start_ts,
        temperature: first_temp,
    });
    if count == 1 || bytes.len() <= HEADER_SIZE {
        return decoded;
    }

    let mut reader = BitReader::new(&bytes[HEADER_SIZE..]);
    let mut prev_temp = first_temp;
    let mut idx = 1u64;

    while decoded.len() < count && reader.has_more() {
        if reader.read_bits(1) == 0 {
            // Single zero: 0
            decoded.push(Reading {
                ts: start_ts + idx * interval,
                temperature: prev_temp,
            });
            idx += 1;
            continue;
        }
        if reader.read_bits(1) == 0 {
            // ±1: 10 + sign
            prev_temp = prev_temp.wrapping_add(if reader.read_bits(1) == 0 { 1 } else { -1 });
            decoded.push(Reading {
                ts: start_ts + idx * interval,
                temperature: prev_temp,
            });
            idx += 1;
            continue;
        }
        if reader.read_bits(1) == 0 {
            // Zero run 2-5: 110 + 2 bits
            for _ in 0..reader.read_bits(2) + 2 {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: start_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
            }
            continue;
        }
        if reader.read_bits(1) == 0 {
            // Zero run 6-21: 1110 + 4 bits
            for _ in 0..reader.read_bits(4) + 6 {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: start_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
            }
            continue;
        }
        if reader.read_bits(1) == 0 {
            // Zero run 22-149: 11110 + 7 bits
            for _ in 0..reader.read_bits(7) + 22 {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: start_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
            }
            continue;
        }
        if reader.read_bits(1) == 0 {
            // ±2: 111110 + sign
            prev_temp = prev_temp.wrapping_add(if reader.read_bits(1) == 0 { 2 } else { -2 });
            decoded.push(Reading {
                ts: start_ts + idx * interval,
                temperature: prev_temp,
            });
            idx += 1;
            continue;
        }
        if reader.read_bits(1) == 0 {
            // ±3-10: 1111110 + 4 bits
            let e = reader.read_bits(4) as i32;
            prev_temp = prev_temp.wrapping_add(if e < 8 { e - 10 } else { e - 5 });
            decoded.push(Reading {
                ts: start_ts + idx * interval,
                temperature: prev_temp,
            });
            idx += 1;
            continue;
        }
        if reader.read_bits(1) == 0 {
            // Large delta: 11111110 + 11 bits signed
            let raw = reader.read_bits(11);
            let delta = if raw & 0x400 != 0 {
                (raw | 0xFFFF_F800) as i32
            } else {
                raw as i32
            };
            prev_temp = prev_temp.wrapping_add(delta);
            decoded.push(Reading {
                ts: start_ts + idx * interval,
                temperature: prev_temp,
            });
            idx += 1;
        } else {
            // Gap marker: 11111111 + 6 bits
            idx += u64::from(reader.read_bits(6) + 1);
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
