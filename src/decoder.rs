//! Decoding functionality for nibblerun encoded data.

use crate::error::DecodeError;
use crate::reading::Reading;
use crate::value::Value;

/// Frozen format header size (base_ts: 4 + count: 2 + first_value: V::BYTES)
const fn frozen_header_size<V: Value>() -> usize {
    4 + 2 + V::BYTES
}

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
/// * `Ok(Vec<Reading<V>>)` - Vector of decoded readings
/// * `Err(DecodeError::BufferTooShort)` - Buffer is too short to contain valid header
///
/// # Example
/// ```
/// use nibblerun::{Encoder, decode_frozen};
///
/// let mut enc = Encoder::<i32, 300>::new();
/// enc.append(1000, 22).unwrap();
/// enc.append(1300, 23).unwrap();
///
/// let frozen = enc.freeze();
/// let readings = decode_frozen::<i32, 300>(&frozen).unwrap();
/// assert_eq!(readings.len(), 2);
/// ```
#[must_use = "decoding returns readings that should be used"]
pub fn decode_frozen<V: Value, const INTERVAL: u16>(buf: &[u8]) -> Result<Vec<Reading<V>>, DecodeError> {
    let header_size = frozen_header_size::<V>();

    if buf.len() < header_size {
        if buf.is_empty() {
            return Ok(Vec::new());
        }
        return Err(DecodeError::BufferTooShort { expected: header_size, actual: buf.len() });
    }

    let base_ts = read_u32_le(buf, 0)?;
    let count = read_u16_le(buf, 4)? as usize;
    let first_value = V::read_le(&buf[6..]).to_i32();

    if count == 0 {
        return Ok(Vec::new());
    }

    let interval = u32::from(INTERVAL);
    let mut result = Vec::with_capacity(count);
    result.push(Reading { ts: base_ts, value: V::from_i32(first_value) });

    if count == 1 {
        return Ok(result);
    }

    let data = &buf[header_size..];
    let mut reader = BitReader::new(data);

    decode_bitstream::<V>(&mut reader, &mut result, base_ts, interval, count, first_value)?;

    Ok(result)
}

/// Decode bitstream into readings
///
/// Shared decode logic used by both `decode_frozen` and `Encoder::decode`.
/// This function processes the variable-length bit codes and appends readings to the result.
///
/// Returns `Err(DecodeError::MalformedData)` if arithmetic overflow is detected.
#[inline]
pub(crate) fn decode_bitstream<V: Value>(
    reader: &mut BitReader<'_>,
    result: &mut Vec<Reading<V>>,
    base_ts: u32,
    interval: u32,
    count: usize,
    first_value: i32,
) -> Result<(), DecodeError> {
    let mut prev_val = first_value;
    let mut idx = 1u32;

    while result.len() < count && reader.has_more() {
        let ts = compute_ts(base_ts, idx, interval)?;

        // 0 = repeat previous value
        if reader.read_bits(1) == 0 {
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
            continue;
        }

        // 10x = delta ±1
        if reader.read_bits(1) == 0 {
            let delta = if reader.read_bits(1) == 0 { 1 } else { -1 };
            prev_val = prev_val.checked_add(delta).ok_or(DecodeError::MalformedData)?;
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
            continue;
        }

        // 110 = single gap (skip one interval)
        if reader.read_bits(1) == 0 {
            idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
            continue;
        }

        // 1110x = delta ±2
        if reader.read_bits(1) == 0 {
            let delta = if reader.read_bits(1) == 0 { 2 } else { -2 };
            prev_val = prev_val.checked_add(delta).ok_or(DecodeError::MalformedData)?;
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
            continue;
        }

        // 11110xxxx = zero run (8-23 repeats)
        if reader.read_bits(1) == 0 {
            let run_len = reader.read_bits(4).saturating_add(8);
            for _ in 0..run_len {
                if result.len() >= count { break; }
                let ts = compute_ts(base_ts, idx, interval)?;
                result.push(Reading { ts, value: V::from_i32(prev_val) });
                idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
            }
            continue;
        }

        // 111110xxxxxxx = zero run (22-149 repeats)
        if reader.read_bits(1) == 0 {
            let run_len = reader.read_bits(7).saturating_add(22);
            for _ in 0..run_len {
                if result.len() >= count { break; }
                let ts = compute_ts(base_ts, idx, interval)?;
                result.push(Reading { ts, value: V::from_i32(prev_val) });
                idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
            }
            continue;
        }

        // 1111110xxxx = small delta [-10, -3] or [3, 10]
        if reader.read_bits(1) == 0 {
            let e = reader.read_bits(4) as i32;
            let delta = if e < 8 { e - 10 } else { e - 5 };
            prev_val = prev_val.checked_add(delta).ok_or(DecodeError::MalformedData)?;
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
            continue;
        }

        // 11111110xxxxxxxxxxx = large delta (11-bit signed)
        if reader.read_bits(1) == 0 {
            let raw = reader.read_bits(11);
            // Sign extend from 11 bits
            let delta = if raw & 0x400 != 0 { (raw | 0xFFFF_F800) as i32 } else { raw as i32 };
            prev_val = prev_val.checked_add(delta).ok_or(DecodeError::MalformedData)?;
            result.push(Reading { ts, value: V::from_i32(prev_val) });
            idx = idx.checked_add(1).ok_or(DecodeError::MalformedData)?;
        } else {
            // 11111111xxxxxx = multi-gap (skip 2-65 intervals)
            let gap = reader.read_bits(6).saturating_add(2);
            idx = idx.checked_add(gap).ok_or(DecodeError::MalformedData)?;
        }
    }

    Ok(())
}

/// Compute timestamp with overflow checking
#[inline]
fn compute_ts(base_ts: u32, idx: u32, interval: u32) -> Result<u32, DecodeError> {
    let offset = idx.checked_mul(interval).ok_or(DecodeError::MalformedData)?;
    base_ts.checked_add(offset).ok_or(DecodeError::MalformedData)
}

// Helper functions

/// Read a u16 from buffer in little-endian format
#[inline]
fn read_u16_le(buf: &[u8], offset: usize) -> Result<u16, DecodeError> {
    let end = offset.saturating_add(2);
    if end > buf.len() {
        return Err(DecodeError::BufferTooShort { expected: end, actual: buf.len() });
    }
    Ok(u16::from_le_bytes([buf[offset], buf[offset + 1]]))
}

/// Read a u32 from buffer in little-endian format
#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> Result<u32, DecodeError> {
    let end = offset.saturating_add(4);
    if end > buf.len() {
        return Err(DecodeError::BufferTooShort { expected: end, actual: buf.len() });
    }
    Ok(u32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]]))
}

/// Bit reader for decoding variable-length bit codes
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    bits: u64,
    left: u32,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        let mut r = Self { buf, pos: 0, bits: 0, left: 0 };
        r.refill();
        r
    }

    /// Refill the bit buffer from the byte buffer
    #[inline]
    fn refill(&mut self) {
        while self.left <= 56 && self.pos < self.buf.len() {
            self.bits = (self.bits << 8) | u64::from(self.buf[self.pos]);
            self.pos += 1;
            self.left = self.left.saturating_add(8);
        }
    }

    /// Read n bits from the buffer (max 32 bits)
    ///
    /// Returns 0 if not enough bits available.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32, "cannot read more than 32 bits at a time");
        if self.left < n { self.refill(); }
        if self.left < n { return 0; }
        self.left -= n;
        // Safe: n <= 32, so 1 << n won't overflow
        ((self.bits >> self.left) & ((1u64 << n) - 1)) as u32
    }

    /// Check if there are more bits to read
    #[inline]
    pub fn has_more(&self) -> bool {
        self.left > 0 || self.pos < self.buf.len()
    }
}
