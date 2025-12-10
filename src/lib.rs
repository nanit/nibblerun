//! `NibbleRun` - High-performance time series compression for slow-changing data
//!
//! A bit-packed compression format optimized for sensor data that changes gradually,
//! such as temperature, humidity, or similar environmental readings. Achieves ~70-85x
//! compression with O(1) append operations.
//!
//! # Features
//! - **High compression**: ~40-50 bytes per day for typical sensor data
//! - **Fast encoding**: ~250M inserts/sec single-threaded
//! - **O(1) append**: Add new readings without re-encoding
//! - **Configurable intervals**: Timestamps quantized to fixed intervals (default 5 min)
//!
//! # Lossy vs Lossless
//!
//! The compression is **lossless when there is exactly one reading per interval**.
//! When multiple readings fall within the same interval, they are **averaged together**,
//! which is lossy. This is intentional - for sensor data sampled more frequently than
//! the storage interval, averaging provides a representative value while maintaining
//! the fixed interval structure.
//!
//! # Example
//! ```
//! use nibblerun::{Encoder, decode};
//!
//! let mut encoder = Encoder::new();
//! let base_ts = 1_761_000_000_u64;
//!
//! // Append sensor readings (timestamp, value)
//! encoder.append(base_ts, 22).unwrap();
//! encoder.append(base_ts + 300, 22).unwrap();  // 5 minutes later
//! encoder.append(base_ts + 600, 23).unwrap();  // value changed
//!
//! // Serialize to bytes
//! let bytes = encoder.to_bytes();
//! println!("Encoded size: {} bytes", bytes.len());
//!
//! // Decode back
//! let readings = decode(&bytes);
//! for r in &readings {
//!     println!("ts={}, temp={}", r.ts, r.temperature);
//! }
//! ```
//!
//! # Wire Format
//!
//! ## Header (14 bytes)
//!
//! | Offset | Size | Field | Description |
//! |--------|------|-------|-------------|
//! | 0 | 4 | `base_ts_offset` | First timestamp minus epoch base (1,760,000,000). Reconstructed as `epoch_base + offset`. |
//! | 4 | 2 | `duration` | Number of intervals from first to last reading. Metadata for quick time-span queries without decoding. |
//! | 6 | 2 | `count` | Total number of readings stored. Used by decoder to know when to stop. |
//! | 8 | 4 | `first_temp` | First value as i32, stored directly (not delta-encoded). |
//! | 12 | 2 | `interval` | Interval between readings in seconds (1-65535). |
//!
//! ## Bit-Packed Data
//!
//! After the header, values are stored as variable-length bit-packed deltas:
//!
//! | Delta | Encoding | Bits | Description |
//! |-------|----------|------|-------------|
//! | 0 | `0` | 1 | Single unchanged value |
//! | 0 (run) | `110xx` | 5 | 2-5 consecutive unchanged values |
//! | 0 (run) | `1110xxxx` | 8 | 6-21 consecutive unchanged values |
//! | 0 (run) | `11110xxxxxxx` | 12 | 22-149 consecutive unchanged values |
//! | ±1 | `10x` | 3 | x=0 for +1, x=1 for -1 |
//! | ±2 | `111110x` | 7 | x=0 for +2, x=1 for -2 |
//! | ±3..±10 | `1111110xxxx` | 11 | 4-bit signed offset from ±3 |
//! | ±11..±1023 | `11111110xxxxxxxxxxx` | 19 | 11-bit signed value |
//! | gap | `11111111xxxxxx` | 14 | Skip 1-64 intervals (no data). Larger gaps use multiple markers. |
//!
//! # Internal Implementation
//!
//! ## Bit Accumulator
//!
//! Bits are accumulated in a 64-bit register (`bit_accum`) and flushed to the output
//! buffer in 8-bit chunks when full. This avoids byte-alignment overhead and allows
//! efficient variable-length encoding.
//!
//! ## Pending State Packing
//!
//! To keep the `Encoder` struct compact (72 bytes), multiple values are packed into
//! a single `u64` field (`pending_state`):
//! - Bits 0-5: Current bit accumulator count (0-63)
//! - Bits 6-15: Pending reading count for averaging (0-1023)
//! - Bits 16-47: Pending sum for averaging (i32 range)
//!
//! This allows in-interval averaging without additional struct fields.
//!
//! ## Zero-Run Encoding
//!
//! Consecutive unchanged values are encoded using run-length encoding with tiered
//! prefix codes. A single `0` bit means "same as previous". Longer runs use progressively
//! longer codes to encode the run length, up to 149 values per code. Runs longer than
//! 149 use multiple codes.
//!
//! ## Gap Encoding
//!
//! When intervals have no data (sensor offline, gaps in collection), a special 14-bit
//! gap marker encodes up to 64 skipped intervals. This is more efficient than storing
//! placeholder values and preserves the actual timing of readings.
//!
//! ## Supported Ranges
//! - Values: full i32 range (first value), ±1023 per delta
//! - Readings per encoder: up to 65,535
//! - Readings per interval: up to 1,023 (averaged together)
//! - Timestamp intervals: 1-65,535 seconds (~18 hours max)

use serde::{Deserialize, Serialize};
use std::fmt;

/// Error returned when appending a reading fails
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError {
    /// Timestamp is before the base timestamp (first reading's timestamp)
    TimestampBeforeBase { ts: u64, base_ts: u64 },
    /// Timestamp would place reading in an earlier interval (out of order)
    OutOfOrder {
        ts: u64,
        logical_idx: u32,
        prev_logical_idx: u32,
    },
    /// Too many readings in the same interval (max 1023)
    IntervalOverflow { count: u16 },
    /// Too many total readings (max 65535)
    CountOverflow,
    /// Temperature delta exceeds encodable range (must be in [-1024, 1023])
    DeltaOverflow {
        delta: i32,
        prev_temp: i32,
        new_temp: i32,
    },
}

impl fmt::Display for AppendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimestampBeforeBase { ts, base_ts } => {
                write!(f, "timestamp {ts} is before base timestamp {base_ts}")
            }
            Self::OutOfOrder {
                ts,
                logical_idx,
                prev_logical_idx,
            } => {
                write!(
                    f,
                    "timestamp {ts} (interval {logical_idx}) is before previous interval {prev_logical_idx}"
                )
            }
            Self::IntervalOverflow { count } => {
                write!(f, "too many readings in interval ({count}), max is 1023")
            }
            Self::CountOverflow => write!(f, "too many total readings, max is 65535"),
            Self::DeltaOverflow {
                delta,
                prev_temp,
                new_temp,
            } => {
                write!(
                    f,
                    "temperature delta {delta} ({prev_temp} -> {new_temp}) exceeds range [-1024, 1023]"
                )
            }
        }
    }
}

impl std::error::Error for AppendError {}

// Branch hints using #[cold] attribute (stable Rust)
#[cold]
#[inline(never)]
fn cold_gap_handler() {}

/// Base epoch for timestamp compression (reduces storage by ~4 bytes)
const EPOCH_BASE: u64 = 1_760_000_000;

/// Default interval between readings (5 minutes)
pub const DEFAULT_INTERVAL: u64 = 300;

/// Header size in bytes (4 + 2 + 2 + 4 + 2 = 14)
const HEADER_SIZE: usize = 14;

/// Division by interval
#[inline]
fn div_by_interval(x: u64, interval: u16) -> u64 {
    x / u64::from(interval)
}

/// Compute average with proper rounding (round half away from zero)
/// This ensures the average is always within [min, max] of the input values
#[inline]
fn rounded_avg(sum: i32, count: u16) -> i32 {
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
fn pack_pending(bits: u32, count: u16, sum: i32) -> u64 {
    (u64::from(sum as u32) << 16) | ((u64::from(count) & 0x3FF) << 6) | (u64::from(bits) & 0x3F)
}

/// Unpack pending averaging state from `pending_state`
/// Returns (`bit_accum_count`, `pending_count`, `pending_sum`)
#[inline]
fn unpack_pending(packed: u64) -> (u32, u16, i32) {
    let bits = (packed & 0x3F) as u32;
    let count = ((packed >> 6) & 0x3FF) as u16;
    let sum = (packed >> 16) as u32 as i32;
    (bits, count, sum)
}

// Precomputed delta encoding table: (bits, num_bits) for deltas -10 to +10
const DELTA_ENCODE: [(u32, u8); 21] = [
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

/// Encoder for `NibbleRun` format
///
/// Accumulates temperature readings and produces compressed output.
/// Optimized for cache efficiency with compact struct layout.
#[repr(C)]
#[derive(Clone, Serialize, Deserialize)]
pub struct Encoder {
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
}

impl Encoder {
    /// Create a new encoder with default 300-second (5-minute) interval
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_interval(DEFAULT_INTERVAL as u16)
    }

    /// Create a new encoder with a custom interval
    ///
    /// # Arguments
    /// * `interval` - Interval between readings in seconds (e.g., 300 for 5 minutes)
    #[inline]
    #[must_use]
    pub fn with_interval(interval: u16) -> Self {
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
        }
    }

    /// Get the interval in seconds
    #[inline]
    #[must_use]
    pub fn interval(&self) -> u16 {
        self.interval
    }

    /// Append a temperature reading
    ///
    /// Multiple readings in the same interval are averaged.
    ///
    /// # Arguments
    /// * `ts` - Unix timestamp in seconds
    /// * `temperature` - Temperature value
    ///
    /// # Errors
    /// Returns an error if:
    /// - Timestamp is before the base timestamp
    /// - Timestamp is out of order (earlier interval than previous)
    /// - Too many readings in the same interval (max 1023)
    /// - Too many total readings (max 65535)
    /// - Temperature delta exceeds encodable range [-1024, 1023]
    #[inline]
    pub fn append(&mut self, ts: u64, temperature: i32) -> Result<(), AppendError> {
        // First reading - initialize pending state
        if self.count == 0 {
            self.base_ts = ts;
            self.last_ts = ts;
            self.first_temp = temperature;
            self.prev_temp = temperature;
            self.prev_logical_idx = 0;
            self.count = 1;
            // Initialize pending: count=1, sum=temperature
            self.pending_state = pack_pending(0, 1, temperature);
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
            self.pending_state = pack_pending(bits, count + 1, sum.saturating_add(temperature));
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
                    prev_temp: self.prev_temp,
                    new_temp: avg,
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
        // Initialize pending for new interval: count=1, sum=temperature
        let (bits, _, _) = unpack_pending(self.pending_state);
        self.pending_state = pack_pending(bits, 1, temperature);

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

        // Extract pending state
        let (actual_bits, pending_count, pending_sum) = unpack_pending(self.pending_state);

        // For single interval, no additional bits needed
        if self.count == 1 {
            return HEADER_SIZE;
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

        HEADER_SIZE + self.data.len() + (total_bits as usize).div_ceil(8)
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
        19 // Large delta encoding
    }

    /// Calculate the number of bits needed to encode a zero run
    #[inline]
    fn zero_run_bits(mut n: u32) -> u32 {
        let mut bits = 0;
        while n > 0 {
            if n == 1 {
                bits += 1;
                n = 0;
            } else if n <= 5 {
                bits += 5;
                n = 0;
            } else if n <= 21 {
                bits += 8;
                n = 0;
            } else if n <= 149 {
                bits += 12;
                n = 0;
            } else {
                bits += 12;
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
    pub fn decode(&self) -> Vec<Reading> {
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
            temperature: first_temp,
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
        while decoded.len() < count && reader.has_more() {
            if reader.read_bits(1) == 0 {
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                prev_temp += if reader.read_bits(1) == 0 { 1 } else { -1 };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                for _ in 0..reader.read_bits(2) + 2 {
                    if decoded.len() >= count {
                        break;
                    }
                    decoded.push(Reading {
                        ts: self.base_ts + idx * interval,
                        temperature: prev_temp,
                    });
                    idx += 1;
                }
                continue;
            }
            if reader.read_bits(1) == 0 {
                for _ in 0..reader.read_bits(4) + 6 {
                    if decoded.len() >= count {
                        break;
                    }
                    decoded.push(Reading {
                        ts: self.base_ts + idx * interval,
                        temperature: prev_temp,
                    });
                    idx += 1;
                }
                continue;
            }
            if reader.read_bits(1) == 0 {
                for _ in 0..reader.read_bits(7) + 22 {
                    if decoded.len() >= count {
                        break;
                    }
                    decoded.push(Reading {
                        ts: self.base_ts + idx * interval,
                        temperature: prev_temp,
                    });
                    idx += 1;
                }
                continue;
            }
            if reader.read_bits(1) == 0 {
                prev_temp += if reader.read_bits(1) == 0 { 2 } else { -2 };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                let e = reader.read_bits(4) as i32;
                prev_temp += if e < 8 { e - 10 } else { e - 5 };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
                continue;
            }
            if reader.read_bits(1) == 0 {
                let raw = reader.read_bits(11);
                prev_temp += if raw & 0x400 != 0 {
                    (raw | 0xFFFF_F800) as i32
                } else {
                    raw as i32
                };
                decoded.push(Reading {
                    ts: self.base_ts + idx * interval,
                    temperature: prev_temp,
                });
                idx += 1;
            } else {
                idx += u64::from(reader.read_bits(6) + 1);
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

        let mut result = Vec::with_capacity(HEADER_SIZE + self.data.len() + 4);

        // Header
        let base_ts_offset = (self.base_ts - EPOCH_BASE) as u32;
        result.extend_from_slice(&base_ts_offset.to_le_bytes());
        let duration = div_by_interval(self.last_ts - self.base_ts, self.interval) as u16;
        result.extend_from_slice(&duration.to_le_bytes());
        result.extend_from_slice(&self.count.to_le_bytes());
        result.extend_from_slice(&first_temp.to_le_bytes());
        result.extend_from_slice(&self.interval.to_le_bytes());

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
        while count > 0 {
            let g = count.min(64);
            self.write_bits((0xFF << 6) | (g - 1), 14);
            count -= g;
        }
    }
}

#[inline]
fn encode_zero_run(n: u32) -> (u32, u32, u32) {
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

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A decoded temperature reading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading {
    /// Unix timestamp in seconds
    pub ts: u64,
    /// Temperature value
    pub temperature: i32,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_by_interval() {
        for x in [0, 1, 299, 300, 301, 599, 600, 1000, 10000, 100000, 200000] {
            assert_eq!(div_by_interval(x, 300), x / 300, "failed for x={}", x);
        }
    }

    #[test]
    fn test_roundtrip() {
        let base = 1761955455u64;
        let temps = [22, 23, 23, 22, 21, 22, 22, 22, 25, 20];
        let mut enc = Encoder::new();
        for (i, &t) in temps.iter().enumerate() {
            enc.append(base + i as u64 * 300, t).unwrap();
        }
        let dec = enc.decode();
        assert_eq!(dec.len(), temps.len());
        for (i, r) in dec.iter().enumerate() {
            assert_eq!(r.temperature, temps[i]);
        }
    }

    #[test]
    fn test_empty() {
        let enc = Encoder::new();
        assert_eq!(enc.count(), 0);
        assert!(enc.to_bytes().is_empty());
    }

    #[test]
    fn test_single_reading() {
        let mut enc = Encoder::new();
        enc.append(1761955455, 22).unwrap();
        let dec = enc.decode();
        assert_eq!(dec.len(), 1);
        assert_eq!(dec[0].temperature, 22);
    }

    #[test]
    fn test_gaps() {
        // Gaps are implicit - just skip intervals by using later timestamps
        let base = 1761955455u64;
        let mut enc = Encoder::new();
        enc.append(base, 22).unwrap();
        // Skip 2 intervals (600 seconds = 2 * 300)
        enc.append(base + 900, 23).unwrap();
        let dec = enc.decode();
        assert_eq!(dec.len(), 2);
        assert_eq!(dec[0].temperature, 22);
        assert_eq!(dec[1].temperature, 23);
        // Gap is preserved in timestamps
        assert_eq!(dec[1].ts - dec[0].ts, 900);
    }

    #[test]
    fn test_long_run() {
        let base = 1761955455u64;
        let mut enc = Encoder::new();
        for i in 0..200 {
            enc.append(base + i * 300, 22).unwrap();
        }
        let dec = enc.decode();
        assert_eq!(dec.len(), 200);
        for r in &dec {
            assert_eq!(r.temperature, 22);
        }
    }

    #[test]
    fn test_all_deltas() {
        let base = 1761955455u64;
        let mut enc = Encoder::new();
        let mut temp = 70;
        let mut temps = vec![temp];
        for d in -10i32..=10 {
            if d == 0 {
                continue;
            }
            temp += d;
            temps.push(temp);
        }
        temps.push(temp + 50);
        temps.push(temp);

        for (i, &t) in temps.iter().enumerate() {
            enc.append(base + i as u64 * 300, t).unwrap();
        }
        let dec = enc.decode();
        assert_eq!(dec.len(), temps.len());
        for (i, r) in dec.iter().enumerate() {
            assert_eq!(r.temperature, temps[i], "mismatch at {}", i);
        }
    }

    #[test]
    fn test_temp_range_25_to_39() {
        let base = 1761955455u64;
        let mut enc = Encoder::new();
        let mut temps: Vec<i32> = (25..=39).collect();
        temps.extend((25..39).rev());

        for (i, &t) in temps.iter().enumerate() {
            enc.append(base + i as u64 * 300, t).unwrap();
        }

        let dec = enc.decode();

        assert_eq!(dec.len(), temps.len(), "count mismatch");
        for (i, r) in dec.iter().enumerate() {
            assert_eq!(r.temperature, temps[i], "mismatch at {}", i);
        }
    }

    #[test]
    fn test_temp_range_neg10_to_39() {
        let base = 1761955455u64;
        let mut enc = Encoder::new();
        let temps: Vec<i32> = (-10..=39).collect();

        for (i, &t) in temps.iter().enumerate() {
            enc.append(base + i as u64 * 300, t).unwrap();
        }

        let dec = enc.decode();

        assert_eq!(dec.len(), temps.len(), "count mismatch");
        for (i, r) in dec.iter().enumerate() {
            assert_eq!(r.temperature, temps[i], "mismatch at {}", i);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let base = 1761955455u64;
        let mut enc = Encoder::new();

        // Simulate typical day: mostly constant with occasional changes
        let mut temp = 22;
        for i in 0..288 {
            // 288 = 24 hours * 12 (5-min intervals)
            if i == 50 {
                temp = 23;
            }
            if i == 150 {
                temp = 22;
            }
            enc.append(base + i * 300, temp).unwrap();
        }

        let bytes = enc.to_bytes();
        // Raw would be 288 * 12 = 3456 bytes
        // NibbleRun should be ~40-50 bytes
        assert!(bytes.len() < 60, "encoded size {} too large", bytes.len());
        assert!(bytes.len() > 10, "encoded size {} too small", bytes.len());
    }

    #[test]
    fn test_constant_temperature() {
        let mut encoder = Encoder::new();
        let base_ts = 1761955455u64;

        for i in 0..10 {
            encoder.append(base_ts + i * 300, 22).unwrap();
        }

        let decoded = encoder.decode();

        assert_eq!(decoded.len(), 10);
        for reading in &decoded {
            assert_eq!(reading.temperature, 22);
        }
    }

    #[test]
    fn test_small_deltas() {
        let mut encoder = Encoder::new();
        let base_ts = 1761955455u64;
        let temps = [22, 23, 22, 21, 22];

        for (i, &temp) in temps.iter().enumerate() {
            encoder.append(base_ts + i as u64 * 300, temp).unwrap();
        }

        let decoded = encoder.decode();

        assert_eq!(decoded.len(), 5);
        for (i, reading) in decoded.iter().enumerate() {
            assert_eq!(reading.temperature, temps[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_medium_delta() {
        let mut encoder = Encoder::new();
        let base_ts = 1761955455u64;

        encoder.append(base_ts, 20).unwrap();
        encoder.append(base_ts + 300, 25).unwrap(); // +5
        encoder.append(base_ts + 600, 20).unwrap(); // -5

        let decoded = encoder.decode();

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].temperature, 20);
        assert_eq!(decoded[1].temperature, 25);
        assert_eq!(decoded[2].temperature, 20);
    }

    #[test]
    fn test_large_delta() {
        let mut encoder = Encoder::new();
        let base_ts = 1761955455u64;

        encoder.append(base_ts, 20).unwrap();
        encoder.append(base_ts + 300, 520).unwrap(); // +500
        encoder.append(base_ts + 600, 20).unwrap(); // -500

        let decoded = encoder.decode();

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].temperature, 20);
        assert_eq!(decoded[1].temperature, 520);
        assert_eq!(decoded[2].temperature, 20);
    }

    #[test]
    fn test_long_zero_run() {
        let mut encoder = Encoder::new();
        let base_ts = 1761955455u64;

        for i in 0..50 {
            encoder.append(base_ts + i * 300, 22).unwrap();
        }

        let decoded = encoder.decode();

        assert_eq!(decoded.len(), 50);
        for reading in &decoded {
            assert_eq!(reading.temperature, 22);
        }
    }

    #[test]
    fn test_with_timestamp_jitter() {
        // Simulate real-world sensor readings with small positive jitter
        // (negative jitter could cause readings to fall into previous interval)
        let base_ts = 1761955455u64;
        let temps = [22, 23, 23, 22, 21, 21, 22, 23, 22, 21];

        // Positive jitter only to ensure each reading lands in its expected interval
        let jitter = [0u64, 3, 2, 5, 5, 1, 3, 4, 1, 2];

        let mut encoder = Encoder::new();
        for (i, (&temp, &j)) in temps.iter().zip(jitter.iter()).enumerate() {
            let ts = base_ts + (i as u64 * 300) + j;
            encoder.append(ts, temp).unwrap();
        }

        let decoded = encoder.decode();

        // All readings should be preserved
        assert_eq!(decoded.len(), 10);

        // Verify temperatures match and timestamps are quantized
        for (i, reading) in decoded.iter().enumerate() {
            assert_eq!(
                reading.temperature, temps[i],
                "temp mismatch at index {}",
                i
            );
            // Timestamp should be aligned to 300-second intervals from base_ts
            assert_eq!(
                (reading.ts - base_ts) % 300,
                0,
                "ts {} not aligned to interval at index {}",
                reading.ts,
                i
            );
        }
    }

    #[test]
    fn test_with_larger_timestamp_jitter() {
        // Test with jitter that stays within the same interval when quantized
        // Using jitter small enough that (i*300 + jitter) / 300 = i
        let base_ts = 1761955455u64;
        let temps = [22, 23, 24, 25, 26];

        // Jitter values carefully chosen so logical index = i
        // For idx i: need (i*300 + jitter) / 300 = i
        // This means: i*300 <= i*300 + jitter < (i+1)*300
        // So: 0 <= jitter < 300 for positive
        let jitter = [0i64, 50, 100, 149, 200];

        let mut encoder = Encoder::new();
        for (i, (&temp, &j)) in temps.iter().zip(jitter.iter()).enumerate() {
            let ts = (base_ts as i64 + (i as i64 * 300) + j) as u64;
            encoder.append(ts, temp).unwrap();
        }

        let decoded = encoder.decode();

        assert_eq!(decoded.len(), 5);

        // Expected: each reading maps to logical index i, so ts = base_ts + i*300
        let expected: [(u64, i32); 5] = [
            (base_ts + 0 * 300, 22),
            (base_ts + 1 * 300, 23),
            (base_ts + 2 * 300, 24),
            (base_ts + 3 * 300, 25),
            (base_ts + 4 * 300, 26),
        ];

        for (i, reading) in decoded.iter().enumerate() {
            assert_eq!(reading.ts, expected[i].0, "ts mismatch at index {}", i);
            assert_eq!(
                reading.temperature, expected[i].1,
                "temp mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_specific_day_with_jitter() {
        // 2025-12-01 00:00:00 UTC = 1764547200
        let day_start = 1764547200u64;

        // Input readings with realistic jitter:
        // 00:00:03 -> temp 25 (logical idx 0)
        // 00:05:10 -> temp 25 (logical idx 1)
        // 00:10:20 -> temp 26 (logical idx 2)
        // 00:15:02 -> temp 25 (logical idx 2 - SKIPPED, same as previous)
        // 01:35:05 -> temp 26 (logical idx 19)
        let inputs: [(u64, i32); 5] = [
            (day_start + 0 * 60 + 3, 25),   // 00:00:03
            (day_start + 5 * 60 + 10, 25),  // 00:05:10
            (day_start + 10 * 60 + 20, 26), // 00:10:20
            (day_start + 15 * 60 + 2, 25),  // 00:15:02 - will be skipped
            (day_start + 95 * 60 + 5, 26),  // 01:35:05
        ];

        let mut encoder = Encoder::new();
        for (ts, temp) in inputs {
            encoder.append(ts, temp).unwrap();
        }

        let decoded = encoder.decode();

        // Only 4 readings - the 4th input is skipped (same logical index as 3rd)
        assert_eq!(decoded.len(), 4);

        // Input analysis (base_ts = day_start + 3 = 1764547203):
        // Reading 0: ts=1764547203, logical_idx=0
        // Reading 1: ts=1764547510, logical_idx=1
        // Reading 2: ts=1764547820, logical_idx=2
        // Reading 3: ts=1764548102, logical_idx=2 -> SKIPPED (duplicate)
        // Reading 4: ts=1764552905, logical_idx=19

        let base_ts = inputs[0].0;

        let expected: [(u64, i32); 4] = [
            (base_ts + 0 * 300, 25),  // logical idx 0
            (base_ts + 1 * 300, 25),  // logical idx 1
            (base_ts + 2 * 300, 26),  // logical idx 2
            (base_ts + 19 * 300, 26), // logical idx 19 (gap of 17 from idx 2)
        ];

        for (i, reading) in decoded.iter().enumerate() {
            assert_eq!(
                reading.ts, expected[i].0,
                "ts mismatch at index {}: got {}, expected {}",
                i, reading.ts, expected[i].0
            );
            assert_eq!(
                reading.temperature, expected[i].1,
                "temp mismatch at index {}: got {}, expected {}",
                i, reading.temperature, expected[i].1
            );
        }

        // Verify the large gap between readings 2 and 3
        assert_eq!(
            decoded[3].ts - decoded[2].ts,
            17 * 300,
            "gap between readings 2 and 3 should be 17 intervals (5100 seconds)"
        );
    }

    #[test]
    fn test_out_of_order_readings_return_error() {
        let base_ts = 1761955455u64;
        let mut encoder = Encoder::new();

        // First reading sets base_ts
        encoder.append(base_ts + 600, 24).unwrap(); // base_ts is set to base_ts + 600
        let actual_base = base_ts + 600;

        // Out-of-order readings return errors
        assert_eq!(
            encoder.append(base_ts, 22),
            Err(AppendError::TimestampBeforeBase {
                ts: base_ts,
                base_ts: actual_base
            })
        );
        assert_eq!(
            encoder.append(base_ts + 300, 23),
            Err(AppendError::TimestampBeforeBase {
                ts: base_ts + 300,
                base_ts: actual_base
            })
        );

        // Reading at a later interval is accepted
        encoder.append(base_ts + 900, 25).unwrap();

        let decoded = encoder.decode();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, 24);
        assert_eq!(decoded[1].temperature, 25);
    }

    #[test]
    fn test_reading_before_base_ts_returns_error() {
        let base_ts = 1761955455u64;
        let mut encoder = Encoder::new();

        // First reading establishes base_ts
        encoder.append(base_ts, 22).unwrap();
        assert_eq!(encoder.count(), 1);

        // Reading before base_ts should return TimestampBeforeBase error
        assert_eq!(
            encoder.append(base_ts - 1, 99),
            Err(AppendError::TimestampBeforeBase {
                ts: base_ts - 1,
                base_ts
            })
        );
        assert_eq!(encoder.count(), 1);

        assert_eq!(
            encoder.append(base_ts - 100, 99),
            Err(AppendError::TimestampBeforeBase {
                ts: base_ts - 100,
                base_ts
            })
        );
        assert_eq!(encoder.count(), 1);

        assert_eq!(
            encoder.append(base_ts - 300, 99),
            Err(AppendError::TimestampBeforeBase {
                ts: base_ts - 300,
                base_ts
            })
        );
        assert_eq!(encoder.count(), 1);

        // Reading at or after base_ts should be accepted
        encoder.append(base_ts + 300, 23).unwrap();
        assert_eq!(encoder.count(), 2);

        let decoded = encoder.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, 22);
        assert_eq!(decoded[1].temperature, 23);
    }

    #[test]
    fn test_reading_before_epoch_base_as_first() {
        // If the first reading has ts < EPOCH_BASE, to_bytes() would panic
        // due to underflow. This tests that we handle this edge case.
        let mut encoder = Encoder::new();

        // Timestamp before EPOCH_BASE (1_760_000_000)
        let old_ts = EPOCH_BASE - 1000;
        encoder.append(old_ts, 22).unwrap();

        // The encoder accepts it as first reading (base_ts = old_ts)
        assert_eq!(encoder.count(), 1);

        // But to_bytes() would underflow when computing base_ts - EPOCH_BASE
        // This is a known limitation: timestamps must be >= EPOCH_BASE
        // For now, we document that this panics
        let result = std::panic::catch_unwind(|| encoder.to_bytes());
        assert!(result.is_err(), "Expected panic when ts < EPOCH_BASE");
    }

    #[test]
    fn test_size_matches_to_bytes() {
        // Empty encoder
        let enc = Encoder::new();
        assert_eq!(enc.size(), enc.to_bytes().len());

        // Single reading
        let mut enc = Encoder::new();
        enc.append(1761955455, 22).unwrap();
        assert_eq!(enc.size(), enc.to_bytes().len());

        // Multiple readings with zero deltas (tests zero_run estimation)
        let mut enc = Encoder::new();
        for i in 0..10 {
            enc.append(1761955455 + i * 300, 22).unwrap();
        }
        assert_eq!(enc.size(), enc.to_bytes().len());

        // Readings with varying deltas
        let mut enc = Encoder::new();
        let temps = [22, 23, 21, 25, 20, 30, 15];
        for (i, &t) in temps.iter().enumerate() {
            enc.append(1761955455 + i as u64 * 300, t).unwrap();
        }
        assert_eq!(enc.size(), enc.to_bytes().len());

        // Long zero run (tests zero_run > 149)
        let mut enc = Encoder::new();
        for i in 0..200 {
            enc.append(1761955455 + i * 300, 22).unwrap();
        }
        assert_eq!(enc.size(), enc.to_bytes().len());

        // Mixed: some zeros, some deltas
        let mut enc = Encoder::new();
        for i in 0..50 {
            let temp = if i % 10 == 0 { 25 } else { 22 };
            enc.append(1761955455 + i * 300, temp).unwrap();
        }
        assert_eq!(enc.size(), enc.to_bytes().len());
    }

    #[test]
    fn test_size_incremental_with_jitter() {
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // Simple PRNG for deterministic jitter (no external deps)
        let mut seed: u32 = 12345;
        let mut next_jitter = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed % 100) as u64 // 0-99 seconds jitter
        };

        // Simple PRNG for temperature changes
        let mut temp_seed: u32 = 67890;
        let mut next_temp_delta = || {
            temp_seed = temp_seed.wrapping_mul(1103515245).wrapping_add(12345);
            match temp_seed % 10 {
                0 => 1,  // 10% chance +1
                1 => -1, // 10% chance -1
                2 => 2,  // 10% chance +2
                3 => -2, // 10% chance -2
                _ => 0,  // 60% chance no change
            }
        };

        let mut temp = 22i32;
        let mut prev_logical_idx = 0u64;

        for i in 0..300 {
            let jitter = next_jitter();
            let ts = base_ts + i * 300 + jitter;

            // Calculate logical index to check if this reading will be accepted
            let logical_idx = if enc.count() == 0 {
                0
            } else {
                (ts - base_ts) / 300
            };

            // Only update temp if reading will be accepted (not a duplicate)
            if enc.count() == 0 || logical_idx > prev_logical_idx {
                temp = (temp + next_temp_delta()).clamp(-50, 100);
                prev_logical_idx = logical_idx;
            }

            enc.append(ts, temp).unwrap();

            // Assert size matches actual encoded size after each append
            let actual_size = enc.to_bytes().len();
            let estimated_size = enc.size();

            assert_eq!(
                estimated_size, actual_size,
                "size mismatch at iteration {}: estimated={}, actual={}",
                i, estimated_size, actual_size
            );
        }
    }

    #[test]
    fn test_size_constant_temperature_incremental() {
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // Constant temperature - maximum compression via zero runs
        for i in 0..300 {
            enc.append(base_ts + i * 300, 22).unwrap();

            let actual_size = enc.to_bytes().len();
            let estimated_size = enc.size();

            assert_eq!(
                estimated_size, actual_size,
                "size mismatch at iteration {}: estimated={}, actual={}",
                i, estimated_size, actual_size
            );
        }
    }

    #[test]
    fn test_size_alternating_temperature_incremental() {
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // Alternating temperature - no zero runs, all ±1 deltas
        for i in 0..300 {
            let temp = if i % 2 == 0 { 22 } else { 23 };
            enc.append(base_ts + i * 300, temp).unwrap();

            let actual_size = enc.to_bytes().len();
            let estimated_size = enc.size();

            assert_eq!(
                estimated_size, actual_size,
                "size mismatch at iteration {}: estimated={}, actual={}",
                i, estimated_size, actual_size
            );
        }
    }

    #[test]
    fn test_size_large_deltas_incremental() {
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // Large temperature swings - tests large delta encoding
        for i in 0..300 {
            let temp = if i % 2 == 0 { 0 } else { 100 };
            enc.append(base_ts + i * 300, temp).unwrap();

            let actual_size = enc.to_bytes().len();
            let estimated_size = enc.size();

            assert_eq!(
                estimated_size, actual_size,
                "size mismatch at iteration {}: estimated={}, actual={}",
                i, estimated_size, actual_size
            );
        }
    }

    #[test]
    fn test_duplicate_day_events_return_error() {
        let base_ts = 1761955455u64;
        let mut encoder = Encoder::new();

        // Append a full day of events (288 readings at 5-min intervals)
        for i in 0..288 {
            encoder.append(base_ts + i * 300, 22).unwrap();
        }
        assert_eq!(encoder.count(), 288);

        // Trying to append the same day again returns OutOfOrder errors
        // (because they're in earlier intervals than the last one)
        for i in 0..287 {
            let result = encoder.append(base_ts + i * 300, 22);
            assert!(
                matches!(result, Err(AppendError::OutOfOrder { .. })),
                "Expected OutOfOrder error for i={}",
                i
            );
        }

        // The last one (i=287) would be in the same interval as current, so it's allowed
        // (same interval = averaging)
        encoder.append(base_ts + 287 * 300, 22).unwrap();

        // Should still be 288 (the extra one was averaged into the last interval)
        assert_eq!(encoder.count(), 288);
    }

    #[test]
    fn test_duplicate_day_events_with_different_timestamps_return_error() {
        let base_ts = 1761955455u64;
        let mut encoder = Encoder::new();

        // First pass: 288 events at start of each interval
        for i in 0..288 {
            encoder.append(base_ts + i * 300, 22).unwrap();
        }
        assert_eq!(encoder.count(), 288);

        // Second pass: trying to go back in time returns OutOfOrder errors
        for i in 0..287 {
            let ts = base_ts + i * 300 + 150; // 150 seconds into each interval
            let result = encoder.append(ts, 22);
            assert!(
                matches!(result, Err(AppendError::OutOfOrder { .. })),
                "Expected OutOfOrder error for i={}",
                i
            );
        }

        // Last interval (i=287) is current, so adding to it works
        encoder.append(base_ts + 287 * 300 + 150, 22).unwrap();
        assert_eq!(encoder.count(), 288);
    }

    #[test]
    fn test_duplicate_timestamps_averaged() {
        let base_ts = 1761955455u64;
        let mut encoder = Encoder::new();

        encoder.append(base_ts, 22).unwrap();
        encoder.append(base_ts, 23).unwrap(); // Same interval - will be averaged
        encoder.append(base_ts + 5, 24).unwrap(); // Same logical index (within same 300s interval)
        encoder.append(base_ts + 300, 25).unwrap(); // Next interval

        let decoded = encoder.decode();

        // Two intervals: first averaged (22+23+24)/3 = 23, second = 25
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, 23); // (22+23+24+1)/3 = 23 (round half up)
        assert_eq!(decoded[1].temperature, 25);
    }

    #[test]
    fn test_averaging_round_half_up() {
        let base_ts = 1761955455u64;

        // Test case: 22 + 23 = 45, (45 + 1) / 2 = 23 (rounds up)
        let mut encoder = Encoder::new();
        encoder.append(base_ts, 22).unwrap();
        encoder.append(base_ts + 1, 23).unwrap();
        let decoded = encoder.decode();
        assert_eq!(decoded[0].temperature, 23);

        // Test case: 22 + 22 + 23 = 67, (67 + 1) / 3 = 22 (rounds down)
        let mut encoder = Encoder::new();
        encoder.append(base_ts, 22).unwrap();
        encoder.append(base_ts + 1, 22).unwrap();
        encoder.append(base_ts + 2, 23).unwrap();
        let decoded = encoder.decode();
        assert_eq!(decoded[0].temperature, 22);

        // Test case: 20 + 21 + 22 + 23 = 86, (86 + 2) / 4 = 22
        let mut encoder = Encoder::new();
        encoder.append(base_ts, 20).unwrap();
        encoder.append(base_ts + 1, 21).unwrap();
        encoder.append(base_ts + 2, 22).unwrap();
        encoder.append(base_ts + 3, 23).unwrap();
        let decoded = encoder.decode();
        assert_eq!(decoded[0].temperature, 22);

        // Test case: negative temperatures - (-16) + (-16) = -32, (-32 - 1) / 2 = -16
        // This tests that rounding works correctly for negative numbers
        let mut encoder = Encoder::new();
        encoder.append(base_ts, -16).unwrap();
        encoder.append(base_ts + 1, -16).unwrap();
        let decoded = encoder.decode();
        assert_eq!(decoded[0].temperature, -16);

        // Test case: negative with rounding - (-15) + (-16) = -31, (-31 - 1) / 2 = -16
        let mut encoder = Encoder::new();
        encoder.append(base_ts, -15).unwrap();
        encoder.append(base_ts + 1, -16).unwrap();
        let decoded = encoder.decode();
        assert_eq!(decoded[0].temperature, -16); // rounds away from zero
    }

    #[test]
    fn test_alternating_readings_same_interval_averaged() {
        let base_ts = 1761955455u64;
        let mut encoder = Encoder::new();

        // 10 readings alternating 25, 21 spread across 5 intervals (2 per interval)
        // Each interval: 25 + 21 = 46, (46 + 1) / 2 = 23 (round half up)
        let readings = [25, 21, 25, 21, 25, 21, 25, 21, 25, 21];
        for (i, &temp) in readings.iter().enumerate() {
            // 2 readings per 300s interval: readings 0,1 in interval 0, 2,3 in interval 1, etc.
            let interval = i / 2;
            let offset_within_interval = (i % 2) * 150; // 0 or 150 seconds
            encoder
                .append(
                    base_ts + (interval as u64) * 300 + offset_within_interval as u64,
                    temp,
                )
                .unwrap();
        }

        let decoded = encoder.decode();

        // Should be 5 readings, all with averaged value 23
        assert_eq!(
            decoded.len(),
            5,
            "expected 5 averaged readings, got {}",
            decoded.len()
        );
        for (i, reading) in decoded.iter().enumerate() {
            assert_eq!(
                reading.temperature, 23,
                "expected temp 23 at index {}, got {}",
                i, reading.temperature
            );
            assert_eq!(
                reading.ts,
                base_ts + (i as u64) * 300,
                "wrong timestamp at index {}",
                i
            );
        }

        // Size should be minimal: header (14 bytes) + zero-run encoding for 4 repeated values
        // First temp (23) is in header, then 4 zeros encoded as single zero-run
        // Zero run of 4 uses 5-bit encoding: 110xx (5 bits) = 1 byte when padded
        let size = encoder.size();
        assert_eq!(
            size, 15,
            "expected size of 15 bytes (header + 1 byte zero-run), got {}",
            size
        );
    }

    #[test]
    fn test_custom_interval() {
        let base_ts = 1761955455u64;

        // Test with 60-second interval
        let mut enc = Encoder::with_interval(60);
        assert_eq!(enc.interval(), 60);

        enc.append(base_ts, 22).unwrap();
        enc.append(base_ts + 60, 23).unwrap();
        enc.append(base_ts + 120, 24).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0].ts, base_ts);
        assert_eq!(decoded[1].ts, base_ts + 60);
        assert_eq!(decoded[2].ts, base_ts + 120);
        assert_eq!(decoded[0].temperature, 22);
        assert_eq!(decoded[1].temperature, 23);
        assert_eq!(decoded[2].temperature, 24);

        // Test roundtrip via bytes
        let bytes = enc.to_bytes();
        let decoded_bytes = decode(&bytes);
        assert_eq!(decoded_bytes.len(), 3);
        assert_eq!(decoded_bytes[1].ts, base_ts + 60);
    }

    #[test]
    fn test_custom_interval_averaging() {
        let base_ts = 1761955455u64;

        // Test averaging with 60-second interval
        let mut enc = Encoder::with_interval(60);

        // Two readings in same 60-second interval
        enc.append(base_ts, 20).unwrap();
        enc.append(base_ts + 30, 24).unwrap(); // Same interval, should average to 22

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].temperature, 22); // (20 + 24) / 2 = 22
    }

    #[test]
    fn test_single_reading_per_interval_exact() {
        // When exactly one reading falls in an interval, decoded value should equal input exactly
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        let temps = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
        for (i, &temp) in temps.iter().enumerate() {
            enc.append(base_ts + (i as u64) * 300, temp).unwrap();
        }

        let decoded = enc.decode();
        assert_eq!(decoded.len(), temps.len());
        for (i, reading) in decoded.iter().enumerate() {
            assert_eq!(
                reading.temperature, temps[i],
                "Single reading at interval {} should be exact: expected {}, got {}",
                i, temps[i], reading.temperature
            );
        }
    }

    #[test]
    fn test_max_readings_65535() {
        // Encode exactly 65535 readings (u16::MAX), verify roundtrip
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        for i in 0..65535u64 {
            let temp = ((i % 20) as i32) + 15; // Temps 15-34
            enc.append(base_ts + i * 300, temp).unwrap();
        }

        assert_eq!(enc.count(), 65535);

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 65535);

        // Verify via bytes roundtrip
        let bytes = enc.to_bytes();
        let decoded_bytes = decode(&bytes);
        assert_eq!(decoded_bytes.len(), 65535);
    }

    #[test]
    fn test_beyond_max_readings() {
        // Verify behavior when appending reading 65536+
        // Currently the encoder will panic on overflow - this documents that behavior
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // Fill to max
        for i in 0..65535u64 {
            enc.append(base_ts + i * 300, 22).unwrap();
        }
        assert_eq!(enc.count(), 65535);

        // Adding one more would cause overflow panic in debug mode
        // In release mode it would wrap to 0, causing corruption
        // This test documents that 65535 is the hard limit
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            enc.append(base_ts + 65535 * 300, 23).unwrap();
        }));
        // In debug mode, this panics; in release mode, it may not
        // Either way, we verify the encoder has 65535 readings
        if result.is_ok() {
            // If it didn't panic, count may have wrapped - check decode still works
            let decoded = enc.decode();
            assert!(decoded.len() <= 65535);
        }
    }

    #[test]
    fn test_extreme_temps_boundaries() {
        // Test i32 temperatures (now stored as full i32)
        // Note: first_temp can be any i32, but:
        // - deltas between readings are limited to ±1024
        // - averaging accumulator is limited to ±1M sum range
        let base_ts = 1761955455u64;

        // Test large negative first_temp (within ±1M for averaging)
        let mut enc = Encoder::new();
        enc.append(base_ts, -500_000).unwrap();
        enc.append(base_ts + 300, -499_500).unwrap(); // delta = +500
        let decoded = enc.decode();
        assert_eq!(decoded[0].temperature, -500_000);
        assert_eq!(decoded[1].temperature, -499_500);

        // Test large positive first_temp
        let mut enc = Encoder::new();
        enc.append(base_ts, 500_000).unwrap();
        enc.append(base_ts + 300, 500_500).unwrap(); // delta = +500
        let decoded = enc.decode();
        assert_eq!(decoded[0].temperature, 500_000);
        assert_eq!(decoded[1].temperature, 500_500);

        // Test a sequence with various temps, all within ±1024 delta of each other
        let mut enc = Encoder::new();
        let temps = [-1000, -500, 0, 500, 1000, 500, 0, -500, -1000];
        for (i, &temp) in temps.iter().enumerate() {
            enc.append(base_ts + (i as u64) * 300, temp).unwrap();
        }

        let decoded = enc.decode();
        assert_eq!(decoded.len(), temps.len());
        for (i, reading) in decoded.iter().enumerate() {
            assert_eq!(
                reading.temperature, temps[i],
                "Extreme temp at index {}: expected {}, got {}",
                i, temps[i], reading.temperature
            );
        }

        // Test maximum delta range (±1023, since ±1024 is the limit)
        let mut enc2 = Encoder::new();
        enc2.append(base_ts, 0).unwrap();
        enc2.append(base_ts + 300, 1023).unwrap(); // delta = +1023
        enc2.append(base_ts + 600, 0).unwrap(); // delta = -1023

        let decoded2 = enc2.decode();
        assert_eq!(decoded2[0].temperature, 0);
        assert_eq!(decoded2[1].temperature, 1023);
        assert_eq!(decoded2[2].temperature, 0);

        // Verify roundtrip via bytes works for large temperatures
        let mut enc3 = Encoder::new();
        enc3.append(base_ts, 100_000).unwrap();
        enc3.append(base_ts + 300, 100_500).unwrap();
        let bytes = enc3.to_bytes();
        let decoded3 = decode(&bytes);
        assert_eq!(decoded3[0].temperature, 100_000);
        assert_eq!(decoded3[1].temperature, 100_500);
    }

    #[test]
    fn test_interval_1_second() {
        // interval = 1, readings every second
        let base_ts = 1761955455u64;
        let mut enc = Encoder::with_interval(1);

        for i in 0..100u64 {
            enc.append(base_ts + i, 22 + (i % 5) as i32).unwrap();
        }

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 100);

        // Verify timestamps are 1 second apart
        for window in decoded.windows(2) {
            assert_eq!(window[1].ts - window[0].ts, 1);
        }
    }

    #[test]
    fn test_interval_65535_seconds() {
        // interval = 65535 (~18 hours)
        let base_ts = 1761955455u64;
        let mut enc = Encoder::with_interval(65535);

        enc.append(base_ts, 22).unwrap();
        enc.append(base_ts + 65535, 23).unwrap();
        enc.append(base_ts + 65535 * 2, 24).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[1].ts - decoded[0].ts, 65535);
        assert_eq!(decoded[2].ts - decoded[1].ts, 65535);
    }

    #[test]
    fn test_zero_run_tier_boundaries() {
        // Zero-run encoding tiers:
        // - Single zero: 1 bit (prefix 0)
        // - 2-5 zeros: 5 bits (prefix 110 + 2 bits)
        // - 6-21 zeros: 8 bits (prefix 1110 + 4 bits)
        // - 22-149 zeros: 12 bits (prefix 11110 + 7 bits)

        let base_ts = 1761955455u64;

        // Test exactly 1 zero (single zero encoding)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        enc.append(base_ts + 300, 22).unwrap(); // 1 zero delta
        enc.append(base_ts + 600, 23).unwrap();
        let decoded = enc.decode();
        assert_eq!(decoded.len(), 3);

        // Test exactly 5 zeros (boundary of 2-5 tier)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        for i in 1..=5 {
            enc.append(base_ts + i * 300, 22).unwrap(); // 5 zeros
        }
        enc.append(base_ts + 6 * 300, 23).unwrap();
        let decoded = enc.decode();
        assert_eq!(decoded.len(), 7);

        // Test exactly 21 zeros (boundary of 6-21 tier)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        for i in 1..=21 {
            enc.append(base_ts + i * 300, 22).unwrap(); // 21 zeros
        }
        enc.append(base_ts + 22 * 300, 23).unwrap();
        let decoded = enc.decode();
        assert_eq!(decoded.len(), 23);

        // Test exactly 149 zeros (boundary of 22-149 tier)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        for i in 1..=149 {
            enc.append(base_ts + i * 300, 22).unwrap(); // 149 zeros
        }
        enc.append(base_ts + 150 * 300, 23).unwrap();
        let decoded = enc.decode();
        assert_eq!(decoded.len(), 151);

        // Test 150 zeros (exceeds single run, needs 2 encodings)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        for i in 1..=150 {
            enc.append(base_ts + i * 300, 22).unwrap(); // 150 zeros
        }
        enc.append(base_ts + 151 * 300, 23).unwrap();
        let decoded = enc.decode();
        assert_eq!(decoded.len(), 152);
    }

    #[test]
    fn test_gap_encoding_boundaries() {
        // Gap marker: 11111111 + 6 bits = up to 64 gaps per marker
        // Gaps are implicit from timestamp jumps
        let base_ts = 1761955455u64;

        // Test gap of exactly 64 intervals (max per single marker)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        enc.append(base_ts + 65 * 300, 23).unwrap(); // Skip 64 intervals

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].ts, base_ts);
        assert_eq!(decoded[1].ts, base_ts + 65 * 300);

        // Test gap of 65 (requires 2 gap markers: 64 + 1)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        enc.append(base_ts + 66 * 300, 23).unwrap(); // Skip 65 intervals

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[1].ts, base_ts + 66 * 300);

        // Test gap of 128 (requires 2 gap markers: 64 + 64)
        let mut enc = Encoder::new();
        enc.append(base_ts, 22).unwrap();
        enc.append(base_ts + 129 * 300, 24).unwrap(); // Skip 128 intervals

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[1].ts, base_ts + 129 * 300);
    }

    #[test]
    fn test_large_timestamp_offset() {
        // Test timestamps where ts - base_ts is large
        // With 300s interval, offset = gap_intervals * 300
        // Max practical gap is limited by gap encoding (64 per marker, ~19200s per marker)
        let base_ts = EPOCH_BASE + 1_000_000; // Well after epoch base
        let mut enc = Encoder::new();

        enc.append(base_ts, 22).unwrap();

        // Use offset that's multiple of interval and reasonable for gap encoding
        // 1000 intervals * 300s = 300,000 seconds (~3.5 days)
        let large_offset = 1000u64 * 300;
        enc.append(base_ts + large_offset, 23).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        // The gap should be exactly large_offset / interval = 1000 intervals
        // But encoded timestamps are quantized to interval
        let expected_ts_diff = large_offset;
        assert_eq!(decoded[1].ts - decoded[0].ts, expected_ts_diff);
    }

    #[test]
    fn test_decode_truncated_header() {
        // < 14 bytes should return empty vec (header is now 14 bytes)
        assert!(decode(&[]).is_empty());
        assert!(decode(&[0]).is_empty());
        assert!(decode(&[0; 13]).is_empty());

        // Exactly 14 bytes is valid header
        let mut header = [0u8; 14];
        // Set count to 0 - should return empty vec
        let decoded = decode(&header);
        assert!(decoded.is_empty());

        // Set count to 1, first_temp as i32
        header[6] = 1; // count = 1
        header[7] = 0;
        // first_temp = 22 as i32 little-endian (bytes 8-11)
        header[8] = 22;
        header[9] = 0;
        header[10] = 0;
        header[11] = 0;
        // interval = 300 as u16 little-endian (bytes 12-13)
        header[12] = 44; // 300 low byte
        header[13] = 1; // 300 high byte
        let decoded = decode(&header);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].temperature, 22);
    }

    #[test]
    fn test_decode_corrupted_count() {
        // count field larger than actual data - should not panic
        let mut enc = Encoder::new();
        enc.append(1761955455, 22).unwrap();
        enc.append(1761955455 + 300, 23).unwrap();

        let mut bytes = enc.to_bytes();

        // Corrupt count field to be larger
        bytes[6] = 255; // count = 255 but only 2 readings encoded
        bytes[7] = 0;

        // Should not panic, may return partial data
        let decoded = decode(&bytes);
        // Behavior: decode will try to read more than available, but should handle gracefully
        assert!(decoded.len() <= 255);
    }

    #[test]
    fn test_decode_zero_interval() {
        // interval = 0 in header - edge case
        let mut header = [0u8; 14];
        header[6] = 1; // count = 1
        // first_temp = 22 as i32 little-endian (bytes 8-11)
        header[8] = 22;
        header[9] = 0;
        header[10] = 0;
        header[11] = 0;
        // interval = 0 as u16 little-endian (bytes 12-13)
        header[12] = 0; // interval = 0 (low byte)
        header[13] = 0; // interval = 0 (high byte)

        // Should handle gracefully (interval 0 would cause div-by-zero if not handled)
        let decoded = decode(&header);
        // With interval=0, behavior depends on implementation
        // At minimum, should not panic
        assert!(decoded.len() <= 1);
    }

    #[test]
    fn test_31_readings_same_interval() {
        // Test 31 readings in the same interval (legacy test, still valid)
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // 31 readings in same interval
        for i in 0..31 {
            enc.append(base_ts + i * 5, 20 + (i as i32 % 10)).unwrap(); // Temps 20-29
        }

        // Move to next interval
        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);

        // Average of 31 readings with temps 20-29 (repeating 3x + 20)
        // Sum = (20+21+22+23+24+25+26+27+28+29) * 3 + 20 = 245 * 3 + 20 = 755
        // Avg = 755 / 31 = 24.35... ≈ 24
        let expected_avg = (0..31).map(|i| 20 + (i % 10)).sum::<i32>() / 31;
        assert_eq!(decoded[0].temperature, expected_avg);
    }

    #[test]
    fn test_32_readings_same_interval() {
        // 32 readings now works (new limit is 1023)
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // 32 readings in same interval
        for i in 0..32 {
            enc.append(base_ts + i * 5, 20).unwrap();
        }

        // Move to next interval
        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        // All 32 readings are now included in the average
        assert_eq!(decoded[0].temperature, 20);
    }

    #[test]
    fn test_100_readings_same_interval() {
        // Test 100 readings in the same interval
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // 100 readings in same interval, alternating 20 and 30
        for i in 0..100 {
            let temp = if i % 2 == 0 { 20 } else { 30 };
            enc.append(base_ts + i, temp).unwrap();
        }

        // Move to next interval
        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        // Average of 50 x 20 + 50 x 30 = 1000 + 1500 = 2500 / 100 = 25
        assert_eq!(decoded[0].temperature, 25);
    }

    #[test]
    fn test_500_readings_same_interval() {
        // Test 500 readings in the same interval
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // 500 readings in same interval
        for i in 0..500 {
            enc.append(base_ts + i, 22).unwrap();
        }

        // Move to next interval
        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, 22);
    }

    #[test]
    fn test_1023_readings_same_interval() {
        // Test max pending_count = 1023 (10 bits)
        // Use 1-second interval so all 1023 readings fit in one interval
        let base_ts = 1761955455u64;
        let mut enc = Encoder::with_interval(2000); // 2000 second interval

        // 1023 readings in same interval (all within first 1023 seconds)
        for i in 0..1023 {
            enc.append(base_ts + i, 20 + (i as i32 % 10)).unwrap(); // Temps 20-29
        }

        // Move to next interval
        enc.append(base_ts + 2000, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);

        // Average of 1023 readings with temps 20-29 (102 complete cycles + partial)
        let expected_sum: i32 = (0..1023).map(|i| 20 + (i % 10)).sum();
        let expected_avg = expected_sum / 1023;
        assert_eq!(decoded[0].temperature, expected_avg);
    }

    #[test]
    fn test_1024_readings_same_interval() {
        // 1024 readings exceeds pending_count max of 1023 - 1024th should return error
        let base_ts = 1761955455u64;
        let mut enc = Encoder::with_interval(2000); // 2000 second interval

        // 1023 readings in same interval (max allowed)
        for i in 0..1023 {
            enc.append(base_ts + i, 20).unwrap();
        }
        // This 1024th reading should return IntervalOverflow error
        assert!(matches!(
            enc.append(base_ts + 1023, 100),
            Err(AppendError::IntervalOverflow { count: 1023 })
        ));

        // Move to next interval - should work since previous was rejected
        enc.append(base_ts + 2000, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        // Average should be 20, the rejected 100 was not included
        assert_eq!(decoded[0].temperature, 20);
    }

    #[test]
    fn test_high_count_with_large_temps() {
        // Test that sum doesn't overflow with many readings of large temps
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // 500 readings of temperature 500 (sum = 250,000)
        for i in 0..500 {
            enc.append(base_ts + i, 500).unwrap();
        }

        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, 500);
    }

    #[test]
    fn test_high_count_with_negative_temps() {
        // Test averaging with many negative temperatures
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // 500 readings of temperature -500 (sum = -250,000)
        for i in 0..500 {
            enc.append(base_ts + i, -500).unwrap();
        }

        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, -500);
    }

    #[test]
    fn test_high_count_mixed_temps() {
        // Test averaging with mix of positive and negative temps
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // 400 readings: alternating -100 and +100
        for i in 0..400 {
            let temp = if i % 2 == 0 { -100 } else { 100 };
            enc.append(base_ts + i, temp).unwrap();
        }

        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        // Average of 200 x -100 + 200 x 100 = 0
        assert_eq!(decoded[0].temperature, 0);
    }

    #[test]
    fn test_averaging_to_zero() {
        // positive + negative temps that average to 0
        let base_ts = 1761955455u64;
        let mut enc = Encoder::new();

        // -10 and +10 average to 0
        enc.append(base_ts, -10).unwrap();
        enc.append(base_ts + 1, 10).unwrap();
        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, 0);

        // -50, -30, +40, +40 = sum 0, avg 0
        let mut enc = Encoder::new();
        enc.append(base_ts, -50).unwrap();
        enc.append(base_ts + 1, -30).unwrap();
        enc.append(base_ts + 2, 40).unwrap();
        enc.append(base_ts + 3, 40).unwrap();
        enc.append(base_ts + 300, 25).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].temperature, 0);
    }

    #[test]
    fn test_delta_overflow_returns_error() {
        // Delta > 1023 or < -1024 should return DeltaOverflow error
        // Note: Error is returned during finalize_pending_interval,
        // which is called when crossing from one interval to another.
        // We need at least 4 intervals to trigger this:
        // - interval 0: establishes base
        // - interval 1: first delta (from interval 0)
        // - interval 2: large temp that will overflow
        // - interval 3: triggers finalize of interval 2, returning error
        let base_ts = 1761955455u64;

        // Test positive delta overflow: 0 to 2000 = delta of +2000 (out of range)
        let mut enc = Encoder::new();
        enc.append(base_ts, 0).unwrap(); // interval 0
        enc.append(base_ts + 300, 0).unwrap(); // interval 1
        enc.append(base_ts + 600, 2000).unwrap(); // interval 2: temp=2000
        // interval 3: triggers finalize(2), delta=2000
        assert!(matches!(
            enc.append(base_ts + 900, 0),
            Err(AppendError::DeltaOverflow {
                delta: 2000,
                prev_temp: 0,
                new_temp: 2000
            })
        ));

        // Test negative delta overflow: 2000 to 0 = delta of -2000 (out of range)
        let mut enc = Encoder::new();
        enc.append(base_ts, 2000).unwrap(); // interval 0
        enc.append(base_ts + 300, 2000).unwrap(); // interval 1
        enc.append(base_ts + 600, 0).unwrap(); // interval 2: temp=0
        // interval 3: triggers finalize(2), delta=-2000
        assert!(matches!(
            enc.append(base_ts + 900, 0),
            Err(AppendError::DeltaOverflow {
                delta: -2000,
                prev_temp: 2000,
                new_temp: 0
            })
        ));

        // Test boundary: delta of exactly +1024 should return error
        let mut enc = Encoder::new();
        enc.append(base_ts, 0).unwrap();
        enc.append(base_ts + 300, 0).unwrap();
        enc.append(base_ts + 600, 1024).unwrap(); // delta will be +1024
        assert!(matches!(
            enc.append(base_ts + 900, 0),
            Err(AppendError::DeltaOverflow {
                delta: 1024,
                prev_temp: 0,
                new_temp: 1024
            })
        ));

        // Test boundary: delta of exactly -1025 should return error
        let mut enc = Encoder::new();
        enc.append(base_ts, 1025).unwrap();
        enc.append(base_ts + 300, 1025).unwrap();
        enc.append(base_ts + 600, 0).unwrap(); // delta will be -1025
        assert!(matches!(
            enc.append(base_ts + 900, 0),
            Err(AppendError::DeltaOverflow {
                delta: -1025,
                prev_temp: 1025,
                new_temp: 0
            })
        ));

        // Test boundary: delta of exactly +1023 should NOT return error
        let mut enc = Encoder::new();
        enc.append(base_ts, 0).unwrap();
        enc.append(base_ts + 300, 0).unwrap();
        enc.append(base_ts + 600, 1023).unwrap(); // delta will be +1023
        enc.append(base_ts + 900, 0).unwrap(); // Should succeed

        // Test boundary: delta of exactly -1024 should NOT return error
        let mut enc = Encoder::new();
        enc.append(base_ts, 1024).unwrap();
        enc.append(base_ts + 300, 1024).unwrap();
        enc.append(base_ts + 600, 0).unwrap(); // delta will be -1024
        enc.append(base_ts + 900, 0).unwrap(); // Should succeed
    }

    #[test]
    fn test_interval_zero_encoder() {
        // Creating encoder with interval=0 is questionable but shouldn't crash during creation
        // The behavior during append may vary (division by zero)
        let enc = Encoder::with_interval(0);
        assert_eq!(enc.count(), 0);
        // Don't try to append - would cause division by zero
    }

    #[test]
    fn test_max_interval_65535() {
        // Test maximum interval value
        let base_ts = 1761955455u64;
        let mut enc = Encoder::with_interval(65535);

        enc.append(base_ts, 22).unwrap();
        enc.append(base_ts + 65535, 23).unwrap(); // Exactly one interval later
        enc.append(base_ts + 65535 * 2, 24).unwrap(); // Two intervals later

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[1].ts - decoded[0].ts, 65535);
        assert_eq!(decoded[2].ts - decoded[1].ts, 65535);
    }

    #[test]
    fn test_timestamp_at_epoch_base() {
        // Timestamp exactly at EPOCH_BASE should work
        let mut enc = Encoder::new();
        enc.append(1_760_000_000, 22).unwrap(); // Exactly EPOCH_BASE
        enc.append(1_760_000_300, 23).unwrap();

        let decoded = enc.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].ts, 1_760_000_000);
    }

    #[test]
    fn test_count_at_max_u16() {
        // Test that count() returns correct value at max
        let base_ts = 1761955455u64;
        let mut enc = Encoder::with_interval(1);

        // Add exactly 65535 readings
        for i in 0..65535u64 {
            enc.append(base_ts + i, 22).unwrap();
        }

        assert_eq!(enc.count(), 65535);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Start well after EPOCH_BASE to ensure negative jitter doesn't underflow
    const BASE_TS: u64 = 1_760_100_000;

    prop_compose! {
        /// Generate a sequence of readings with jitter relative to the interval
        /// Jitter is ±10% of the interval
        fn arb_readings()(
            interval in prop::sample::select(vec![60u16, 300, 600, 3600]),
            count in 0usize..500,
        )(
            jitters in prop::collection::vec(-(interval as i64 / 10)..=(interval as i64 / 10), count),
            temps in prop::collection::vec(-100i32..140, count),
            interval in Just(interval),
        ) -> (Vec<(u64, i32)>, u16) {
            let readings: Vec<(u64, i32)> = jitters.iter().zip(temps.iter()).enumerate()
                .map(|(i, (&jitter, &temp))| {
                    let nominal_ts = BASE_TS + (i as u64) * (interval as u64);
                    let jittered_ts = (nominal_ts as i64 + jitter).max(0) as u64;
                    (jittered_ts, temp)
                })
                .collect();
            (readings, interval)
        }
    }

    proptest! {
        /// Property: size() must always equal to_bytes().len()
        #[test]
        fn prop_size_accuracy((readings, interval) in arb_readings()) {
            let mut enc = Encoder::with_interval(interval);
            for (ts, temp) in readings {
                enc.append(ts, temp).unwrap();
            }
            prop_assert_eq!(enc.size(), enc.to_bytes().len());
        }

        /// Property: decoded length must equal count()
        #[test]
        fn prop_count_consistency((readings, interval) in arb_readings()) {
            let mut enc = Encoder::with_interval(interval);
            for (ts, temp) in readings {
                enc.append(ts, temp).unwrap();
            }
            let decoded = enc.decode();
            prop_assert_eq!(decoded.len(), enc.count());
        }

        /// Property: encode then decode via bytes equals direct decode
        #[test]
        fn prop_roundtrip_via_bytes((readings, interval) in arb_readings()) {
            let mut enc = Encoder::with_interval(interval);
            for (ts, temp) in readings {
                enc.append(ts, temp).unwrap();
            }

            let direct = enc.decode();
            let via_bytes = decode(&enc.to_bytes());

            prop_assert_eq!(direct.len(), via_bytes.len());
            for (d, b) in direct.iter().zip(via_bytes.iter()) {
                prop_assert_eq!(d.ts, b.ts);
                prop_assert_eq!(d.temperature, b.temperature);
            }
        }

        /// Property: decoded timestamps are strictly monotonic
        #[test]
        fn prop_monotonic_timestamps((readings, interval) in arb_readings()) {
            let mut enc = Encoder::with_interval(interval);
            for (ts, temp) in readings {
                enc.append(ts, temp).unwrap();
            }

            let decoded = enc.decode();
            for window in decoded.windows(2) {
                prop_assert!(window[0].ts < window[1].ts,
                    "Timestamps not monotonic: {} >= {}", window[0].ts, window[1].ts);
            }
        }

        /// Property: to_bytes() is idempotent (multiple calls return same result)
        #[test]
        fn prop_idempotent_serialization((readings, interval) in arb_readings()) {
            let mut enc = Encoder::with_interval(interval);
            for (ts, temp) in readings {
                enc.append(ts, temp).unwrap();
            }

            let bytes1 = enc.to_bytes();
            let bytes2 = enc.to_bytes();
            prop_assert_eq!(bytes1, bytes2);
        }

        /// Property: all decoded timestamps are multiples of interval from base
        #[test]
        fn prop_timestamp_alignment((readings, interval) in arb_readings()) {
            let mut enc = Encoder::with_interval(interval);
            for (ts, temp) in readings {
                enc.append(ts, temp).unwrap();
            }

            let decoded = enc.decode();
            if let Some(first) = decoded.first() {
                let base = first.ts;
                for reading in &decoded {
                    let offset = reading.ts - base;
                    prop_assert_eq!(offset % (interval as u64), 0,
                        "Timestamp {} not aligned to interval {} from base {}",
                        reading.ts, interval, base);
                }
            }
        }

        /// Property: decoded readings are "close" to input readings
        /// - Timestamps are quantized to interval boundaries
        /// - Temperatures are averaged when multiple readings fall in same interval
        /// - The decoded average is within [min, max] of input temps for that interval
        #[test]
        fn prop_lossy_compression_bounds((readings, interval) in arb_readings()) {
            if readings.is_empty() {
                return Ok(());
            }

            let mut enc = Encoder::with_interval(interval);
            for &(ts, temp) in &readings {
                enc.append(ts, temp).unwrap();
            }

            let decoded = enc.decode();
            if decoded.is_empty() {
                return Ok(());
            }

            // Group input readings by their quantized interval (using encoder's base_ts)
            let base_ts = readings[0].0;
            let mut intervals: std::collections::BTreeMap<u64, Vec<i32>> = std::collections::BTreeMap::new();
            let mut prev_idx = 0u64;

            for &(ts, temp) in &readings {
                if ts < base_ts {
                    continue; // Skipped by encoder
                }
                let idx = (ts - base_ts) / (interval as u64);
                if intervals.is_empty() || idx >= prev_idx {
                    intervals.entry(idx).or_default().push(temp);
                    if idx > prev_idx || intervals.len() == 1 {
                        prev_idx = idx;
                    }
                }
            }

            // For each decoded reading, verify:
            // 1. Timestamp is at an interval boundary
            // 2. Temperature is within [min, max] of readings in that interval
            for reading in &decoded {
                let idx = (reading.ts - base_ts) / (interval as u64);

                if let Some(temps) = intervals.get(&idx) {
                    let min_temp = *temps.iter().min().unwrap();
                    let max_temp = *temps.iter().max().unwrap();

                    prop_assert!(
                        reading.temperature >= min_temp && reading.temperature <= max_temp,
                        "Decoded temp {} not in range [{}, {}] for interval {}",
                        reading.temperature, min_temp, max_temp, idx
                    );
                }
            }
        }

        /// Property: with exactly one reading per interval, decoded equals input exactly
        #[test]
        fn prop_single_reading_identity(
            interval in prop::sample::select(vec![60u16, 300, 600, 3600]),
            temps in prop::collection::vec(-100i32..140, 1..100),
        ) {
            let mut enc = Encoder::with_interval(interval);

            // One reading per interval, no jitter
            for (i, &temp) in temps.iter().enumerate() {
                let ts = BASE_TS + (i as u64) * (interval as u64);
                enc.append(ts, temp).unwrap();
            }

            let decoded = enc.decode();
            prop_assert_eq!(decoded.len(), temps.len(),
                "Decoded count {} doesn't match input count {}", decoded.len(), temps.len());

            for (i, reading) in decoded.iter().enumerate() {
                prop_assert_eq!(reading.temperature, temps[i],
                    "Single reading at interval {} should be exact: expected {}, got {}",
                    i, temps[i], reading.temperature);
            }
        }

        /// Property: multiple readings in the same interval are averaged correctly
        /// The decoded temperature should equal the arithmetic mean (rounded)
        #[test]
        fn prop_averaging_within_interval(
            interval in prop::sample::select(vec![300u16, 600, 3600]),
            // Generate 1-20 intervals, each with 1-50 readings
            interval_temps in prop::collection::vec(
                prop::collection::vec(-500i32..500, 1..50),
                1..20
            ),
        ) {
            let mut enc = Encoder::with_interval(interval);

            // For each interval, add all its readings within the interval window
            for (interval_idx, temps) in interval_temps.iter().enumerate() {
                let interval_start = BASE_TS + (interval_idx as u64) * (interval as u64);
                for (j, &temp) in temps.iter().enumerate() {
                    // Spread readings within the interval
                    let offset = (j as u64) % (interval as u64);
                    enc.append(interval_start + offset, temp).unwrap();
                }
            }

            let decoded = enc.decode();
            prop_assert_eq!(decoded.len(), interval_temps.len(),
                "Expected {} intervals, got {}", interval_temps.len(), decoded.len());

            // Verify each decoded reading is the correct average
            for (i, (reading, temps)) in decoded.iter().zip(interval_temps.iter()).enumerate() {
                let sum: i32 = temps.iter().sum();
                let count = temps.len() as i32;
                // Round half away from zero (same as rounded_avg)
                let expected_avg = if sum >= 0 {
                    (sum + count / 2) / count
                } else {
                    (sum - count / 2) / count
                };

                prop_assert_eq!(reading.temperature, expected_avg,
                    "Interval {}: expected avg {} (sum={}, count={}), got {}",
                    i, expected_avg, sum, count, reading.temperature);
            }
        }

        /// Property: timestamps are quantized to interval boundaries
        /// Input timestamps with jitter should produce output timestamps aligned to intervals
        #[test]
        fn prop_timestamp_quantization(
            interval in prop::sample::select(vec![60u16, 300, 600]),
            // Generate readings with random jitter within each interval
            readings_per_interval in prop::collection::vec(1u8..5, 1..50),
        ) {
            let mut enc = Encoder::with_interval(interval);
            let mut expected_intervals = Vec::new();

            for (interval_idx, &count) in readings_per_interval.iter().enumerate() {
                let interval_start = BASE_TS + (interval_idx as u64) * (interval as u64);
                expected_intervals.push(interval_start);

                for j in 0..count {
                    // Add jitter within interval bounds
                    let jitter = (j as u64 * 17) % (interval as u64);
                    enc.append(interval_start + jitter, 22).unwrap();
                }
            }

            let decoded = enc.decode();
            prop_assert_eq!(decoded.len(), expected_intervals.len(),
                "Expected {} readings, got {}", expected_intervals.len(), decoded.len());

            for (reading, &expected_ts) in decoded.iter().zip(expected_intervals.iter()) {
                prop_assert_eq!(reading.ts, expected_ts,
                    "Expected timestamp {}, got {}", expected_ts, reading.ts);
            }
        }

        /// Property: gaps in input timestamps are preserved in output
        /// If we skip intervals in the input, the output should reflect those gaps
        #[test]
        fn prop_gap_preservation(
            interval in prop::sample::select(vec![60u16, 300, 600]),
            // Generate a list of interval indices (sorted, unique) to have readings
            interval_indices in prop::collection::btree_set(0u64..100, 1..30),
        ) {
            let mut enc = Encoder::with_interval(interval);
            let indices: Vec<u64> = interval_indices.into_iter().collect();

            // Add one reading per selected interval
            for &idx in &indices {
                let ts = BASE_TS + idx * (interval as u64);
                enc.append(ts, 22).unwrap();
            }

            let decoded = enc.decode();
            prop_assert_eq!(decoded.len(), indices.len(),
                "Expected {} readings, got {}", indices.len(), decoded.len());

            // Verify timestamps match the expected intervals (with gaps)
            for (reading, &expected_idx) in decoded.iter().zip(indices.iter()) {
                let expected_ts = BASE_TS + expected_idx * (interval as u64);
                prop_assert_eq!(reading.ts, expected_ts,
                    "Expected timestamp {} (interval {}), got {} (interval {})",
                    expected_ts, expected_idx, reading.ts,
                    (reading.ts - BASE_TS) / (interval as u64));
            }

            // Verify gaps: consecutive readings should have timestamp difference
            // equal to (index_diff * interval)
            for (i, window) in decoded.windows(2).enumerate() {
                let ts_diff = window[1].ts - window[0].ts;
                let idx_diff = indices[i + 1] - indices[i];
                let expected_diff = idx_diff * (interval as u64);

                prop_assert_eq!(ts_diff, expected_diff,
                    "Gap between readings {} and {}: expected {} seconds ({} intervals), got {} seconds",
                    i, i + 1, expected_diff, idx_diff, ts_diff);
            }
        }

        /// Property: decoded count equals number of unique intervals in input
        /// Multiple readings in the same interval count as one
        #[test]
        fn prop_interval_deduplication(
            interval in prop::sample::select(vec![300u16, 600]),
            // Generate timestamps that may fall into same or different intervals
            timestamps in prop::collection::vec(0u64..10000, 1..100),
        ) {
            if timestamps.is_empty() {
                return Ok(());
            }

            // Sort timestamps to ensure monotonic order (encoder rejects out-of-order)
            let mut sorted_timestamps = timestamps.clone();
            sorted_timestamps.sort();

            let mut enc = Encoder::with_interval(interval);

            // Add readings at each timestamp
            for &ts_offset in &sorted_timestamps {
                enc.append(BASE_TS + ts_offset, 22).unwrap();
            }

            let decoded = enc.decode();

            // Calculate expected unique intervals relative to the FIRST timestamp (base_ts)
            // The encoder uses the first reading's timestamp as base_ts
            let base_offset = sorted_timestamps[0];
            let mut unique_intervals = std::collections::BTreeSet::new();
            let mut prev_idx: Option<u64> = None;

            for &ts_offset in &sorted_timestamps {
                let idx = (ts_offset - base_offset) / (interval as u64);
                // Only count if it's the first or advances the index (encoder skips same/backwards)
                if prev_idx.is_none() || idx > prev_idx.unwrap() {
                    unique_intervals.insert(idx);
                    prev_idx = Some(idx);
                }
            }

            prop_assert_eq!(decoded.len(), unique_intervals.len(),
                "Expected {} unique intervals, got {} decoded readings. timestamps={:?}, base_offset={}, interval={}",
                unique_intervals.len(), decoded.len(), sorted_timestamps, base_offset, interval);
        }
    }
}
