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

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

mod constants;
mod decoder;
mod encoder;
mod error;
mod reading;

#[cfg(test)]
mod tests;

// Re-export public API
pub use decoder::decode;
pub use encoder::Encoder;
pub use error::AppendError;
pub use reading::Reading;

/// Default interval between readings (5 minutes)
pub const DEFAULT_INTERVAL: u64 = 300;
