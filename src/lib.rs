//! `NibbleRun` - High-performance time series compression for slow-changing data
//!
//! A bit-packed compression format optimized for sensor data that changes gradually,
//! such as temperature, humidity, or similar environmental readings.
//!
//! # Example
//! ```
//! use nibblerun::Encoder;
//!
//! let mut encoder: Encoder<i32> = Encoder::new();
//! let base_ts = 1_761_000_000_u32;
//!
//! encoder.append(base_ts, 22).unwrap();
//! encoder.append(base_ts + 300, 22).unwrap();
//! encoder.append(base_ts + 600, 23).unwrap();
//!
//! // Decode from encoder
//! let readings = encoder.decode().unwrap();
//! for r in &readings {
//!     println!("ts={}, value={}", r.ts, r.value);
//! }
//!
//! // Freeze for storage and decode later
//! let frozen = encoder.freeze();
//! let readings = nibblerun::decode_frozen::<i32, 300>(&frozen).unwrap();
//! ```
//!
//! # Two Serialization Formats
//!
//! NibbleRun provides two formats:
//!
//! - **Appendable format** (`to_bytes()` / `from_bytes()`): Preserves internal state,
//!   allowing you to continue appending readings after deserialization.
//!
//! - **Frozen format** (`freeze()` / `decode_frozen()`): Compact read-only format
//!   for long-term storage. Cannot be restored to an encoder.

// Panic-free library: disallow unwrap/expect/panic in library code
#![deny(clippy::panic)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

pub mod constants;
mod decoder;
mod encoder;
mod error;
mod reading;
mod value;

#[cfg(test)]
mod tests;

// Re-export public API
pub use decoder::decode_frozen;
pub use encoder::Encoder;
pub use error::{AppendError, DecodeError};
pub use reading::Reading;
pub use value::Value;
