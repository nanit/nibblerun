//! Decoding functionality for nibblerun encoded data.

use crate::appendable;
use crate::reading::Reading;
use crate::value::Value;

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
    appendable::decode::<V, INTERVAL>(bytes)
}
