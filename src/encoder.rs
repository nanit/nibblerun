//! Encoder for nibblerun time series compression.

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::appendable::{self, header_size_for_value_bytes};
use crate::error::AppendError;
use crate::reading::Reading;
use crate::value::Value;

/// Encoder for `NibbleRun` format
///
/// Accumulates sensor readings and produces compressed output.
/// Generic over value type V (i8, i16, or i32) and interval INTERVAL (compile-time constant).
///
/// Internally backed by an appendable byte buffer that can be serialized and
/// resumed without losing the ability to append more readings.
#[derive(Clone, Serialize, Deserialize)]
pub struct Encoder<V: Value, const INTERVAL: u16 = 300> {
    /// Internal byte buffer in appendable format
    buf: Vec<u8>,
    #[serde(skip)]
    _marker: PhantomData<V>,
}

impl<V: Value, const INTERVAL: u16> Encoder<V, INTERVAL> {
    /// Create a new encoder
    ///
    /// The interval is specified as a const generic parameter (default: 300 seconds).
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Get the interval in seconds
    #[inline]
    #[must_use]
    pub const fn interval() -> u16 {
        INTERVAL
    }

    /// Header size for this encoder's value type
    ///
    /// Note: This returns the legacy header size for backward compatibility
    /// with the original wire format. The internal buffer uses a larger header.
    #[inline]
    #[must_use]
    pub const fn header_size() -> usize {
        4 + 2 + V::BYTES // base_ts_offset (4) + count (2) + first_value (V::BYTES)
    }

    /// Append a sensor reading
    ///
    /// Multiple readings in the same interval are averaged.
    /// The value type V provides compile-time range checking.
    ///
    /// # Arguments
    /// * `ts` - Unix timestamp in seconds
    /// * `value` - Sensor value (type checked at compile time)
    ///
    /// # Errors
    /// Returns an error if:
    /// - Timestamp is before the base timestamp
    /// - Timestamp is out of order (earlier interval than previous)
    /// - Too many readings in the same interval (max 1023)
    /// - Too many total readings (max 65535)
    /// - Value delta exceeds encodable range [-1024, 1023]
    #[inline]
    pub fn append(&mut self, ts: u64, value: V) -> Result<(), AppendError> {
        if self.buf.is_empty() {
            self.buf = appendable::create::<V, INTERVAL>(ts, value);
            Ok(())
        } else {
            appendable::append::<V, INTERVAL>(&mut self.buf, ts, value)
        }
    }

    /// Get the encoded size in bytes
    ///
    /// Note: This calculates what the finalized size would be (including
    /// flushing pending state), which may differ from `as_bytes().len()`.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        if self.buf.is_empty() {
            return 0;
        }
        // For now, return the buffer size. This includes the full header.
        self.buf.len()
    }

    /// Get the number of readings encoded
    #[inline]
    #[must_use]
    pub fn count(&self) -> usize {
        appendable::count(&self.buf).unwrap_or(0) as usize
    }

    /// Decode the encoder's contents back to readings
    #[must_use]
    pub fn decode(&self) -> Vec<Reading<V>> {
        appendable::decode::<V, INTERVAL>(&self.buf)
    }

    /// Finalize and return the encoded bytes
    ///
    /// Returns the internal buffer in appendable format. This buffer can be
    /// used with `Encoder::from_bytes()` to continue appending, or with
    /// `appendable::decode()` to decode readings.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.buf.clone()
    }

    /// Get a reference to the internal byte buffer
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Create an encoder from a previously serialized buffer
    ///
    /// # Errors
    /// Returns an error if the buffer is too short.
    ///
    /// Note: The caller is responsible for ensuring the buffer was created with
    /// the same value type V and interval INTERVAL as this encoder.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, AppendError> {
        if bytes.is_empty() {
            return Ok(Self::new());
        }
        let header_size = header_size_for_value_bytes(V::BYTES);
        if bytes.len() < header_size {
            return Err(AppendError::CountOverflow); // TODO: better error
        }
        Ok(Self {
            buf: bytes,
            _marker: PhantomData,
        })
    }

}

impl<V: Value, const INTERVAL: u16> Default for Encoder<V, INTERVAL> {
    fn default() -> Self {
        Self::new()
    }
}
