//! Error types for nibblerun encoding and decoding operations.

use std::fmt;

use crate::value::Value;

/// Error returned when appending a reading fails
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError<V: Value> {
    /// Timestamp is before the base timestamp (first reading's timestamp)
    TimestampBeforeBase { ts: u32, base_ts: u32 },
    /// Timestamp would place reading in an earlier interval (out of order)
    OutOfOrder {
        ts: u32,
        logical_idx: u32,
        prev_logical_idx: u32,
    },
    /// Too many readings in the same interval (max 1023)
    IntervalOverflow { count: u16 },
    /// Too many total readings (max 65535)
    CountOverflow,
    /// Value delta exceeds encodable range (must be in [-1024, 1023])
    DeltaOverflow {
        delta: i32,
        current_value: V,
        new_value: V,
    },
    /// Timestamp exceeds maximum time span (~227 days at 300s interval, ~45 days at 60s)
    TimeSpanOverflow { ts: u32, base_ts: u32, max_intervals: u32 },
    /// Buffer is too short to contain valid encoded data
    BufferTooShort { expected: usize, actual: usize },
    /// Encoded data is malformed or corrupted
    MalformedData,
}

/// Error returned when decoding fails
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer is too short to contain valid encoded data
    BufferTooShort { expected: usize, actual: usize },
    /// Header contains invalid values
    InvalidHeader,
    /// Encoded data is malformed or corrupted
    MalformedData,
}

impl<V: Value> fmt::Display for AppendError<V> {
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
                current_value,
                new_value,
            } => {
                write!(
                    f,
                    "value delta {delta} ({} -> {}) exceeds range [-1024, 1023]",
                    current_value.to_i32(),
                    new_value.to_i32()
                )
            }
            Self::TimeSpanOverflow { ts, base_ts, max_intervals } => {
                write!(
                    f,
                    "timestamp {ts} exceeds maximum time span from base {base_ts} (max {max_intervals} intervals)"
                )
            }
            Self::BufferTooShort { expected, actual } => {
                write!(f, "buffer too short: expected at least {expected} bytes, got {actual}")
            }
            Self::MalformedData => write!(f, "encoded data is malformed or corrupted"),
        }
    }
}

impl<V: Value> std::error::Error for AppendError<V> {}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooShort { expected, actual } => {
                write!(f, "buffer too short: expected at least {expected} bytes, got {actual}")
            }
            Self::InvalidHeader => write!(f, "invalid header in encoded data"),
            Self::MalformedData => write!(f, "encoded data is malformed or corrupted"),
        }
    }
}

impl std::error::Error for DecodeError {}
