//! Error types for nibblerun encoding operations.

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
