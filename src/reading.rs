//! Reading struct for decoded time series data.

/// A decoded sensor reading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading {
    /// Unix timestamp in seconds
    pub ts: u64,
    /// Sensor value
    pub value: i32,
}
