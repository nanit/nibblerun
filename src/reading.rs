//! Reading struct for decoded time series data.

/// A decoded temperature reading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading {
    /// Unix timestamp in seconds
    pub ts: u64,
    /// Temperature value
    pub temperature: i32,
}
