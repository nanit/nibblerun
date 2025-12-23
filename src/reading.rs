//! Reading struct for decoded time series data.

use crate::value::Value;

/// A decoded sensor reading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reading<V: Value> {
    /// Unix timestamp in seconds
    pub ts: u32,
    /// Sensor value
    pub value: V,
}
