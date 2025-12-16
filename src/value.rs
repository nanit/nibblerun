//! Value trait for type-safe encoding with configurable value sizes.
//!
//! Uses the sealed trait pattern to restrict implementations to i8, i16, i32.

use std::fmt::Debug;

/// Private module to seal the trait - users cannot implement `Value` for other types
mod private {
    pub trait Sealed {}

    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
}

/// Trait for values that can be encoded by `Encoder`.
///
/// This trait is sealed - it can only be implemented for `i8`, `i16`, and `i32`.
/// This provides compile-time type safety for value ranges.
pub trait Value: private::Sealed + Copy + Debug + PartialEq + Default {
    /// Number of bytes used to store this value in the header
    const BYTES: usize;

    /// Minimum value (as i32 for comparison)
    const MIN: i32;

    /// Maximum value (as i32 for comparison)
    const MAX: i32;

    /// Convert to i32 for internal calculations
    fn to_i32(self) -> i32;

    /// Convert from i32 (used during decoding)
    /// Caller must ensure value is in range
    fn from_i32(v: i32) -> Self;

    /// Write value to byte slice (little-endian)
    fn write_le(self, buf: &mut [u8]);

    /// Read value from byte slice (little-endian)
    fn read_le(buf: &[u8]) -> Self;
}

impl Value for i8 {
    const BYTES: usize = 1;
    const MIN: i32 = Self::MIN as i32;
    const MAX: i32 = Self::MAX as i32;

    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        v as Self
    }

    #[inline]
    fn write_le(self, buf: &mut [u8]) {
        buf[0] = self as u8;
    }

    #[inline]
    fn read_le(buf: &[u8]) -> Self {
        buf[0] as Self
    }
}

impl Value for i16 {
    const BYTES: usize = 2;
    const MIN: i32 = Self::MIN as i32;
    const MAX: i32 = Self::MAX as i32;

    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        v as Self
    }

    #[inline]
    fn write_le(self, buf: &mut [u8]) {
        let bytes = self.to_le_bytes();
        buf[0] = bytes[0];
        buf[1] = bytes[1];
    }

    #[inline]
    fn read_le(buf: &[u8]) -> Self {
        Self::from_le_bytes([buf[0], buf[1]])
    }
}

impl Value for i32 {
    const BYTES: usize = 4;
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;

    #[inline]
    fn to_i32(self) -> i32 {
        self
    }

    #[inline]
    fn from_i32(v: i32) -> Self {
        v
    }

    #[inline]
    fn write_le(self, buf: &mut [u8]) {
        let bytes = self.to_le_bytes();
        buf[0] = bytes[0];
        buf[1] = bytes[1];
        buf[2] = bytes[2];
        buf[3] = bytes[3];
    }

    #[inline]
    fn read_le(buf: &[u8]) -> Self {
        Self::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
    }
}

