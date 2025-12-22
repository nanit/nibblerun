# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build            # Build the library
cargo test             # Run all tests
cargo test <test_name> # Run a single test (e.g., cargo test test_roundtrip)
cargo clippy           # Run linter
cargo fmt              # Format code
```

## Project Overview

NibbleRun is a Rust library for high-performance lossless time series compression, optimized for temperature sensor data with 5-minute intervals. It achieves ~70-85x compression with O(1) append operations.

## Architecture

**Single-file library** (`src/lib.rs`) with two main components:

- **Encoder**: Accumulates temperature readings and produces compressed output using variable-length bit codes
- **decode()**: Reconstructs readings from compressed bytes

**Key encoding optimizations**:
- Zero-delta runs (89% of readings) use 1-2 bits
- ±1 deltas use 3 bits
- Larger deltas use 7-19 bits
- Gap markers for missing sensor readings (SENTINEL_VALUE = -1000)
- Fast division by interval using multiplication by reciprocal

**Binary format** (two variants):
- **Appendable format**: 14-20 byte header (depends on value type) + bit-packed data. Used by `Encoder::to_bytes()`, supports O(1) append resumption.
- **Frozen format**: 7-10 byte header + bit-packed data. Created by `freeze()`, compact read-only format for storage.
- Timestamps are stored as offsets from EPOCH_BASE (1,760,000,000)
- Values are stored as deltas with variable-length bit codes
- DEFAULT_INTERVAL is 300 seconds (5 minutes), configurable via const generic

## Constraints

- Uses Rust 2024 edition
- Value types: i8, i16, i32 (generic)
- Max readings per encoder: 65535
- Max delta between readings: ±1023
- Default interval: 300 seconds (configurable via const generic)

## Guidelines

- Never calculate - always measure. When benchmarking or reporting performance, measure actual values rather than deriving them from other measurements.
