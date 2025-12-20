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
- Fast division by 300 using multiplication by reciprocal

**Binary format** (10-byte header + variable bit-packed data):
- Timestamps are stored as offsets from EPOCH_BASE (1,760,000,000)
- Temperatures are stored as deltas from TEMP_BASE (20)
- DEFAULT_INTERVAL is 300 seconds (5 minutes)

## Constraints

- Uses Rust 2024 edition
- Temperature range: -108 to 147 (i8 offset from base of 20)
- Max readings per encoder: 65535
- Interval: 300 seconds fixed

## Guidelines

- Never calculate - always measure. When benchmarking or reporting performance, measure actual values rather than deriving them from other measurements.
