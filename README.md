# nibblerun

A high-performance time series compression library for Rust, optimized for temperature sensor data.

## Features

- Bit-packed delta encoding with variable-length codes
- Zero-run length encoding for repeated values
- Configurable interval-based timestamp quantization
- Automatic averaging of multiple readings within the same interval
- ~250M readings/second encoding throughput

## Example

```rust
use nibblerun::{Encoder, decode};

// Create encoder with 5-minute (300-second) intervals
let mut enc = Encoder::with_interval(300);

// Append readings (timestamp, temperature)
// Readings are quantized to interval boundaries
enc.append(1700000000, 23).unwrap();  // 00:00:00 -> interval 0
enc.append(1700000150, 25).unwrap();  // 00:02:30 -> same interval, averaged with above
enc.append(1700000300, 24).unwrap();  // 00:05:00 -> interval 1
enc.append(1700000600, 22).unwrap();  // 00:10:00 -> interval 2

// Serialize to bytes
let bytes = enc.to_bytes();
println!("Compressed size: {} bytes", bytes.len());

// Decode back
let readings = decode(&bytes);
for r in readings {
    println!("ts: {}, temp: {}", r.ts, r.temperature);
}
// Output:
// ts: 1700000000, temp: 24  (average of 23 and 25)
// ts: 1700000300, temp: 24
// ts: 1700000600, temp: 22
```

### Handling Gaps

Missing intervals are preserved in the output:

```rust
use nibblerun::Encoder;

let mut enc = Encoder::with_interval(300);

enc.append(1700000000, 22).unwrap();   // 00:00 - interval 0
enc.append(1700000300, 23).unwrap();   // 00:05 - interval 1
// No data for 00:10, 00:15, 00:20...
enc.append(1700003000, 25).unwrap();   // 00:50 - interval 10

let readings = enc.decode();
assert_eq!(readings.len(), 3);
assert_eq!(readings[2].ts - readings[1].ts, 2700); // 45-minute gap preserved
```

## How It Works

### Timestamp Quantization

Timestamps are quantized to configurable intervals (default: 300 seconds). The first reading's timestamp becomes the base, and all subsequent readings are mapped to interval indices:

```
interval_idx = (timestamp - base_ts) / interval
```

Multiple readings in the same interval are averaged together.

### Delta Encoding

Temperature values are stored as deltas from the previous reading. Deltas are encoded with variable-length bit codes optimized for typical temperature data:

| Delta | Encoding | Bits |
|-------|----------|------|
| 0 (repeated) | `0` | 1 |
| ±1 | `10x` | 3 |
| ±2 | `1110x` | 5 |
| ±3 to ±10 | `1111110xxxx` | 11 |
| ±11 to ±1023 | `11111110xxxxxxxxxxx` | 19 |

### Zero-Run Encoding

Consecutive zero deltas (unchanged temperatures) are optimized for efficiency:

| Run Length | Encoding | Bits | Notes |
|------------|----------|------|-------|
| 1-7 | `0` × n | 1-7 | Individual zeros (more efficient than run encoding) |
| 8-21 | `11110xxxx` | 9 | Run-length encoding |
| 22-149 | `111110xxxxxxx` | 13 | Run-length encoding |

### Gap Encoding

Missing intervals (sensor offline, network issues) are encoded efficiently:

| Gap Size | Encoding | Bits | Notes |
|----------|----------|------|-------|
| 1 interval | `110` | 3 | Optimized for common single-interval gaps |
| 2-65 intervals | `11111111xxxxxx` | 14 | Larger gaps |

### Wire Format

The encoded format consists of a 14-byte header followed by bit-packed data:

```
Header (14 bytes):
┌─────────────────┬─────────┬─────────┬────────────┬──────────┐
│ base_ts_offset  │duration │  count  │ first_temp │ interval │
│    (4 bytes)    │(2 bytes)│(2 bytes)│  (4 bytes) │ (2 bytes)│
└─────────────────┴─────────┴─────────┴────────────┴──────────┘

Data: Variable-length bit-packed deltas and zero-runs
```

- `base_ts_offset`: First timestamp minus epoch base (1,760,000,000)
- `duration`: Number of intervals spanned
- `count`: Number of readings (max 65,535)
- `first_temp`: First temperature as i32
- `interval`: Interval in seconds (1-65,535)

### Memory Layout

The `Encoder` struct is optimized for cache efficiency (72 bytes):

```rust
Encoder {
    base_ts: u64,        // First timestamp
    last_ts: u64,        // Most recent timestamp
    bit_accum: u64,      // Bit accumulator for encoding
    pending_state: u64,  // Packed: bit count + pending avg state
    data: Vec<u8>,       // Encoded output buffer
    prev_temp: i32,      // Previous temperature (for delta)
    first_temp: i32,     // First temperature (for header)
    zero_run: u32,       // Current zero-run length
    prev_logical_idx: u32, // Previous interval index
    count: u16,          // Reading count
    interval: u16,       // Interval in seconds
}
```

The `pending_state` field packs multiple values to avoid extra fields:
- Bits 0-5: Bit accumulator count (0-63)
- Bits 6-15: Pending reading count for averaging (0-1023)
- Bits 16-47: Pending sum for averaging (full i32 range)

## Assumptions and Limitations

### Assumptions

- **Timestamps are monotonically increasing**: Out-of-order readings return an error
- **Timestamps are Unix seconds**: The library uses an epoch base of 1,760,000,000 (~2025)
- **Temperature changes are gradual**: The encoding is optimized for small deltas (±10)

### Limitations

| Limit | Value | Notes |
|-------|-------|-------|
| Max readings per encoder | 65,535 | `count` is u16 |
| Max delta between readings | ±1,023 | Larger deltas return error |
| Max readings per interval | 1,023 | Additional readings return error |
| Min timestamp | 1,760,000,000 | ~2025-10-09, returns error if earlier |
| Interval range | 1-65,535 seconds | ~18 hours max |

### Performance Characteristics

- **Encoding**: O(1) per reading, ~250M readings/second
- **Decoding**: O(n) where n = reading count
- **Compression**: ~40-50 bytes/day for typical temperature data (vs ~3.5KB raw)
- **Memory**: 72 bytes per encoder + output buffer

## Real-World Data Analysis

To validate the encoding scheme, we analyzed **424,353 real-world temperature sensor files** containing **116 million events** (109M readings + 6.8M gaps). The data represents temperature sensors reporting at 5-minute intervals.

### Dataset Characteristics

| Metric | Value |
|--------|-------|
| Files analyzed | 424,353 |
| Total events | 116,019,760 |
| Actual readings | 109,228,832 |
| Gap markers | 6,790,928 |
| Temperature range | -46°C to 120°C |
| Delta range | -134 to +138 |

### Event Distribution

The analysis reveals that temperature sensor data is highly compressible:

| Event Type | Count | % of Events | Bits Each | Total Bits | % of Bits |
|------------|-------|-------------|-----------|------------|-----------|
| Zero deltas | 96,782,708 | 83.42% | ~0.47* | 45,773,215 | 40.1% |
| ±1 deltas | 11,842,800 | 10.21% | 3 | 35,528,400 | 31.2% |
| Single gaps | 6,750,396 | 5.82% | 3 | 20,251,188 | 17.8% |
| Multi gaps | 40,532 | 0.03% | 14 | 567,448 | 0.5% |
| ±2 deltas | 172,920 | 0.15% | 5 | 864,600 | 0.8% |
| ±3-10 deltas | 26,322 | 0.02% | 11 | 289,542 | 0.3% |
| ±11+ deltas | 1,481 | 0.00% | 19 | 28,139 | 0.0% |

*Zero deltas average ~0.47 bits due to run-length encoding efficiency.

**Note:** The encoding was optimized based on this analysis - single-interval gaps (99.4% of all gaps) now use only 3 bits instead of 13, reducing total output by ~36%.

### Key Findings

1. **Temperature stability dominates**: 83.42% of readings show no change from the previous interval. The zero-run encoding is highly effective, achieving ~0.47 bits per zero delta on average.

2. **Small deltas are rare beyond ±1**: Only 10.21% of readings change by ±1°C, and just 0.15% change by ±2°C. Larger deltas (±3 or more) are essentially non-existent at 0.02%.

3. **Gap optimization pays off**: 99.4% of gaps are single-interval, so the optimized 3-bit encoding for single gaps dramatically reduces overhead compared to the uniform 13-bit encoding.

4. **Compression achieved**: ~95x compression ratio (~14 MB compressed vs ~1.3 GB raw at 12 bytes/reading) after gap optimization.

### Delta Breakdown (Top 10)

| Delta | Count | Percentage |
|-------|-------|------------|
| 0 | 96,782,708 | 88.605% |
| -1 | 5,974,307 | 5.470% |
| +1 | 5,868,493 | 5.373% |
| +2 | 114,957 | 0.105% |
| -2 | 57,963 | 0.053% |
| +3 | 10,965 | 0.010% |
| -3 | 5,449 | 0.005% |
| +4 | 2,865 | 0.003% |
| -4 | 1,907 | 0.002% |
| -5 | 1,149 | 0.001% |

### Zero-Run Length Distribution

| Run Length | Count | % of Runs |
|------------|-------|-----------|
| 1-7 (individual bits) | 6,374,763 | 68.55% |
| 8-21 (9-bit encoding) | 1,704,866 | 18.33% |
| 22-149 (13-bit encoding) | 1,202,846 | 12.93% |
| 150+ (multiple chunks) | 16,911 | 0.18% |

The majority of zero-runs are short (1-7 consecutive zeros), validating the decision to use individual zero bits for small runs rather than run-length encoding overhead.

### Gap Analysis

| Gap Size | Count | % of Gaps |
|----------|-------|-----------|
| 1 interval | 6,750,396 | 99.4% |
| 2 intervals | 3,362 | 0.05% |
| 3+ intervals | 37,170 | 0.55% |

99.4% of gaps are single-interval gaps, suggesting potential optimization by using shorter encoding for common single-interval gaps.

### Optimizations Implemented

Based on this analysis, the following optimizations were implemented:

1. **3-bit single-interval gaps**: Since 99.4% of gaps are single-interval, these now use only 3 bits (`110`) instead of 14 bits, reducing gap overhead by 79%.

2. **Prefix-free code design**: The encoding uses a carefully designed prefix-free code that assigns the shortest codes to the most frequent events (zero, ±1, single-gap) while maintaining decodability.

3. **±2 tier retained**: Despite low frequency (0.15%), the 5-bit ±2 tier is cost-effective — removing it would require using 11-bit encoding for those values.

### Running the Analysis

The analysis tool is included as `nbl-analyze`:

```bash
# Build and run
cargo build --release --bin nbl-analyze
./target/release/nbl-analyze /path/to/csv/directory/

# With options
./target/release/nbl-analyze /path/to/csv/ --max-files 10000 --progress 1000
```

CSV files should have the format:
```csv
ts,temperature
1760000000,22
1760000300,23
```

Values of -1000 are treated as gap markers and excluded from temperature statistics.

## Testing

Run unit tests:
```bash
cargo test
```

Run property-based tests (included in unit tests via proptest):
```bash
cargo test proptests
```

### Property Tests

The library includes 12 property-based tests that verify invariants across random inputs:

| Property | Description |
|----------|-------------|
| `prop_size_accuracy` | `size() == to_bytes().len()` |
| `prop_count_consistency` | `decode().len() == count()` |
| `prop_roundtrip_via_bytes` | `decode(to_bytes()) == decode()` |
| `prop_monotonic_timestamps` | Decoded timestamps are strictly increasing |
| `prop_idempotent_serialization` | Multiple `to_bytes()` calls return identical results |
| `prop_timestamp_alignment` | All timestamps align to interval boundaries |
| `prop_lossy_compression_bounds` | Decoded temps are within [min, max] of interval inputs |
| `prop_single_reading_identity` | Single reading per interval decodes exactly |
| `prop_averaging_within_interval` | Multiple readings per interval are averaged correctly |
| `prop_timestamp_quantization` | Timestamps are quantized to interval boundaries |
| `prop_gap_preservation` | Gaps between readings are preserved correctly |
| `prop_interval_deduplication` | Multiple readings in same interval produce one output |

## Fuzzing

The library includes fuzz targets using cargo-fuzz. Install cargo-fuzz first:
```bash
cargo install cargo-fuzz
```

### Fuzz Targets

| Target | Description |
|--------|-------------|
| `fuzz_roundtrip` | Tests encode/decode invariants with arbitrary inputs |
| `fuzz_decode` | Tests that `decode()` never panics on arbitrary bytes |
| `fuzz_idempotent` | Tests that multiple `to_bytes()` calls return identical results |
| `fuzz_lossy_bounds` | Tests that decoded temps are within [min, max] of interval inputs |
| `fuzz_single_reading` | Tests that single reading per interval decodes exactly |
| `fuzz_averaging` | Tests that multiple readings per interval are averaged correctly |
| `fuzz_gaps` | Tests that gaps between readings are preserved correctly |
| `fuzz_lossless` | Tests lossless compression with one reading per interval at exact boundaries, including gaps |

Run fuzz targets:
```bash
# Roundtrip fuzzing (tests encode/decode invariants)
cargo fuzz run fuzz_roundtrip

# Decode-only fuzzing (tests decode never panics on arbitrary input)
cargo fuzz run fuzz_decode

# Run with time limit (e.g., 60 seconds)
cargo fuzz run fuzz_roundtrip -- -max_total_time=60
```

## Benchmarks

Run benchmarks using [Criterion](https://github.com/bheisler/criterion.rs):

```bash
# Run all benchmarks
make bench
# or
cargo bench

# Run specific benchmark group
cargo bench --bench benchmarks -- encode
cargo bench --bench benchmarks -- decode

# Save baseline for comparison
cargo bench --bench benchmarks -- --save-baseline v1

# Compare against saved baseline
cargo bench --bench benchmarks -- --baseline v1
```

HTML reports are generated at `target/criterion/report/index.html`.

### Benchmark Groups

| Group | Description |
|-------|-------------|
| `encode` | Encoding throughput (100, 1000, 10000 readings) |
| `decode` | Decoding throughput (10000 readings) |
| `roundtrip` | Full encode + decode cycle (1000 readings) |

## Code Coverage

### Prerequisites

Install the required tools:

```bash
# Install cargo-llvm-cov for unit test coverage
cargo install cargo-llvm-cov

# Install cargo-fuzz for fuzz testing (requires nightly)
cargo install cargo-fuzz

# Install llvm-tools for fuzz coverage reports
rustup component add llvm-tools-preview
rustup component add --toolchain nightly llvm-tools-preview
```

### Running Coverage

Using the Makefile:

```bash
# Unit test coverage summary
make coverage

# Unit test coverage with HTML report
make coverage-html
# Report at: target/llvm-cov/html/index.html

# Run all fuzz targets (30 seconds each)
make fuzz

# Generate combined fuzz coverage report
make fuzz-coverage
```

Or manually:

```bash
# Unit test coverage
cargo llvm-cov --summary-only
cargo llvm-cov --html

# Fuzz coverage (single target)
cargo +nightly fuzz run fuzz_decode -- -max_total_time=30
cargo +nightly fuzz coverage fuzz_decode
```

### Current Coverage

Unit tests achieve ~98% line coverage across all source files.

## CLI Tools

The crate includes three command-line utilities for generating, visualizing, and analyzing encoded data.

### nbl-gen

Generate sample nibblerun time series data:

```bash
# Generate 24 hours of data (288 readings at 5-min intervals)
nbl-gen day.nbl

# Generate with random gaps (sensor offline periods)
nbl-gen day.nbl --gaps

# Generate with occasional temperature spikes
nbl-gen day.nbl --spikes

# Customize readings count and interval
nbl-gen custom.nbl --readings 100 --interval 600 --base-temp 25
```

Options:
- `--readings N` - Number of readings (default: 288)
- `--gaps` - Include random gaps (5% chance per reading)
- `--spikes` - Include occasional large temperature changes (2% chance)
- `--base-temp N` - Base temperature in Celsius (default: 22)
- `--interval N` - Interval in seconds (default: 300)

### nbl-viz

Visualize the internal bit-level structure of encoded data as SVG:

```bash
# Generate SVG visualization
nbl-viz day.nbl -o day.svg

# Output defaults to input filename with .svg extension
nbl-viz day.nbl  # creates day.svg
```

The SVG shows:
- **Header section**: 14 bytes with labeled fields (base_ts, duration, count, first_value, interval)
- **Bit grid**: Each bit as a colored square, grouped by encoding type
- **Span labels**: Decoded meaning below each bit group (+1, -2, run=5, gap=3, etc.)
- **Decoded timeline**: Human-readable list of timestamps and values

Color scheme:
| Encoding | Color |
|----------|-------|
| Header fields | Light blue |
| Zero (single) | Light green |
| Zero-run 8-21 | Green |
| Zero-run 22-149 | Darker green |
| Delta ±1 | Light orange |
| Delta ±2 | Orange |
| Delta ±3-10 | Darker orange |
| Large delta | Red |
| Single-gap | Purple |
| Multi-gap | Purple |

### nbl-analyze

Analyze CSV files to compute delta frequency distributions and encoding statistics:

```bash
# Analyze all CSV files in a directory
nbl-analyze /path/to/csv/directory/

# Limit to first N files
nbl-analyze /path/to/csv/ --max-files 10000

# Show progress every N files
nbl-analyze /path/to/csv/ --progress 5000
```

Options:
- `--max-files N` - Maximum files to process (default: 0 = all)
- `--progress N` - Show progress every N files (default: 10000)

Output includes:
- Delta frequency distribution by encoding tier
- Zero-run length histogram
- Gap analysis
- Bit cost breakdown
- Optimization recommendations

See [Real-World Data Analysis](#real-world-data-analysis) for example output from analyzing 424K sensor files.

## License

MIT
