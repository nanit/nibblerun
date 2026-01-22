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

// Create encoder with 5-minute (300-second) intervals (default)
let mut enc: Encoder<i32> = Encoder::new();

// Append readings (timestamp, value)
// Readings are quantized to interval boundaries
enc.append(1761000000, 23).unwrap();  // 00:00:00 -> interval 0
enc.append(1761000150, 25).unwrap();  // 00:02:30 -> same interval, averaged with above
enc.append(1761000300, 24).unwrap();  // 00:05:00 -> interval 1
enc.append(1761000600, 22).unwrap();  // 00:10:00 -> interval 2

// Serialize to bytes
let bytes = enc.to_bytes();
println!("Compressed size: {} bytes", bytes.len());

// Decode back (interval must match encoding)
let readings = decode::<i32, 300>(&bytes);
for r in readings {
    println!("ts: {}, value: {}", r.ts, r.value);
}
// Output:
// ts: 1761000000, value: 24  (average of 23 and 25)
// ts: 1761000300, value: 24
// ts: 1761000600, value: 22

// Custom interval example (10-minute intervals)
let mut enc_10min: Encoder<i32, 600> = Encoder::new();
```

### Handling Gaps

Missing intervals are preserved in the output:

```rust
use nibblerun::Encoder;

let mut enc: Encoder<i32> = Encoder::new();

enc.append(1761000000, 22).unwrap();   // 00:00 - interval 0
enc.append(1761000300, 23).unwrap();   // 00:05 - interval 1
// No data for 00:10, 00:15, 00:20...
enc.append(1761003000, 25).unwrap();   // 00:50 - interval 10

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
- **Memory**: 56-64 bytes per encoder (depends on value type) + output buffer

## Analyzing Your Data

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

## License

MIT
