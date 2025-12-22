# NibbleRun Appendable Binary Format Specification

This document describes the binary format used by the appendable buffer in NibbleRun, which enables O(1) append operations for time series data.

## Header Layout (14 bytes for i8 values)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `base_ts_offset` | Timestamp offset from EPOCH_BASE (1,760,000,000) |
| 4 | 2 | `count` | Number of readings stored |
| 6 | 2 | `prev_logical_idx` | Interval index of the last reading (for gap detection) |
| 8 | 1 | `first_value` | First temperature value |
| 9 | 1 | `prev_value` | Previous temperature value (for delta calculation) |
| 10 | 1 | `current_value` | Current/last temperature value |
| 11 | 1 | `zero_run` | Count of consecutive zero deltas (buffered, not yet written) |
| 12 | 1 | `bit_count` | Number of bits in accumulator (0-7) |
| 13 | 1 | `bit_accum` | Bit accumulator for partial byte |
| 14+ | var | `data[]` | Bit-packed delta encodings |

### Key Fields Explained

- **prev_logical_idx**: The interval index of the last reading. When a new reading arrives at index N, gap is calculated as `N - prev_logical_idx - 1`. This enables O(1) gap detection without scanning existing data.

- **zero_run**: Buffers consecutive zero deltas. These are NOT written immediately - they're flushed when a non-zero delta arrives or when a gap is detected. This enables efficient run-length encoding.

- **bit_count/bit_accum**: Partial byte buffer. Bits accumulate here until we have 8, then they flush to `data[]`. This avoids byte-alignment overhead.

## Delta Encoding Scheme

| Pattern | Bits | Meaning |
|---------|------|---------|
| `0` | 1 | Zero delta (same value as previous) |
| `100` | 3 | +1 delta |
| `101` | 3 | -1 delta |
| `110` | 3 | Single-interval gap marker |
| `11100` | 5 | +2 delta |
| `11101` | 5 | -2 delta |
| `1111110xxxx` | 11 | ±3 to ±10 delta (4-bit payload) |
| `11111110xxxxxxxxxxx` | 19 | Large delta ±11 to ±1023 (11-bit signed payload) |
| `11111111xxxxxx` | 14 | Large gap of 2-65 intervals (6-bit payload = gap-2) |

## Zero Run Encoding (when flushed)

| Count | Encoding | Total bits |
|-------|----------|------------|
| 1-7 | Individual `0` bits | 1-7 |
| 8-21 | `11110xxxx` (prefix + 4-bit count-8) | 9 |
| 22-149 | `111110xxxxxxx` (prefix + 7-bit count-22) | 13 |
| 150+ | Multiple 13-bit chunks | 13 each |

## Append Operation Flow

```
append(timestamp, value)
    │
    ├─ Calculate new_idx from timestamp
    │
    ├─ Finalize PREVIOUS interval's delta (if count > 1):
    │   │
    │   ├─ if delta == 0 → zero_run++
    │   │
    │   └─ if delta != 0 → flush_zeros(), encode_delta()
    │
    ├─ Gap handling (if new_idx - prev_idx > 1):
    │   │
    │   ├─ flush_zeros()  ← CRITICAL: zeros flushed BEFORE gap
    │   │
    │   └─ write_gaps()
    │
    ├─ Update header fields:
    │   │
    │   ├─ prev_value = current_value
    │   ├─ current_value = new_value
    │   ├─ prev_logical_idx = new_idx
    │   └─ count++
    │
    └─ Done (O(1) operation)
```

### Critical: Temporal Ordering in Bitstream

The bitstream MUST preserve chronological order:
```
<zeros><gap><zeros><gap><delta>
```

This is achieved by:
1. Flushing accumulated zeros BEFORE writing any gap marker
2. Processing the previous interval's delta BEFORE handling gaps

Example scenario:
- zero_run=100, then gap, then 100 more same values, then gap, then +25 delta
- Bitstream: `[100 zeros encoded][gap][100 zeros encoded][gap][+25 delta]`

## Why O(1) Append?

1. **No scanning**: Header tracks all state needed to append
2. **No rewriting**: Only append to `data[]`, never modify existing bytes
3. **Deferred writes**: `zero_run` buffers zeros, `bit_accum` buffers partial bytes
4. **Fixed-cost operations**: Each append does at most ~2-3 byte writes

## Scenarios Reference

### Scenario 1: Append Same Value (Zero Delta)
- Just increment `zero_run` and `count`
- No bits written anywhere
- Header changes: `count++`, `prev_idx++`, `zero_run++`

### Scenario 2: Append Different Value (Non-Zero Delta)
- Flush accumulated zeros to `bit_accum`
- Write delta encoding to `bit_accum`
- If `bit_count >= 8`, overflow byte goes to `data[]`
- Header changes: `count++`, `prev_idx++`, `prev_v=curr_v`, `curr_v=new`, `zero_run=0`, `bit_count`, `bit_accum`

### Scenario 3: Zero Run Threshold
- When `zero_run` exceeds 7, run-length encoding becomes more efficient
- 7 zeros = 7 bits (individual)
- 8 zeros = 9 bits (run-encoded), but same encoding handles up to 21 zeros

### Scenario 4: Single-Interval Gap
- Detected when `new_idx - prev_idx > 1`
- Steps:
  1. Flush any accumulated zeros
  2. Write gap marker `110`
  3. Handle value delta (may start new zero_run)

### Scenario 5: Large Gap (2-65 intervals)
- Uses 14-bit encoding: `11111111` prefix + 6-bit `(gap-2)`
- Example: gap of 10 → `11111111_001000`

### Scenario 6: Large Delta
- Deltas ±11 to ±1023 use 19-bit encoding
- `11111110` prefix + 11-bit signed value
- Beyond ±1023, append fails (returns error)

## Source Code Reference

Main implementation: `src/appendable.rs`

Key functions:
- `append()` - Main append function (lines ~320-400)
- `flush_zeros()` - Flushes zero_run to bit stream
- `encode_delta()` - Writes delta encoding
- `write_gaps()` - Writes gap markers
- `decode()` / `decode_frozen()` - Decode back to readings

## Visual Reference

See `docs/appendable-format.svg` for visual diagrams showing before/after states for each scenario.

---

## Frozen Format

The frozen format is a compact read-only format produced by `freeze()`. It removes all append state overhead, making it ideal for storage and distribution.

### Frozen Header Layout

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `base_ts_offset` | Timestamp offset from EPOCH_BASE (1,760,000,000) |
| 4 | 2 | `count` | Number of readings stored |
| 6 | V::BYTES | `first_value` | First value (1 byte for i8, 2 for i16, 4 for i32) |
| 6+V::BYTES | var | `data[]` | Bit-packed delta encodings (finalized) |

**Header sizes by value type:**
- i8: 7 bytes
- i16: 8 bytes
- i32: 10 bytes

### freeze() Function

`freeze()` converts an appendable buffer to frozen format:

```rust
pub fn freeze<V: Value, const INTERVAL: u16>(buf: &[u8]) -> Vec<u8>
```

**Process:**
1. Read appendable header to extract state (base_ts_offset, count, first_value, prev_value, current_value, zero_run, bit_count, bit_accum)
2. Finalize the pending delta between prev_value and current_value:
   - If delta == 0: add to zero_run
   - If delta != 0: flush zeros, encode delta
3. Flush any remaining zeros
4. Flush partial bits in accumulator (pad to byte boundary)
5. Write minimal frozen header + finalized bit data

**Space savings:**
The frozen format removes 7 bytes of per-encoder overhead (prev_logical_idx, prev_value, current_value, zero_run, bit_count, bit_accum). For many small time series, this represents significant savings.

### decode_frozen() Function

Use `decode_frozen()` to decode frozen format buffers:

```rust
pub fn decode_frozen<V: Value, const INTERVAL: u16>(buf: &[u8]) -> Vec<Reading<V>>
```

The standard `decode()` function works with appendable format buffers. Use `decode_frozen()` specifically for frozen format.

### When to Use Each Format

| Use Case | Format | Function |
|----------|--------|----------|
| Active encoding, may append more | Appendable | `Encoder::to_bytes()` |
| Long-term storage | Frozen | `freeze()` |
| Network transmission | Frozen | `freeze()` |
| Resume appending later | Appendable | `Encoder::from_bytes()` |
