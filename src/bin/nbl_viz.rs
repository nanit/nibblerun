//! Visualize nibblerun encoded data as interactive SVG.

use clap::Parser;
use nibblerun::{Encoder, Reading, Value};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

// ============================================================================
// Bit span types (for visualization)
// ============================================================================

/// Header size for i8 values
const HEADER_SIZE: usize = header_size_for_value_bytes(1);

/// Compute header size based on value byte size
const fn header_size_for_value_bytes(value_bytes: usize) -> usize {
    4 + 2 + 2 + value_bytes + value_bytes + value_bytes + 1 + 1 + 1
}

// Header field offsets
const OFF_BASE_TS_OFFSET: usize = 0;
const OFF_COUNT: usize = 4;
const OFF_PREV_LOGICAL_IDX: usize = 6;
const OFF_FIRST_VALUE: usize = 8;

#[inline]
const fn off_prev_value(value_bytes: usize) -> usize {
    8 + value_bytes
}

#[inline]
const fn off_current_value(value_bytes: usize) -> usize {
    8 + 2 * value_bytes
}

#[inline]
const fn off_zero_run(value_bytes: usize) -> usize {
    8 + 3 * value_bytes
}

#[inline]
const fn off_bit_count(value_bytes: usize) -> usize {
    8 + 3 * value_bytes + 1
}

#[inline]
const fn off_bit_accum(value_bytes: usize) -> usize {
    8 + 3 * value_bytes + 2
}

/// Information about a span of bits in the encoded data (for visualization)
#[derive(Debug, Clone)]
struct BitSpan {
    /// Unique identifier for this span
    id: usize,
    /// Starting bit position in the encoded data
    start_bit: usize,
    /// Number of bits in this span
    length: usize,
    /// What this span represents
    kind: BitSpanKind,
}

/// The kind of data a bit span represents
#[derive(Debug, Clone)]
enum BitSpanKind {
    /// Header: base timestamp
    HeaderBaseTs(u32),
    /// Header: reading count
    HeaderCount(u16),
    /// Header: previous logical index
    HeaderPrevLogicalIdx(u16),
    /// Header: first value
    HeaderFirstValue(i32),
    /// Header: previous value (delta reference)
    HeaderPrevValue(i32),
    /// Header: current value (current interval)
    HeaderCurrentValue(i32),
    /// Header: zero run counter
    HeaderZeroRun(u8),
    /// Header: bit count
    HeaderBitCount(u8),
    /// Header: bit accumulator
    HeaderBitAccum(u8),
    /// Zero delta (unchanged value)
    Zero,
    /// Zero run 8-21 values
    ZeroRun8_21(u32),
    /// Zero run 22-149 values
    ZeroRun22_149(u32),
    /// Delta ±1
    Delta1(i32),
    /// Delta ±2
    Delta2(i32),
    /// Delta ±3 to ±10
    Delta3_10(i32),
    /// Large delta ±11 to ±1023
    LargeDelta(i32),
    /// Single-interval gap
    SingleGap,
    /// Multi-interval gap (2-65 intervals)
    Gap(u32),
}

// Delta encoding table
const DELTA_ENCODE: [(u32, u8); 21] = [
    (0b1111110_0000, 11), // -10
    (0b1111110_0001, 11), // -9
    (0b1111110_0010, 11), // -8
    (0b1111110_0011, 11), // -7
    (0b1111110_0100, 11), // -6
    (0b1111110_0101, 11), // -5
    (0b1111110_0110, 11), // -4
    (0b1111110_0111, 11), // -3
    (0b11101, 5),         // -2
    (0b101, 3),           // -1
    (0, 0),               // 0 (unused)
    (0b100, 3),           // +1
    (0b11100, 5),         // +2
    (0b1111110_1000, 11), // +3
    (0b1111110_1001, 11), // +4
    (0b1111110_1010, 11), // +5
    (0b1111110_1011, 11), // +6
    (0b1111110_1100, 11), // +7
    (0b1111110_1101, 11), // +8
    (0b1111110_1110, 11), // +9
    (0b1111110_1111, 11), // +10
];

/// Encode a zero run, returning (bits, num_bits, consumed)
#[inline]
const fn encode_zero_run(n: u32) -> (u32, u32, u32) {
    if n <= 7 {
        (0, 1, 1)
    } else if n <= 21 {
        ((0b11110 << 4) | (n - 8), 9, n)
    } else if n <= 149 {
        ((0b11_1110 << 7) | (n - 22), 13, n)
    } else {
        ((0b11_1110 << 7) | 127, 13, 149)
    }
}

/// Encode a delta value, returning (bits, num_bits)
#[inline]
fn encode_delta_value(delta: i32) -> (u32, u32) {
    let idx = delta.wrapping_add(10) as usize;
    if idx <= 20 {
        let (bits, num_bits) = DELTA_ENCODE[idx];
        if num_bits > 0 {
            return (bits, u32::from(num_bits));
        }
    }
    let clamped = delta.clamp(-1024, 1023);
    let bits = (0b1111_1110_u32 << 11) | ((clamped as u32) & 0x7FF);
    (bits, 19)
}

#[inline]
fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([buf[offset], buf[offset + 1]])
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
}

/// BitReader with position tracking for span calculation
struct BitReaderWithPos<'a> {
    buf: &'a [u8],
    byte_pos: usize,
    bits: u64,
    left: u32,
}

impl<'a> BitReaderWithPos<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut r = BitReaderWithPos {
            buf,
            byte_pos: 0,
            bits: 0,
            left: 0,
        };
        r.refill();
        r
    }

    fn refill(&mut self) {
        while self.left <= 56 && self.byte_pos < self.buf.len() {
            self.bits = (self.bits << 8) | u64::from(self.buf[self.byte_pos]);
            self.byte_pos += 1;
            self.left += 8;
        }
    }

    fn read_bits(&mut self, n: u32) -> u32 {
        if self.left < n {
            self.refill();
        }
        if self.left < n {
            return 0;
        }
        self.left -= n;
        ((self.bits >> self.left) & ((1 << n) - 1)) as u32
    }

    fn bit_pos(&self) -> usize {
        (self.byte_pos * 8).saturating_sub(self.left as usize)
    }

    const fn has_more(&self) -> bool {
        self.left > 0 || self.byte_pos < self.buf.len()
    }
}

/// Decode buffer to readings with bit span information for visualization.
fn decode_with_spans<V: Value, const INTERVAL: u16>(
    buf: &[u8],
) -> (Vec<Reading<V>>, Vec<BitSpan>, Vec<BitSpan>) {
    let header_size = header_size_for_value_bytes(V::BYTES);
    let value_bits = V::BYTES * 8;
    if buf.len() < header_size {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut header_spans = Vec::new();
    let mut data_spans = Vec::new();
    let mut span_id = 0;

    // Parse header and create header spans
    let base_ts = read_u32_le(buf, OFF_BASE_TS_OFFSET);
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_BASE_TS_OFFSET * 8,
        length: 32,
        kind: BitSpanKind::HeaderBaseTs(base_ts),
    });
    span_id += 1;

    let count = read_u16_le(buf, OFF_COUNT) as usize;
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_COUNT * 8,
        length: 16,
        kind: BitSpanKind::HeaderCount(count as u16),
    });
    span_id += 1;

    let prev_logical_idx = read_u16_le(buf, OFF_PREV_LOGICAL_IDX);
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_PREV_LOGICAL_IDX * 8,
        length: 16,
        kind: BitSpanKind::HeaderPrevLogicalIdx(prev_logical_idx),
    });
    span_id += 1;

    let first_value = V::read_le(&buf[OFF_FIRST_VALUE..]).to_i32();
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: OFF_FIRST_VALUE * 8,
        length: value_bits,
        kind: BitSpanKind::HeaderFirstValue(first_value),
    });
    span_id += 1;

    let prev_value = V::read_le(&buf[off_prev_value(V::BYTES)..]).to_i32();
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_prev_value(V::BYTES) * 8,
        length: value_bits,
        kind: BitSpanKind::HeaderPrevValue(prev_value),
    });
    span_id += 1;

    let current_value = V::read_le(&buf[off_current_value(V::BYTES)..]).to_i32();
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_current_value(V::BYTES) * 8,
        length: value_bits,
        kind: BitSpanKind::HeaderCurrentValue(current_value),
    });
    span_id += 1;

    let zero_run = buf[off_zero_run(V::BYTES)];
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_zero_run(V::BYTES) * 8,
        length: 8,
        kind: BitSpanKind::HeaderZeroRun(zero_run),
    });
    span_id += 1;

    let bit_count = buf[off_bit_count(V::BYTES)];
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_bit_count(V::BYTES) * 8,
        length: 8,
        kind: BitSpanKind::HeaderBitCount(bit_count),
    });
    span_id += 1;

    let bit_accum = u32::from(buf[off_bit_accum(V::BYTES)]);
    header_spans.push(BitSpan {
        id: span_id,
        start_bit: off_bit_accum(V::BYTES) * 8,
        length: 8,
        kind: BitSpanKind::HeaderBitAccum(bit_accum as u8),
    });
    span_id += 1;

    if count == 0 {
        return (Vec::new(), header_spans, data_spans);
    }

    let first_val = if count == 1 {
        V::from_i32(current_value)
    } else {
        V::from_i32(first_value)
    };

    let mut decoded = Vec::with_capacity(count);
    decoded.push(Reading {
        ts: base_ts as u32,
        value: first_val,
    });

    if count == 1 {
        return (decoded, header_spans, data_spans);
    }

    // Build finalized bit data
    let data = &buf[header_size..];
    let mut final_data = data.to_vec();
    let mut accum = bit_accum;
    let mut bits = u32::from(bit_count) & 7;
    let mut run_zeros = u32::from(zero_run);

    let flush_bytes = |accum: &mut u32, bits: &mut u32, out: &mut Vec<u8>| {
        while *bits >= 8 {
            *bits -= 8;
            out.push((*accum >> *bits) as u8);
        }
    };

    let delta = current_value.wrapping_sub(prev_value);
    if delta == 0 {
        run_zeros = run_zeros.saturating_add(1);
    } else {
        let max_iterations = 1000u32;
        let mut iterations = 0u32;
        while run_zeros > 0 && iterations < max_iterations {
            let (b, n, c) = encode_zero_run(run_zeros);
            accum = (accum << n) | b;
            bits = bits.saturating_add(n);
            run_zeros = run_zeros.saturating_sub(c);
            flush_bytes(&mut accum, &mut bits, &mut final_data);
            iterations += 1;
        }
        let (delta_bits, delta_num_bits) = encode_delta_value(delta);
        accum = (accum << delta_num_bits) | delta_bits;
        bits = bits.saturating_add(delta_num_bits);
        flush_bytes(&mut accum, &mut bits, &mut final_data);
    }

    let max_iterations = 1000u32;
    let mut iterations = 0u32;
    while run_zeros > 0 && iterations < max_iterations {
        let (b, n, c) = encode_zero_run(run_zeros);
        accum = (accum << n) | b;
        bits = bits.saturating_add(n);
        run_zeros = run_zeros.saturating_sub(c);
        flush_bytes(&mut accum, &mut bits, &mut final_data);
        iterations += 1;
    }

    flush_bytes(&mut accum, &mut bits, &mut final_data);
    if bits > 0 {
        final_data.push((accum << (8 - bits)) as u8);
    }

    // Now decode with span tracking
    let mut reader = BitReaderWithPos::new(&final_data);
    let mut prev_val = first_value;
    let mut idx = 1u32;
    let interval = u32::from(INTERVAL);
    let header_bits = header_size * 8;

    while decoded.len() < count && reader.has_more() {
        let start_bit = reader.bit_pos();

        if reader.read_bits(1) == 0 {
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 1,
                kind: BitSpanKind::Zero,
            });
            span_id += 1;
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        if reader.read_bits(1) == 0 {
            let sign = reader.read_bits(1);
            let delta = if sign == 0 { 1 } else { -1 };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 3,
                kind: BitSpanKind::Delta1(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        if reader.read_bits(1) == 0 {
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 3,
                kind: BitSpanKind::SingleGap,
            });
            span_id += 1;
            idx += 1;
            continue;
        }

        if reader.read_bits(1) == 0 {
            let sign = reader.read_bits(1);
            let delta = if sign == 0 { 2 } else { -2 };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 5,
                kind: BitSpanKind::Delta2(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        if reader.read_bits(1) == 0 {
            let n = reader.read_bits(4) + 8;
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 9,
                kind: BitSpanKind::ZeroRun8_21(n),
            });
            span_id += 1;
            for _ in 0..n {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: base_ts + idx * interval,
                    value: V::from_i32(prev_val),
                });
                idx += 1;
            }
            continue;
        }

        if reader.read_bits(1) == 0 {
            let n = reader.read_bits(7) + 22;
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 13,
                kind: BitSpanKind::ZeroRun22_149(n),
            });
            span_id += 1;
            for _ in 0..n {
                if decoded.len() >= count {
                    break;
                }
                decoded.push(Reading {
                    ts: base_ts + idx * interval,
                    value: V::from_i32(prev_val),
                });
                idx += 1;
            }
            continue;
        }

        if reader.read_bits(1) == 0 {
            let e = reader.read_bits(4) as i32;
            let delta = if e < 8 { e - 10 } else { e - 5 };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 11,
                kind: BitSpanKind::Delta3_10(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
            continue;
        }

        if reader.read_bits(1) == 0 {
            let raw = reader.read_bits(11);
            let delta = if raw & 0x400 != 0 {
                (raw | 0xFFFF_F800) as i32
            } else {
                raw as i32
            };
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 19,
                kind: BitSpanKind::LargeDelta(delta),
            });
            span_id += 1;
            prev_val = prev_val.wrapping_add(delta);
            decoded.push(Reading {
                ts: base_ts + idx * interval,
                value: V::from_i32(prev_val),
            });
            idx += 1;
        } else {
            let n = reader.read_bits(6) + 2;
            data_spans.push(BitSpan {
                id: span_id,
                start_bit: header_bits + start_bit,
                length: 14,
                kind: BitSpanKind::Gap(n),
            });
            span_id += 1;
            idx += u32::from(n);
        }
    }

    (decoded, header_spans, data_spans)
}

// ============================================================================
// Visualization code
// ============================================================================

/// Original reading from CSV for ground truth comparison
#[derive(Debug, Clone)]
struct CsvReading {
    ts: u32,
    value: i8,
}

const DEFAULT_INTERVAL: u16 = 300;

// Layout constants
#[allow(dead_code)]
const LEFT_WIDTH: usize = 320;
const RIGHT_X: usize = 380;
const ROW_HEIGHT: usize = 24;
const BIT_SIZE: usize = 14;
const BIT_GAP: usize = 2;
const MARGIN: usize = 20;
const LEGEND_WIDTH: usize = 280;
const LEGEND_X: usize = 900;
// Ground truth column (when CSV input)
const GT_X: usize = 700;
const GT_WIDTH: usize = 300;

// Bitstream view constants
const BITSTREAM_Y_OFFSET: usize = 40;
const BITSTREAM_BIT_SIZE: usize = 12;
const BITSTREAM_MAX_WIDTH: usize = 192;

#[derive(Parser)]
#[command(name = "nbl-viz")]
#[command(about = "Visualize nibblerun encoded data as interactive SVG")]
struct Args {
    /// Input file (.nbl or .csv with ts,temperature columns)
    input: PathBuf,

    /// Output SVG file (default: input with .svg extension)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// Colors for different encoding types
mod colors {
    pub const HEADER: &str = "#E3F2FD";
    pub const ZERO: &str = "#E8F5E9";
    pub const ZERO_RUN_8_21: &str = "#A5D6A7";
    pub const ZERO_RUN_22_149: &str = "#81C784";
    pub const DELTA_1: &str = "#FFF3E0";
    pub const DELTA_2: &str = "#FFE0B2";
    pub const DELTA_3_10: &str = "#FFCC80";
    pub const LARGE_DELTA: &str = "#FFCDD2";
    pub const GAP: &str = "#E1BEE7";
    pub const BORDER: &str = "#9E9E9E";
    pub const HIGHLIGHT: &str = "#1976D2";
}

/// SpanKind wraps BitSpanKind with visualization-specific methods
#[derive(Debug, Clone)]
struct SpanKind(BitSpanKind);

impl SpanKind {
    fn color(&self) -> &'static str {
        match &self.0 {
            BitSpanKind::HeaderBaseTs(_)
            | BitSpanKind::HeaderCount(_)
            | BitSpanKind::HeaderPrevLogicalIdx(_)
            | BitSpanKind::HeaderFirstValue(_)
            | BitSpanKind::HeaderPrevValue(_)
            | BitSpanKind::HeaderCurrentValue(_)
            | BitSpanKind::HeaderZeroRun(_)
            | BitSpanKind::HeaderBitCount(_)
            | BitSpanKind::HeaderBitAccum(_) => colors::HEADER,
            BitSpanKind::Zero => colors::ZERO,
            BitSpanKind::ZeroRun8_21(_) => colors::ZERO_RUN_8_21,
            BitSpanKind::ZeroRun22_149(_) => colors::ZERO_RUN_22_149,
            BitSpanKind::Delta1(_) => colors::DELTA_1,
            BitSpanKind::Delta2(_) => colors::DELTA_2,
            BitSpanKind::Delta3_10(_) => colors::DELTA_3_10,
            BitSpanKind::LargeDelta(_) => colors::LARGE_DELTA,
            BitSpanKind::SingleGap | BitSpanKind::Gap(_) => colors::GAP,
        }
    }

    fn label(&self) -> String {
        match &self.0 {
            BitSpanKind::HeaderBaseTs(v) => format!("base_ts: {v}"),
            BitSpanKind::HeaderCount(v) => format!("count: {v}"),
            BitSpanKind::HeaderPrevLogicalIdx(v) => format!("prev_idx: {v}"),
            BitSpanKind::HeaderFirstValue(v) => format!("first_val: {v}"),
            BitSpanKind::HeaderPrevValue(v) => format!("prev_val: {v}"),
            BitSpanKind::HeaderCurrentValue(v) => format!("curr_val: {v}"),
            BitSpanKind::HeaderZeroRun(v) => format!("zero_run: {v}"),
            BitSpanKind::HeaderBitCount(v) => format!("bit_count: {v}"),
            BitSpanKind::HeaderBitAccum(v) => format!("bit_accum: {v:#x}"),
            BitSpanKind::Zero => "=0".to_string(),
            BitSpanKind::ZeroRun8_21(n) | BitSpanKind::ZeroRun22_149(n) => format!("run={n}"),
            BitSpanKind::Delta1(d) | BitSpanKind::Delta2(d) | BitSpanKind::Delta3_10(d) | BitSpanKind::LargeDelta(d) => {
                format!("{d:+}")
            }
            BitSpanKind::SingleGap => "gap=1".to_string(),
            BitSpanKind::Gap(n) => format!("gap={n}"),
        }
    }

    /// Generate tooltip text for bitstream view
    fn tooltip_text(&self, ts: u32, value: i32) -> String {
        let ts_str = format_timestamp(ts);
        match &self.0 {
            BitSpanKind::HeaderBaseTs(v) => format!("base_ts: {} ({})", v, format_timestamp(*v)),
            BitSpanKind::HeaderCount(v) => format!("count: {v} readings"),
            BitSpanKind::HeaderPrevLogicalIdx(v) => format!("prev_logical_idx: {v}"),
            BitSpanKind::HeaderFirstValue(v) => format!("first_value: {v}"),
            BitSpanKind::HeaderPrevValue(v) => format!("prev_value: {v}"),
            BitSpanKind::HeaderCurrentValue(v) => format!("current_value: {v}"),
            BitSpanKind::HeaderZeroRun(v) => format!("zero_run: {v}"),
            BitSpanKind::HeaderBitCount(v) => format!("bit_count: {v}"),
            BitSpanKind::HeaderBitAccum(v) => format!("bit_accum: {v:#04x}"),
            BitSpanKind::Zero => format!("{ts_str} val={value} (=0)"),
            BitSpanKind::ZeroRun8_21(n) | BitSpanKind::ZeroRun22_149(n) => {
                format!("{ts_str} val={value} run={n}")
            }
            BitSpanKind::Delta1(d) | BitSpanKind::Delta2(d) | BitSpanKind::Delta3_10(d) | BitSpanKind::LargeDelta(d) => {
                format!("{ts_str} val={value} ({d:+})")
            }
            BitSpanKind::SingleGap => "gap: 1 interval".to_string(),
            BitSpanKind::Gap(n) => format!("gap: {n} intervals"),
        }
    }

    /// Generate expanded decoded readings for hover display
    fn decoded_readings(&self, start_ts: u32, value: i32, interval: u16) -> Vec<String> {
        match &self.0 {
            BitSpanKind::HeaderBaseTs(_)
            | BitSpanKind::HeaderCount(_)
            | BitSpanKind::HeaderPrevLogicalIdx(_)
            | BitSpanKind::HeaderFirstValue(_)
            | BitSpanKind::HeaderPrevValue(_)
            | BitSpanKind::HeaderCurrentValue(_)
            | BitSpanKind::HeaderZeroRun(_)
            | BitSpanKind::HeaderBitCount(_)
            | BitSpanKind::HeaderBitAccum(_) => vec![],
            BitSpanKind::Zero => {
                vec![format!("{} → {}", format_timestamp(start_ts), value)]
            }
            BitSpanKind::ZeroRun8_21(n) | BitSpanKind::ZeroRun22_149(n) => {
                let mut readings = Vec::new();
                for i in 0..u32::from(*n) {
                    let ts = start_ts + i * u32::from(interval);
                    readings.push(format!("{} → {}", format_timestamp(ts), value));
                }
                readings
            }
            BitSpanKind::Delta1(d) | BitSpanKind::Delta2(d) | BitSpanKind::Delta3_10(d) | BitSpanKind::LargeDelta(d) => {
                let new_value = value + d;
                vec![format!("{} → {}", format_timestamp(start_ts), new_value)]
            }
            BitSpanKind::SingleGap => {
                vec![format!("{} → (gap)", format_timestamp(start_ts))]
            }
            BitSpanKind::Gap(n) => {
                let mut readings = Vec::new();
                for i in 0..u32::from(*n) {
                    let ts = start_ts + i * u32::from(interval);
                    readings.push(format!("{} → (gap)", format_timestamp(ts)));
                }
                readings
            }
        }
    }

    fn is_header(&self) -> bool {
        matches!(
            &self.0,
            BitSpanKind::HeaderBaseTs(_)
                | BitSpanKind::HeaderCount(_)
                | BitSpanKind::HeaderPrevLogicalIdx(_)
                | BitSpanKind::HeaderFirstValue(_)
                | BitSpanKind::HeaderPrevValue(_)
                | BitSpanKind::HeaderCurrentValue(_)
                | BitSpanKind::HeaderZeroRun(_)
                | BitSpanKind::HeaderBitCount(_)
                | BitSpanKind::HeaderBitAccum(_)
        )
    }

    fn is_gap(&self) -> bool {
        matches!(&self.0, BitSpanKind::SingleGap | BitSpanKind::Gap(_))
    }

    /// Get delta value if this is a delta span
    fn delta(&self) -> Option<i32> {
        match &self.0 {
            BitSpanKind::Delta1(d) | BitSpanKind::Delta2(d) | BitSpanKind::Delta3_10(d) | BitSpanKind::LargeDelta(d) => {
                Some(*d)
            }
            _ => None,
        }
    }

    /// Get run length if this is a zero run
    fn run_length(&self) -> Option<u32> {
        match &self.0 {
            BitSpanKind::ZeroRun8_21(n) | BitSpanKind::ZeroRun22_149(n) => Some(*n),
            _ => None,
        }
    }

    /// Get gap length if this is a gap
    fn gap_length(&self) -> Option<u32> {
        match &self.0 {
            BitSpanKind::SingleGap => Some(1),
            BitSpanKind::Gap(n) => Some(*n),
            _ => None,
        }
    }
}

/// Wrapper for BitSpan with visualization-specific SpanKind
struct VizSpan {
    id: usize,
    start_bit: usize,
    length: usize,
    kind: SpanKind,
}

impl From<BitSpan> for VizSpan {
    fn from(span: BitSpan) -> Self {
        VizSpan {
            id: span.id,
            start_bit: span.start_bit,
            length: span.length,
            kind: SpanKind(span.kind),
        }
    }
}

#[derive(Debug)]
enum RowKind {
    Header { span_id: usize },
    Event { span_id: Option<usize>, show_bits: bool },
    Gap { span_id: usize },
}

#[derive(Debug)]
struct Row {
    kind: RowKind,
    left_text: String,
    /// Ground truth comparison (original CSV value if available, and whether it matches)
    ground_truth: Option<GroundTruth>,
}

#[derive(Debug)]
struct GroundTruth {
    original_ts: u32,
    original_value: i8,
    decoded_ts: u32,
    decoded_value: i8,
    ts_matches: bool,
    value_matches: bool,
}

fn build_rows(
    header_spans: &[VizSpan],
    data_spans: &[VizSpan],
    base_ts: u32,
    first_value: i32,
    interval: u16,
    csv_readings: Option<&[CsvReading]>,
) -> Vec<Row> {
    let mut rows = Vec::new();

    // Header rows
    for span in header_spans {
        rows.push(Row {
            kind: RowKind::Header { span_id: span.id },
            left_text: span.kind.label(),
            ground_truth: None,
        });
    }

    // First reading (from header)
    let first_value_i8 = first_value as i8;
    let first_gt = csv_readings.and_then(|readings| {
        readings.first().map(|r| GroundTruth {
            original_ts: r.ts,
            original_value: r.value,
            decoded_ts: base_ts,
            decoded_value: first_value_i8,
            ts_matches: r.ts == base_ts,
            value_matches: r.value == first_value_i8,
        })
    });
    rows.push(Row {
        kind: RowKind::Event { span_id: None, show_bits: false },
        left_text: format_event(0, base_ts, first_value, "(header)"),
        ground_truth: first_gt,
    });

    // Data rows
    let mut value = first_value;
    let mut idx: u32 = 1;
    let mut event_num = 1;
    let mut csv_idx = 1usize;

    for span in data_spans {
        if span.kind.is_gap() {
            let gap_len = span.kind.gap_length().unwrap_or(1);
            rows.push(Row {
                kind: RowKind::Gap { span_id: span.id },
                left_text: if gap_len == 1 {
                    "─── gap: 1 interval ───".to_string()
                } else {
                    format!("─── gap: {gap_len} intervals ───")
                },
                ground_truth: None,
            });
            idx += u32::from(gap_len);
        } else if let Some(run_len) = span.kind.run_length() {
            // Zero run
            let value_i8 = value as i8;
            for i in 0..run_len {
                let ts = base_ts + idx * u32::from(interval);
                let show_bits = i == 0;
                let gt = csv_readings.and_then(|readings| {
                    readings.get(csv_idx).map(|r| GroundTruth {
                        original_ts: r.ts,
                        original_value: r.value,
                        decoded_ts: ts,
                        decoded_value: value_i8,
                        ts_matches: r.ts == ts,
                        value_matches: r.value == value_i8,
                    })
                });
                rows.push(Row {
                    kind: RowKind::Event { span_id: Some(span.id), show_bits },
                    left_text: format_event(event_num, ts, value, &format!("(run {}/{run_len})", i + 1)),
                    ground_truth: gt,
                });
                event_num += 1;
                idx += 1;
                csv_idx += 1;
            }
        } else if let Some(delta) = span.kind.delta() {
            // Delta
            value += delta;
            let ts = base_ts + idx * u32::from(interval);
            let value_i8 = value as i8;
            let gt = csv_readings.and_then(|readings| {
                readings.get(csv_idx).map(|r| GroundTruth {
                    original_ts: r.ts,
                    original_value: r.value,
                    decoded_ts: ts,
                    decoded_value: value_i8,
                    ts_matches: r.ts == ts,
                    value_matches: r.value == value_i8,
                })
            });
            rows.push(Row {
                kind: RowKind::Event { span_id: Some(span.id), show_bits: true },
                left_text: format_event(event_num, ts, value, &format!("({delta:+})")),
                ground_truth: gt,
            });
            event_num += 1;
            idx += 1;
            csv_idx += 1;
        } else if matches!(span.kind.0, BitSpanKind::Zero) {
            // Single zero delta
            let ts = base_ts + idx * u32::from(interval);
            let value_i8 = value as i8;
            let gt = csv_readings.and_then(|readings| {
                readings.get(csv_idx).map(|r| GroundTruth {
                    original_ts: r.ts,
                    original_value: r.value,
                    decoded_ts: ts,
                    decoded_value: value_i8,
                    ts_matches: r.ts == ts,
                    value_matches: r.value == value_i8,
                })
            });
            rows.push(Row {
                kind: RowKind::Event { span_id: Some(span.id), show_bits: true },
                left_text: format_event(event_num, ts, value, "(=0)"),
                ground_truth: gt,
            });
            event_num += 1;
            idx += 1;
            csv_idx += 1;
        }
    }

    rows
}

fn format_event(num: usize, ts: u32, value: i32, delta: &str) -> String {
    let time_str = format_timestamp(ts);
    format!("#{num:<3} {time_str}  val={value:<4} {delta}")
}

/// Format a unix timestamp as human-readable date/time
fn format_timestamp(ts: u32) -> String {
    let secs_per_day: u32 = 86400;
    let days_since_epoch = ts / secs_per_day;
    let time_of_day = ts % secs_per_day;

    let hours = time_of_day / 3600;
    let mins = (time_of_day % 3600) / 60;
    let secs = time_of_day % 60;

    let (year, month, day) = days_to_ymd(days_since_epoch);

    format!("{year:04}-{month:02}-{day:02} {hours:02}:{mins:02}:{secs:02}")
}

/// Convert days since 1970-01-01 to (year, month, day)
fn days_to_ymd(days: u32) -> (u32, u32, u32) {
    let mut remaining_days = i64::from(days);
    let mut year: u32 = 1970;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_months: [i64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month: u32 = 1;
    for days_in_month in days_in_months {
        if remaining_days < days_in_month {
            break;
        }
        remaining_days -= days_in_month;
        month += 1;
    }

    let day = remaining_days as u32 + 1;
    (year, month, day)
}

const fn is_leap_year(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn get_bits_str(bytes: &[u8], start_bit: usize, length: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(length);
    for i in 0..length {
        let bit_idx = start_bit + i;
        let byte_idx = bit_idx / 8;
        let bit_pos = 7 - (bit_idx % 8);
        if byte_idx < bytes.len() {
            bits.push((bytes[byte_idx] >> bit_pos) & 1);
        }
    }
    bits
}

/// Stats for CSV compression summary
struct CompressionStats {
    raw_size: usize,
    compressed_size: usize,
    num_readings: usize,
}

impl CompressionStats {
    fn ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            0.0
        } else {
            self.raw_size as f64 / self.compressed_size as f64
        }
    }
}

fn render_svg(
    bytes: &[u8],
    header_spans: &[VizSpan],
    data_spans: &[VizSpan],
    rows: &[Row],
    has_ground_truth: bool,
    compression_stats: Option<&CompressionStats>,
    base_ts: u32,
    first_value: i32,
    interval: u16,
) -> String {
    let num_rows = rows.len();
    let legend_height = 500;
    let stats_height = if compression_stats.is_some() { 60 } else { 0 };
    let content_height = MARGIN * 2 + num_rows * ROW_HEIGHT + 40 + stats_height;

    let total_bits: usize = header_spans.iter().chain(data_spans.iter())
        .map(|s| s.length).sum();
    let bits_per_row = BITSTREAM_MAX_WIDTH / BITSTREAM_BIT_SIZE;
    let bitstream_rows = total_bits.div_ceil(bits_per_row);
    let bitstream_height = bitstream_rows * BITSTREAM_BIT_SIZE + BITSTREAM_Y_OFFSET + MARGIN;

    let main_content_height = content_height.max(legend_height + MARGIN * 2 + stats_height);
    let total_height = main_content_height + bitstream_height;
    let legend_x = if has_ground_truth { LEGEND_X + GT_WIDTH } else { LEGEND_X };
    let total_width = legend_x + LEGEND_WIDTH + MARGIN;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" font-family="monospace" font-size="11">
  <style>
    .row-bg {{ fill: transparent; }}
    .row-bg:hover {{ fill: rgba(25, 118, 210, 0.1); }}
    .bit {{ stroke: {}; stroke-width: 0.5; }}
    .highlight .bit {{ stroke: {} !important; stroke-width: 2 !important; }}
    .event-text {{ fill: #333; }}
    .highlight .event-text {{ fill: {}; font-weight: bold; }}
    .gap-row {{ fill: #F3E5F5; }}
    .header-row {{ fill: #E3F2FD; }}
    .section-title {{ font-size: 12px; font-weight: bold; fill: #333; }}
    .legend-title {{ font-size: 13px; font-weight: bold; fill: #333; }}
    .legend-item {{ font-size: 10px; fill: #333; }}
    .legend-pattern {{ font-size: 9px; fill: #666; font-family: monospace; }}
    .legend-box {{ stroke: #ccc; stroke-width: 0.5; }}
    .gt-match {{ fill: #4CAF50; }}
    .gt-mismatch {{ fill: #F44336; font-weight: bold; }}
    .gt-ts {{ fill: #666; font-size: 9px; }}
    .gt-ts-mismatch {{ fill: #FF9800; font-size: 9px; }}
    .stats-box {{ fill: #F5F5F5; stroke: #ddd; stroke-width: 1; }}
    .stats-title {{ font-size: 12px; font-weight: bold; fill: #333; }}
    .stats-text {{ font-size: 11px; fill: #333; }}
    .stats-highlight {{ font-size: 13px; font-weight: bold; fill: #1976D2; }}
    .hover-info-box {{ fill: #FAFAFA; stroke: #ddd; stroke-width: 1; }}
  </style>
  <rect width="100%" height="100%" fill="white"/>
"#,
        total_width, total_height, colors::BORDER, colors::HIGHLIGHT, colors::HIGHLIGHT
    );

    // JavaScript for hover interactions
    svg.push_str(r#"  <script type="text/javascript"><![CDATA[
    function highlight(spanId) {
      document.querySelectorAll('.span-' + spanId).forEach(el => {
        el.classList.add('highlight');
      });
    }
    function unhighlight(spanId) {
      document.querySelectorAll('.span-' + spanId).forEach(el => {
        el.classList.remove('highlight');
      });
    }
    function showInfo(encoding, bits, readings) {
      var container = document.getElementById('hover-info-content');
      if (!container) return;
      var html = '<div style="font-size:11px;color:#333;margin-bottom:4px"><b>' + encoding + '</b></div>';
      html += '<div style="font-size:9px;color:#1976D2;font-family:monospace;margin-bottom:6px">' + bits + '</div>';
      html += '<div style="font-size:10px;color:#333;line-height:1.4">';
      var lines = readings.split('|');
      for (var i = 0; i < lines.length && i < 20; i++) {
        html += lines[i] + '<br/>';
      }
      if (lines.length > 20) {
        html += '... (' + (lines.length - 20) + ' more)';
      }
      html += '</div>';
      container.innerHTML = html;
    }
    function clearInfo() {
      var container = document.getElementById('hover-info-content');
      if (container) {
        container.innerHTML = '<div style="font-size:11px;color:#999">Hover over bits to see decoded values</div>';
      }
    }
  ]]></script>
"#);

    // Column headers
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" class="section-title">DECODED</text>
  <text x="{}" y="{}" class="section-title">BINARY</text>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#,
        MARGIN, MARGIN + stats_height + 12,
        RIGHT_X, MARGIN + stats_height + 12,
        RIGHT_X - 10, MARGIN + stats_height, RIGHT_X - 10, total_height - MARGIN, "#ccc"
    ));

    // Ground truth column header
    if has_ground_truth {
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="section-title">CSV (ground truth)</text>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#,
            GT_X, MARGIN + stats_height + 12,
            GT_X - 10, MARGIN + stats_height, GT_X - 10, total_height - MARGIN, "#ccc"
        ));
    }

    // Render compression stats box
    if let Some(stats) = compression_stats {
        let box_width = total_width - MARGIN * 2;
        svg.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="{}" height="50" class="stats-box" rx="4"/>
  <text x="{}" y="{}" class="stats-title">COMPRESSION STATS</text>
  <text x="{}" y="{}" class="stats-text">Raw: {} bytes ({} readings × 8 bytes)</text>
  <text x="{}" y="{}" class="stats-text">Compressed: {} bytes</text>
  <text x="{}" y="{}" class="stats-highlight">Ratio: {:.1}x ({:.1}% of original)</text>
"#,
            MARGIN, MARGIN, box_width,
            MARGIN + 10, MARGIN + 18,
            MARGIN + 10, MARGIN + 34, stats.raw_size, stats.num_readings,
            MARGIN + 280, MARGIN + 34, stats.compressed_size,
            MARGIN + 450, MARGIN + 34, stats.ratio(), 100.0 / stats.ratio(),
        ));
    }

    let start_y = MARGIN + stats_height + 30;

    // Build a map from span_id to span for quick lookup
    let all_spans: Vec<_> = header_spans.iter().chain(data_spans.iter()).collect();
    let span_map: std::collections::HashMap<usize, &VizSpan> =
        all_spans.iter().map(|s| (s.id, *s)).collect();

    for (row_idx, row) in rows.iter().enumerate() {
        let y = start_y + row_idx * ROW_HEIGHT;
        let (span_id, show_bits) = match &row.kind {
            RowKind::Header { span_id } => (Some(*span_id), true),
            RowKind::Event { span_id, show_bits } => (*span_id, *show_bits),
            RowKind::Gap { span_id } => (Some(*span_id), true),
        };

        let span_class = span_id.map(|id| format!("span-{id}")).unwrap_or_default();
        let hover_handlers = span_id
            .map(|id| format!(r#"onmouseover="highlight({id})" onmouseout="unhighlight({id})""#))
            .unwrap_or_default();

        let row_bg_class = match &row.kind {
            RowKind::Header { .. } => "header-row",
            RowKind::Gap { .. } => "gap-row",
            _ => "row-bg",
        };

        svg.push_str(&format!(
            r#"  <g class="{}" {}>
    <rect x="{}" y="{}" width="{}" height="{}" class="{}"/>
"#,
            span_class, hover_handlers,
            MARGIN - 5, y - ROW_HEIGHT / 2 - 2,
            total_width - MARGIN * 2 + 10, ROW_HEIGHT - 2,
            row_bg_class
        ));

        // Left side: event text
        svg.push_str(&format!(
            r#"    <text x="{}" y="{}" class="event-text">{}</text>
"#,
            MARGIN, y, row.left_text
        ));

        // Right side: bits
        if let Some(span_id) = span_id {
            if show_bits {
                if let Some(span) = span_map.get(&span_id) {
                    let bits = get_bits_str(bytes, span.start_bit, span.length);
                    let mut x = RIGHT_X;

                    for bit in &bits {
                        let color = span.kind.color();
                        svg.push_str(&format!(
                            r#"    <rect x="{}" y="{}" width="{}" height="{}" fill="{}" class="bit"/>
    <text x="{}" y="{}" text-anchor="middle" font-size="9">{}</text>
"#,
                            x, y - BIT_SIZE + 2, BIT_SIZE, BIT_SIZE, color,
                            x + BIT_SIZE / 2, y - 2, bit
                        ));
                        x += BIT_SIZE + BIT_GAP;
                    }
                }
            } else {
                svg.push_str(&format!(
                    r#"    <text x="{}" y="{}" font-size="9" fill="{}">(covered by run above)</text>
"#,
                    RIGHT_X, y, "#999"
                ));
            }
        } else {
            svg.push_str(&format!(
                r#"    <text x="{}" y="{}" font-size="9" fill="{}">(from header)</text>
"#,
                RIGHT_X, y, "#666"
            ));
        }

        // Render ground truth comparison
        if let Some(gt) = &row.ground_truth {
            let value_class = if gt.value_matches { "gt-match" } else { "gt-mismatch" };

            let value_text = if gt.value_matches {
                format!("val={} ✓", gt.original_value)
            } else {
                format!("val={} (≠{})", gt.original_value, gt.decoded_value)
            };

            let ts_text = if gt.ts_matches {
                format_timestamp(gt.original_ts)
            } else {
                let delta = gt.original_ts as i64 - gt.decoded_ts as i64;
                let delta_str = if delta > 0 {
                    format!("+{delta}s")
                } else {
                    format!("{delta}s")
                };
                format!("{} ({})", format_timestamp(gt.original_ts), delta_str)
            };
            let ts_class = if gt.ts_matches { "gt-ts" } else { "gt-ts-mismatch" };

            svg.push_str(&format!(
                r#"    <text x="{}" y="{}" class="{}">{}</text>
    <text x="{}" y="{}" class="{}">{}</text>
"#,
                GT_X, y, value_class, value_text,
                GT_X + 90, y, ts_class, ts_text
            ));
        }

        svg.push_str("  </g>\n");
    }

    // Render encoding rules legend
    svg.push_str(&render_legend(legend_x));

    // Render bitstream view section
    let bitstream_y_start = main_content_height;
    let (bitstream_svg, _) = render_bitstream(
        header_spans,
        data_spans,
        bytes,
        base_ts,
        first_value,
        interval,
        bitstream_y_start,
    );
    svg.push_str(&bitstream_svg);

    svg.push_str("</svg>\n");
    svg
}

fn render_legend(legend_x: usize) -> String {
    let mut legend = String::new();
    let x = legend_x;
    let mut y = MARGIN;
    let bg_color = "#FAFAFA";
    let line_color = "#ddd";

    legend.push_str(&format!(
        r#"  <rect x="{}" y="{}" width="{}" height="480" fill="{}" class="legend-box" rx="4"/>
"#,
        x - 10, y - 5, LEGEND_WIDTH, bg_color
    ));

    legend.push_str(&format!(
        r#"  <text x="{}" y="{}" class="legend-title">ENCODING RULES</text>
"#,
        x, y + 12
    ));
    y += 30;

    legend.push_str(&format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#,
        x - 10, MARGIN, x - 10, MARGIN + 480, line_color
    ));

    // Header section
    legend.push_str(&format!(
        r#"  <text x="{x}" y="{y}" class="legend-item" font-weight="bold">Header ({HEADER_SIZE} bytes)</text>
"#
    ));
    y += 16;

    let header_fields = [
        ("base_ts_offset", "4 bytes", colors::HEADER),
        ("count", "2 bytes", colors::HEADER),
        ("prev_logical_idx", "2 bytes", colors::HEADER),
        ("first_value", "V::BYTES", colors::HEADER),
        ("prev_value", "V::BYTES", colors::HEADER),
        ("current_value", "V::BYTES", colors::HEADER),
        ("zero_run", "1 byte", colors::HEADER),
        ("bit_count", "1 byte", colors::HEADER),
        ("bit_accum", "1 byte", colors::HEADER),
    ];

    for (name, size, color) in header_fields {
        legend.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="12" height="12" fill="{}" class="legend-box"/>
  <text x="{}" y="{}" class="legend-item">{}: {}</text>
"#,
            x, y - 10, color,
            x + 16, y, name, size
        ));
        y += 16;
    }

    y += 10;

    // Data encodings section
    legend.push_str(&format!(
        r#"  <text x="{x}" y="{y}" class="legend-item" font-weight="bold">Bit-Packed Data</text>
"#
    ));
    y += 18;

    let encodings = [
        ("0", "Delta = 0 (unchanged)", "1 bit", colors::ZERO),
        ("10x", "Delta = ±1", "3 bits", colors::DELTA_1),
        ("110", "Single-interval gap", "3 bits", colors::GAP),
        ("1110x", "Delta = ±2", "5 bits", colors::DELTA_2),
        ("11110xxxx", "Run 8-21 zeros", "9 bits", colors::ZERO_RUN_8_21),
        ("111110xxxxxxx", "Run 22-149 zeros", "13 bits", colors::ZERO_RUN_22_149),
        ("1111110xxxx", "Delta = ±3 to ±10", "11 bits", colors::DELTA_3_10),
        ("11111110xxxxxxxxxxx", "Delta = ±11 to ±1023", "19 bits", colors::LARGE_DELTA),
        ("11111111xxxxxx", "Gap (2-65 intervals)", "14 bits", colors::GAP),
    ];

    for (pattern, desc, bits, color) in encodings {
        legend.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="12" height="12" fill="{}" class="legend-box"/>
  <text x="{}" y="{}" class="legend-item">{}</text>
  <text x="{}" y="{}" class="legend-pattern">{}</text>
  <text x="{}" y="{}" class="legend-pattern" text-anchor="end">{}</text>
"#,
            x, y - 10, color,
            x + 16, y, desc,
            x + 16, y + 11, pattern,
            x + LEGEND_WIDTH - 20, y + 11, bits
        ));
        y += 32;
    }

    y += 10;

    // Notes section
    legend.push_str(&format!(
        r#"  <text x="{x}" y="{y}" class="legend-item" font-weight="bold">Notes</text>
"#
    ));
    y += 16;

    let notes = [
        "• x = sign bit (0=+, 1=-)",
        "• Runs 1-7 use individual 0 bits",
        "• Large gaps use multiple markers",
    ];

    for note in notes {
        legend.push_str(&format!(
            r#"  <text x="{x}" y="{y}" class="legend-item">{note}</text>
"#
        ));
        y += 14;
    }

    legend
}

fn render_bitstream(
    header_spans: &[VizSpan],
    data_spans: &[VizSpan],
    bytes: &[u8],
    base_ts: u32,
    first_value: i32,
    interval: u16,
    y_start: usize,
) -> (String, usize) {
    let mut svg = String::new();
    let max_width = BITSTREAM_MAX_WIDTH;
    let bit_size = BITSTREAM_BIT_SIZE;
    let info_panel_x = MARGIN + max_width + 20;

    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" class="section-title">BITSTREAM VIEW</text>
"#,
        MARGIN, y_start + 15
    ));

    let panel_height = 300;
    svg.push_str(&format!(
        r#"  <rect x="{}" y="{}" width="280" height="{}" class="hover-info-box" rx="4"/>
  <foreignObject x="{}" y="{}" width="270" height="{}">
    <div xmlns="http://www.w3.org/1999/xhtml" id="hover-info-content" style="font-family:monospace;padding:8px;overflow-y:auto;height:{}px">
      <div style="font-size:11px;color:#999">Hover over bits to see decoded values</div>
    </div>
  </foreignObject>
"#,
        info_panel_x, y_start + BITSTREAM_Y_OFFSET, panel_height,
        info_panel_x + 5, y_start + BITSTREAM_Y_OFFSET + 5, panel_height - 10, panel_height - 20,
    ));

    let mut x = MARGIN;
    let mut y = y_start + BITSTREAM_Y_OFFSET;
    let mut current_ts = base_ts;
    let mut current_value = first_value;
    let mut idx: u32 = 0;

    let all_spans: Vec<&VizSpan> = header_spans.iter().chain(data_spans.iter()).collect();

    for span in all_spans {
        let tooltip = span.kind.tooltip_text(current_ts, current_value);
        let color = span.kind.color();
        let span_class = format!("span-{}", span.id);
        let encoding_label = span.kind.label();

        let bits_str = get_bits_str(bytes, span.start_bit, span.length);
        let bits_display: String = bits_str.iter().map(|b| if *b == 0 { '0' } else { '1' }).collect();

        let readings = span.kind.decoded_readings(current_ts, current_value, interval);
        let readings_str = readings.join("|");

        let encoding_escaped = encoding_label.replace('\'', "\\'");
        let readings_escaped = readings_str.replace('\'', "\\'");

        let hover_handlers = format!(
            r#"onmouseover="highlight({});showInfo('{}','{}','{}')" onmouseout="unhighlight({});clearInfo()""#,
            span.id, encoding_escaped, bits_display, readings_escaped, span.id
        );

        for bit_offset in 0..span.length {
            if x + bit_size > MARGIN + max_width && x > MARGIN {
                x = MARGIN;
                y += bit_size;
            }

            let bit_idx = span.start_bit + bit_offset;
            let byte_idx = bit_idx / 8;
            let bit_in_byte = 7 - (bit_idx % 8);
            let bit_val = if byte_idx < bytes.len() {
                (bytes[byte_idx] >> bit_in_byte) & 1
            } else {
                0
            };

            svg.push_str(&format!(
                r#"  <g class="{}" {}>
    <rect x="{}" y="{}" width="{}" height="{}" fill="{}" class="bit"/>
    <text x="{}" y="{}" text-anchor="middle" font-size="8">{}</text>
    <title>{}</title>
  </g>
"#,
                span_class, hover_handlers,
                x, y, bit_size, bit_size, color,
                x + bit_size / 2, y + bit_size / 2 + 3, bit_val,
                tooltip
            ));

            x += bit_size;
        }

        // Update state for next span's tooltip
        if !span.kind.is_header() {
            if let Some(run_len) = span.kind.run_length() {
                idx += u32::from(run_len);
                current_ts = base_ts + idx * u32::from(interval);
            } else if let Some(delta) = span.kind.delta() {
                current_value += delta;
                idx += 1;
                current_ts = base_ts + idx * u32::from(interval);
            } else if let Some(gap_len) = span.kind.gap_length() {
                idx += u32::from(gap_len);
                current_ts = base_ts + idx * u32::from(interval);
            } else if matches!(span.kind.0, BitSpanKind::Zero) {
                idx += 1;
                current_ts = base_ts + idx * u32::from(interval);
            }
        }
    }

    let total_height = (y - y_start) + bit_size + MARGIN;
    (svg, total_height)
}

/// Read a CSV file with ts,temperature columns and encode it
fn read_csv_and_encode(path: &PathBuf) -> Result<(Vec<u8>, Vec<CsvReading>), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open CSV: {e}"))?;
    let reader = BufReader::new(file);

    let mut encoder: Encoder<i8, DEFAULT_INTERVAL> = Encoder::new();
    let mut original_readings = Vec::new();
    let mut line_num = 0;

    for line in reader.lines() {
        line_num += 1;
        let line = line.map_err(|e| format!("Read error at line {line_num}: {e}"))?;
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with("timestamp") || trimmed.starts_with("ts") || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let ts: u32 = parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid timestamp at line {}: '{}'", line_num, parts[0]))?;

        let temp_i32: i32 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid temperature at line {}: '{}'", line_num, parts[1]))?;

        if temp_i32 == -1000 {
            continue;
        }

        let temp = temp_i32 as i8;
        original_readings.push(CsvReading { ts, value: temp });

        encoder
            .append(ts, temp)
            .map_err(|e| format!("Encode error at line {line_num}: {e:?}"))?;
    }

    if original_readings.is_empty() {
        return Err("No valid readings found in CSV".to_string());
    }

    eprintln!("Encoded {} readings from CSV", original_readings.len());
    Ok((encoder.to_bytes(), original_readings))
}

/// Extract base_ts and first_value from header for visualization
fn parse_header_info(bytes: &[u8]) -> (u32, i32, u16) {
    if bytes.len() < HEADER_SIZE {
        return (0, 0, DEFAULT_INTERVAL);
    }
    // New header layout: base_ts at offset 0
    let base_ts = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    // first_value at offset 8 (for i8, it's just 1 byte but we read it properly)
    let first_value = i32::from(bytes[8] as i8);
    // interval is no longer stored in header, use default
    (base_ts, first_value, DEFAULT_INTERVAL)
}

fn main() {
    let args = Args::parse();

    let is_csv = args
        .input
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"));

    let (bytes, csv_readings) = if is_csv {
        match read_csv_and_encode(&args.input) {
            Ok((b, readings)) => (b, Some(readings)),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        (fs::read(&args.input).expect("Failed to read input file"), None)
    };

    if bytes.len() < HEADER_SIZE {
        eprintln!("Error: File too small to contain header ({} bytes, need {})", bytes.len(), HEADER_SIZE);
        std::process::exit(1);
    }

    // Use our local decode_with_spans function
    let (_readings, header_spans_raw, data_spans_raw): (Vec<Reading<i8>>, _, _) =
        decode_with_spans::<i8, DEFAULT_INTERVAL>(&bytes);

    // Convert to VizSpan for rendering
    let header_spans: Vec<VizSpan> = header_spans_raw.into_iter().map(VizSpan::from).collect();
    let data_spans: Vec<VizSpan> = data_spans_raw.into_iter().map(VizSpan::from).collect();

    let (base_ts, first_value, interval) = parse_header_info(&bytes);

    let rows = build_rows(
        &header_spans,
        &data_spans,
        base_ts,
        first_value,
        interval,
        csv_readings.as_deref(),
    );

    let has_ground_truth = csv_readings.is_some();
    let compression_stats = csv_readings.as_ref().map(|readings| CompressionStats {
        raw_size: readings.len() * 8, // (u32, i8) with alignment padding
        compressed_size: bytes.len(),
        num_readings: readings.len(),
    });
    let svg = render_svg(
        &bytes,
        &header_spans,
        &data_spans,
        &rows,
        has_ground_truth,
        compression_stats.as_ref(),
        base_ts,
        first_value,
        interval,
    );

    let output = args.output.unwrap_or_else(|| {
        let mut p = args.input.clone();
        p.set_extension("svg");
        p
    });

    let mut file = File::create(&output).expect("Failed to create output file");
    file.write_all(svg.as_bytes())
        .expect("Failed to write SVG");

    println!("Generated: {}", output.display());
    println!(
        "Rows: {} ({} header + {} events/gaps)",
        rows.len(),
        header_spans.len(),
        rows.len() - header_spans.len()
    );

    if has_ground_truth {
        let mut mismatches = 0;
        let mut ts_mismatches = 0;
        for row in &rows {
            if let Some(gt) = &row.ground_truth {
                if !gt.value_matches {
                    mismatches += 1;
                }
                if !gt.ts_matches {
                    ts_mismatches += 1;
                }
            }
        }
        if mismatches == 0 && ts_mismatches == 0 {
            println!("Ground truth: All values and timestamps match ✓");
        } else {
            println!(
                "Ground truth: {mismatches} value mismatches, {ts_mismatches} timestamp mismatches"
            );
        }
    }
}
