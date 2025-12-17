//! Visualize nibblerun encoded data as interactive SVG.

use clap::Parser;
use nibblerun::Encoder;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Original reading from CSV for ground truth comparison
#[derive(Debug, Clone)]
struct CsvReading {
    ts: u64,
    value: i8,
}

const EPOCH_BASE: u64 = 1_760_000_000;
const HEADER_SIZE: usize = 7; // 4 (base_ts_offset) + 2 (count) + 1 (first_value as i8)
const DEFAULT_INTERVAL: u16 = 300;

// Layout constants
#[allow(dead_code)]
const LEFT_WIDTH: usize = 280;
const RIGHT_X: usize = 340;
const ROW_HEIGHT: usize = 24;
const BIT_SIZE: usize = 14;
const BIT_GAP: usize = 2;
const MARGIN: usize = 20;
const LEGEND_WIDTH: usize = 280;
const LEGEND_X: usize = 860;
// Ground truth column (when CSV input)
const GT_X: usize = 660;
const GT_WIDTH: usize = 300;

// Bitstream view constants
const BITSTREAM_Y_OFFSET: usize = 40;   // Space above bitstream section
const BITSTREAM_BIT_SIZE: usize = 12;   // Square bit size (width = height)
const BITSTREAM_MAX_WIDTH: usize = 192; // 16 bits per row (16 * 12 = 192)

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

#[derive(Debug, Clone)]
enum SpanKind {
    HeaderBaseTs(u64),
    HeaderCount(u16),
    HeaderFirstValue(i8),
    Zero,
    ZeroRun8_21(u32),
    ZeroRun22_149(u32),
    Delta1(i32),
    Delta2(i32),
    Delta3_10(i32),
    LargeDelta(i32),
    SingleGap,
    Gap(u32),
}

impl SpanKind {
    const fn color(&self) -> &'static str {
        match self {
            Self::HeaderBaseTs(_)
            | Self::HeaderCount(_)
            | Self::HeaderFirstValue(_) => colors::HEADER,
            Self::Zero => colors::ZERO,
            Self::ZeroRun8_21(_) => colors::ZERO_RUN_8_21,
            Self::ZeroRun22_149(_) => colors::ZERO_RUN_22_149,
            Self::Delta1(_) => colors::DELTA_1,
            Self::Delta2(_) => colors::DELTA_2,
            Self::Delta3_10(_) => colors::DELTA_3_10,
            Self::LargeDelta(_) => colors::LARGE_DELTA,
            Self::SingleGap => colors::GAP,
            Self::Gap(_) => colors::GAP,
        }
    }

    fn label(&self) -> String {
        match self {
            Self::HeaderBaseTs(v) => format!("base_ts: {v}"),
            Self::HeaderCount(v) => format!("count: {v}"),
            Self::HeaderFirstValue(v) => format!("first_val: {v}"),
            Self::Zero => "=0".to_string(),
            Self::ZeroRun8_21(n) => format!("run={n}"),
            Self::ZeroRun22_149(n) => format!("run={n}"),
            Self::Delta1(d) => format!("{d:+}"),
            Self::Delta2(d) => format!("{d:+}"),
            Self::Delta3_10(d) => format!("{d:+}"),
            Self::LargeDelta(d) => format!("{d:+}"),
            Self::SingleGap => "gap=1".to_string(),
            Self::Gap(n) => format!("gap={n}"),
        }
    }

    /// Generate tooltip text for bitstream view
    fn tooltip_text(&self, ts: u64, value: i32) -> String {
        let ts_str = format_timestamp(ts);
        match self {
            Self::HeaderBaseTs(v) => format!("base_ts: {} ({})", v, format_timestamp(*v)),
            Self::HeaderCount(v) => format!("count: {v} readings"),
            Self::HeaderFirstValue(v) => format!("first_value: {v}"),
            Self::Zero => format!("{ts_str} val={value} (=0)"),
            Self::ZeroRun8_21(n) => format!("{ts_str} val={value} run={n}"),
            Self::ZeroRun22_149(n) => format!("{ts_str} val={value} run={n}"),
            Self::Delta1(d) => format!("{ts_str} val={value} ({d:+})"),
            Self::Delta2(d) => format!("{ts_str} val={value} ({d:+})"),
            Self::Delta3_10(d) => format!("{ts_str} val={value} ({d:+})"),
            Self::LargeDelta(d) => format!("{ts_str} val={value} ({d:+})"),
            Self::SingleGap => "gap: 1 interval".to_string(),
            Self::Gap(n) => format!("gap: {n} intervals"),
        }
    }

    /// Generate expanded decoded readings for hover display
    /// Returns a list of "timestamp → value" strings
    fn decoded_readings(&self, start_ts: u64, value: i32, interval: u16) -> Vec<String> {
        match self {
            Self::HeaderBaseTs(_)
            | Self::HeaderCount(_)
            | Self::HeaderFirstValue(_) => vec![],
            Self::Zero => {
                vec![format!("{} → {}", format_timestamp(start_ts), value)]
            }
            Self::ZeroRun8_21(n) | Self::ZeroRun22_149(n) => {
                let mut readings = Vec::new();
                for i in 0..u64::from(*n) {
                    let ts = start_ts + i * u64::from(interval);
                    readings.push(format!("{} → {}", format_timestamp(ts), value));
                }
                readings
            }
            Self::Delta1(d) | Self::Delta2(d) | Self::Delta3_10(d) | Self::LargeDelta(d) => {
                let new_value = value + d;
                vec![format!("{} → {}", format_timestamp(start_ts), new_value)]
            }
            Self::SingleGap => {
                vec![format!("{} → (gap)", format_timestamp(start_ts))]
            }
            Self::Gap(n) => {
                let mut readings = Vec::new();
                for i in 0..u64::from(*n) {
                    let ts = start_ts + i * u64::from(interval);
                    readings.push(format!("{} → (gap)", format_timestamp(ts)));
                }
                readings
            }
        }
    }
}

#[derive(Debug)]
struct BitSpan {
    id: usize,
    start_bit: usize,
    length: usize,
    kind: SpanKind,
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
    original_ts: u64,
    original_value: i8,
    decoded_ts: u64,
    decoded_value: i8,
    ts_matches: bool,
    value_matches: bool,
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn read_bits(&mut self, n: usize) -> u32 {
        let mut result = 0u32;
        for _ in 0..n {
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            if byte_idx < self.bytes.len() {
                let bit = (self.bytes[byte_idx] >> bit_idx) & 1;
                result = (result << 1) | u32::from(bit);
            }
            self.bit_pos += 1;
        }
        result
    }

    const fn has_more(&self) -> bool {
        self.bit_pos / 8 < self.bytes.len()
    }

    const fn pos(&self) -> usize {
        self.bit_pos
    }
}

fn parse_header(bytes: &[u8]) -> (Vec<BitSpan>, u64, u16, i8) {
    let mut spans = Vec::new();
    let mut id = 0;

    // Header layout (7 bytes / 56 bits):
    // [0-3]: base_ts_offset (32 bits)
    // [4-5]: count (16 bits)
    // [6]: first_value (8 bits, i8)

    // base_ts_offset: 4 bytes = 32 bits
    let base_ts_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let base_ts = EPOCH_BASE + u64::from(base_ts_offset);
    spans.push(BitSpan {
        id,
        start_bit: 0,
        length: 32,
        kind: SpanKind::HeaderBaseTs(base_ts),
    });
    id += 1;

    // count: 2 bytes = 16 bits
    let count = u16::from_le_bytes([bytes[4], bytes[5]]);
    spans.push(BitSpan {
        id,
        start_bit: 32,
        length: 16,
        kind: SpanKind::HeaderCount(count),
    });
    id += 1;

    // first_value: 1 byte = 8 bits (i8)
    let first_value = bytes[6] as i8;
    spans.push(BitSpan {
        id,
        start_bit: 48,
        length: 8,
        kind: SpanKind::HeaderFirstValue(first_value),
    });

    (spans, base_ts, count, first_value)
}

fn parse_data(bytes: &[u8], count: usize, start_id: usize) -> Vec<BitSpan> {
    let mut spans = Vec::new();
    let data = &bytes[HEADER_SIZE..];
    let mut reader = BitReader::new(data);
    let header_bits = HEADER_SIZE * 8;
    let mut id = start_id;

    let mut readings_decoded = 1; // First reading is in header

    // New encoding scheme:
    // 0       = zero delta (1 bit)
    // 100     = +1, 101 = -1 (3 bits)
    // 110     = single-interval gap (3 bits)
    // 11100   = +2, 11101 = -2 (5 bits)
    // 11110xxxx = zero run 8-21 (9 bits)
    // 111110xxxxxxx = zero run 22-149 (13 bits)
    // 1111110xxxx = ±3-10 delta (11 bits)
    // 11111110xxxxxxxxxxx = large delta (19 bits)
    // 11111111xxxxxx = gap 2-65 (14 bits)

    while readings_decoded < count && reader.has_more() {
        let start = reader.pos();

        // Read first bit
        if reader.read_bits(1) == 0 {
            // 0 = zero delta
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 1,
                kind: SpanKind::Zero,
            });
            id += 1;
            readings_decoded += 1;
            continue;
        }

        // Read second bit
        if reader.read_bits(1) == 0 {
            // 10x = ±1 delta
            let sign = reader.read_bits(1);
            let delta = if sign == 0 { 1 } else { -1 };
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 3,
                kind: SpanKind::Delta1(delta),
            });
            id += 1;
            readings_decoded += 1;
            continue;
        }

        // Read third bit
        if reader.read_bits(1) == 0 {
            // 110 = single-interval gap (3 bits)
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 3,
                kind: SpanKind::SingleGap,
            });
            id += 1;
            // Gap doesn't add readings
            continue;
        }

        // 111...
        // Read fourth bit
        if reader.read_bits(1) == 0 {
            // 1110x = ±2 delta (5 bits)
            let sign = reader.read_bits(1);
            let delta = if sign == 0 { 2 } else { -2 };
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 5,
                kind: SpanKind::Delta2(delta),
            });
            id += 1;
            readings_decoded += 1;
            continue;
        }

        // 1111...
        // Read fifth bit
        if reader.read_bits(1) == 0 {
            // 11110xxxx = zero run 8-21 (9 bits)
            let n = reader.read_bits(4) + 8;
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 9,
                kind: SpanKind::ZeroRun8_21(n),
            });
            id += 1;
            readings_decoded += n as usize;
            continue;
        }

        // 11111...
        // Read sixth bit
        if reader.read_bits(1) == 0 {
            // 111110xxxxxxx = zero run 22-149 (13 bits)
            let n = reader.read_bits(7) + 22;
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 13,
                kind: SpanKind::ZeroRun22_149(n),
            });
            id += 1;
            readings_decoded += n as usize;
            continue;
        }

        // 111111...
        // Read seventh bit
        if reader.read_bits(1) == 0 {
            // 1111110xxxx = ±3-10 delta (11 bits)
            let e = reader.read_bits(4) as i32;
            let delta = if e < 8 { e - 10 } else { e - 5 };
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 11,
                kind: SpanKind::Delta3_10(delta),
            });
            id += 1;
            readings_decoded += 1;
            continue;
        }

        // 1111111...
        // Read eighth bit
        if reader.read_bits(1) == 0 {
            // 11111110xxxxxxxxxxx = large delta (19 bits)
            let raw = reader.read_bits(11);
            let delta = if raw & 0x400 != 0 {
                (raw | 0xFFFF_F800) as i32
            } else {
                raw as i32
            };
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 19,
                kind: SpanKind::LargeDelta(delta),
            });
            id += 1;
            readings_decoded += 1;
        } else {
            // 11111111xxxxxx = gap 2-65 intervals (14 bits)
            let n = reader.read_bits(6) + 2;
            spans.push(BitSpan {
                id,
                start_bit: header_bits + start,
                length: 14,
                kind: SpanKind::Gap(n),
            });
            id += 1;
        }
    }

    spans
}

fn build_rows(
    header_spans: &[BitSpan],
    data_spans: &[BitSpan],
    base_ts: u64,
    first_value: i8,
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
    let first_gt = csv_readings.and_then(|readings| {
        readings.first().map(|r| GroundTruth {
            original_ts: r.ts,
            original_value: r.value,
            decoded_ts: base_ts,
            decoded_value: first_value,
            ts_matches: r.ts == base_ts,
            value_matches: r.value == first_value,
        })
    });
    let first_value_i32 = i32::from(first_value);
    rows.push(Row {
        kind: RowKind::Event { span_id: None, show_bits: false },
        left_text: format_event(0, base_ts, first_value_i32, "(header)"),
        ground_truth: first_gt,
    });

    // Data rows - use i32 internally for delta arithmetic, cast to i8 for comparisons
    let mut value = first_value_i32;
    let mut idx: u64 = 1;
    let mut event_num = 1;
    let mut csv_idx = 1usize; // Start at 1, since 0 was used for header

    for span in data_spans {
        match &span.kind {
            SpanKind::Zero => {
                let ts = base_ts + idx * u64::from(interval);
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
            SpanKind::ZeroRun8_21(n) | SpanKind::ZeroRun22_149(n) => {
                let value_i8 = value as i8;
                for i in 0..*n {
                    let ts = base_ts + idx * u64::from(interval);
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
                        left_text: format_event(event_num, ts, value, &format!("(run {}/{})", i + 1, n)),
                        ground_truth: gt,
                    });
                    event_num += 1;
                    idx += 1;
                    csv_idx += 1;
                }
            }
            SpanKind::Delta1(d) | SpanKind::Delta2(d) | SpanKind::Delta3_10(d) | SpanKind::LargeDelta(d) => {
                value += d;
                let ts = base_ts + idx * u64::from(interval);
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
                    left_text: format_event(event_num, ts, value, &format!("({d:+})")),
                    ground_truth: gt,
                });
                event_num += 1;
                idx += 1;
                csv_idx += 1;
            }
            SpanKind::SingleGap => {
                rows.push(Row {
                    kind: RowKind::Gap { span_id: span.id },
                    left_text: "─── gap: 1 interval ───".to_string(),
                    ground_truth: None,
                });
                idx += 1;
            }
            SpanKind::Gap(n) => {
                rows.push(Row {
                    kind: RowKind::Gap { span_id: span.id },
                    left_text: format!("─── gap: {n} intervals ───"),
                    ground_truth: None,
                });
                idx += u64::from(*n);
            }
            _ => {}
        }
    }

    rows
}

fn format_event(num: usize, ts: u64, value: i32, delta: &str) -> String {
    let time_str = format_timestamp(ts);
    format!("#{num:<3} {time_str}  val={value:<4} {delta}")
}

/// Format a unix timestamp as human-readable date/time
fn format_timestamp(ts: u64) -> String {
    // Simple date calculation from unix timestamp
    // Days since 1970-01-01
    let secs_per_day: u64 = 86400;
    let days_since_epoch = ts / secs_per_day;
    let time_of_day = ts % secs_per_day;

    let hours = time_of_day / 3600;
    let mins = (time_of_day % 3600) / 60;
    let secs = time_of_day % 60;

    // Calculate year/month/day from days since epoch
    // Using a simple algorithm for Gregorian calendar
    let (year, month, day) = days_to_ymd(days_since_epoch);

    format!(
        "{year:04}-{month:02}-{day:02} {hours:02}:{mins:02}:{secs:02}"
    )
}

/// Convert days since 1970-01-01 to (year, month, day)
fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    let mut remaining_days = days as i64;

    // Start from 1970
    let mut year: u32 = 1970;

    // Count years
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    // Count months
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
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
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
    header_spans: &[BitSpan],
    data_spans: &[BitSpan],
    rows: &[Row],
    has_ground_truth: bool,
    compression_stats: Option<&CompressionStats>,
    base_ts: u64,
    first_value: i8,
    interval: u16,
) -> String {
    let num_rows = rows.len();
    let legend_height = 460; // Fixed height for legend (9 encoding types)
    let stats_height = if compression_stats.is_some() { 60 } else { 0 };
    let content_height = MARGIN * 2 + num_rows * ROW_HEIGHT + 40 + stats_height;

    // Calculate bitstream section height
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

    // Ground truth column header (if CSV input)
    if has_ground_truth {
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="section-title">CSV (ground truth)</text>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#,
            GT_X, MARGIN + stats_height + 12,
            GT_X - 10, MARGIN + stats_height, GT_X - 10, total_height - MARGIN, "#ccc"
        ));
    }

    // Render compression stats box (if CSV input)
    if let Some(stats) = compression_stats {
        let box_width = total_width - MARGIN * 2;
        svg.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="{}" height="50" class="stats-box" rx="4"/>
  <text x="{}" y="{}" class="stats-title">COMPRESSION STATS</text>
  <text x="{}" y="{}" class="stats-text">Raw: {} bytes ({} readings × 12 bytes)</text>
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

    // Build a map from span_id to BitSpan for quick lookup
    let all_spans: Vec<_> = header_spans.iter().chain(data_spans.iter()).collect();
    let span_map: std::collections::HashMap<usize, &BitSpan> =
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

        // Row background for hover effect
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
                // Part of a run - show indicator
                svg.push_str(&format!(
                    r#"    <text x="{}" y="{}" font-size="9" fill="{}">(covered by run above)</text>
"#,
                    RIGHT_X, y, "#999"
                ));
            }
        } else {
            // First event from header - no bits, just label
            svg.push_str(&format!(
                r#"    <text x="{}" y="{}" font-size="9" fill="{}">(from header)</text>
"#,
                RIGHT_X, y, "#666"
            ));
        }

        // Render ground truth comparison (if available)
        if let Some(gt) = &row.ground_truth {
            let value_class = if gt.value_matches { "gt-match" } else { "gt-mismatch" };

            // Show value comparison
            let value_text = if gt.value_matches {
                format!("val={} ✓", gt.original_value)
            } else {
                format!("val={} (≠{})", gt.original_value, gt.decoded_value)
            };

            // Format timestamp with delta if mismatched
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

    // Legend box background
    legend.push_str(&format!(
        r#"  <rect x="{}" y="{}" width="{}" height="440" fill="{}" class="legend-box" rx="4"/>
"#,
        x - 10, y - 5, LEGEND_WIDTH, bg_color
    ));

    // Title
    legend.push_str(&format!(
        r#"  <text x="{}" y="{}" class="legend-title">ENCODING RULES</text>
"#,
        x, y + 12
    ));
    y += 30;

    // Vertical separator line
    legend.push_str(&format!(
        r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#,
        x - 10, MARGIN, x - 10, MARGIN + 440, line_color
    ));

    // Header section
    legend.push_str(&format!(
        r#"  <text x="{x}" y="{y}" class="legend-item" font-weight="bold">Header (10 bytes)</text>
"#
    ));
    y += 16;

    let header_fields = [
        ("base_ts_offset", "4 bytes", colors::HEADER),
        ("count", "2 bytes", colors::HEADER),
        ("first_value", "1 byte", colors::HEADER),
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

/// Render the bitstream view section showing all bits as individual squares
fn render_bitstream(
    header_spans: &[BitSpan],
    data_spans: &[BitSpan],
    bytes: &[u8],
    base_ts: u64,
    first_value: i8,
    interval: u16,
    y_start: usize,
) -> (String, usize) {
    let mut svg = String::new();
    let max_width = BITSTREAM_MAX_WIDTH;
    let bit_size = BITSTREAM_BIT_SIZE;
    let info_panel_x = MARGIN + max_width + 20;

    // Section title
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" class="section-title">BITSTREAM VIEW</text>
"#,
        MARGIN, y_start + 15
    ));

    // Hover info panel (to the right of bitstream) using foreignObject for HTML
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
    let mut current_value = i32::from(first_value);
    let mut idx: u64 = 0;

    // Combine header and data spans
    let all_spans: Vec<&BitSpan> = header_spans.iter().chain(data_spans.iter()).collect();

    for span in all_spans {
        // Calculate tooltip text based on span kind
        let tooltip = span.kind.tooltip_text(current_ts, current_value);
        let color = span.kind.color();
        let span_class = format!("span-{}", span.id);
        let encoding_label = span.kind.label();

        // Get bit pattern for this span
        let bits_str = get_bits_str(bytes, span.start_bit, span.length);
        let bits_display: String = bits_str.iter().map(|b| if *b == 0 { '0' } else { '1' }).collect();

        // Get decoded readings for this span
        let readings = span.kind.decoded_readings(current_ts, current_value, interval);
        let readings_str = readings.join("|");

        // Escape quotes for JavaScript
        let encoding_escaped = encoding_label.replace('\'', "\\'");
        let readings_escaped = readings_str.replace('\'', "\\'");

        let hover_handlers = format!(
            r#"onmouseover="highlight({});showInfo('{}','{}','{}')" onmouseout="unhighlight({});clearInfo()""#,
            span.id, encoding_escaped, bits_display, readings_escaped, span.id
        );

        // Render each bit as individual square
        for bit_offset in 0..span.length {
            // Check if we need to wrap to next row
            if x + bit_size > MARGIN + max_width && x > MARGIN {
                x = MARGIN;
                y += bit_size;
            }

            // Get the actual bit value
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
        match &span.kind {
            SpanKind::HeaderBaseTs(_)
            | SpanKind::HeaderCount(_)
            | SpanKind::HeaderFirstValue(_) => {
                // Header spans don't affect timestamp/value tracking
            }
            SpanKind::Zero => {
                idx += 1;
                current_ts = base_ts + idx * u64::from(interval);
            }
            SpanKind::ZeroRun8_21(n) | SpanKind::ZeroRun22_149(n) => {
                idx += u64::from(*n);
                current_ts = base_ts + idx * u64::from(interval);
            }
            SpanKind::Delta1(d) | SpanKind::Delta2(d) | SpanKind::Delta3_10(d) | SpanKind::LargeDelta(d) => {
                current_value += d;
                idx += 1;
                current_ts = base_ts + idx * u64::from(interval);
            }
            SpanKind::SingleGap => {
                idx += 1;
                current_ts = base_ts + idx * u64::from(interval);
            }
            SpanKind::Gap(n) => {
                idx += u64::from(*n);
                current_ts = base_ts + idx * u64::from(interval);
            }
        }
    }

    // Calculate total height used
    let total_height = (y - y_start) + bit_size + MARGIN;

    (svg, total_height)
}

/// Read a CSV file with ts,temperature columns and encode it
/// Returns (`encoded_bytes`, `original_readings`)
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

        // Skip empty lines and header
        if trimmed.is_empty() || trimmed.starts_with("ts") || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let ts: u64 = parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid timestamp at line {}: '{}'", line_num, parts[0]))?;

        let temp_i32: i32 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid temperature at line {}: '{}'", line_num, parts[1]))?;

        // Skip sentinel values (gaps in original data)
        if temp_i32 == -1000 {
            continue;
        }

        // Convert to i8 for encoding (values must be in i8 range: -128 to 127)
        let temp = temp_i32 as i8;

        // Store original reading for ground truth comparison
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

fn main() {
    let args = Args::parse();

    // Determine if input is CSV or NBL based on extension
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
        eprintln!("Error: File too small to contain header");
        std::process::exit(1);
    }

    let (header_spans, base_ts, count, first_value) = parse_header(&bytes);
    let interval = DEFAULT_INTERVAL;
    let data_spans = parse_data(&bytes, count as usize, header_spans.len());
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
        raw_size: readings.len() * 12, // 8 bytes timestamp + 4 bytes value
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

    // Summary of ground truth comparison
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
