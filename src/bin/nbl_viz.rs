//! Visualize nibblerun encoded data as interactive SVG.

use clap::Parser;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

const EPOCH_BASE: u64 = 1_760_000_000;
const HEADER_SIZE: usize = 14;

// Layout constants
#[allow(dead_code)]
const LEFT_WIDTH: usize = 280;
const RIGHT_X: usize = 300;
const ROW_HEIGHT: usize = 24;
const BIT_SIZE: usize = 14;
const BIT_GAP: usize = 2;
const MARGIN: usize = 20;
const LEGEND_WIDTH: usize = 280;
const LEGEND_X: usize = 820;

#[derive(Parser)]
#[command(name = "nbl-viz")]
#[command(about = "Visualize nibblerun encoded data as interactive SVG")]
struct Args {
    /// Input .nbl file
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
    HeaderDuration(u16),
    HeaderCount(u16),
    HeaderFirstValue(i32),
    HeaderInterval(u16),
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
    fn color(&self) -> &'static str {
        match self {
            SpanKind::HeaderBaseTs(_)
            | SpanKind::HeaderDuration(_)
            | SpanKind::HeaderCount(_)
            | SpanKind::HeaderFirstValue(_)
            | SpanKind::HeaderInterval(_) => colors::HEADER,
            SpanKind::Zero => colors::ZERO,
            SpanKind::ZeroRun8_21(_) => colors::ZERO_RUN_8_21,
            SpanKind::ZeroRun22_149(_) => colors::ZERO_RUN_22_149,
            SpanKind::Delta1(_) => colors::DELTA_1,
            SpanKind::Delta2(_) => colors::DELTA_2,
            SpanKind::Delta3_10(_) => colors::DELTA_3_10,
            SpanKind::LargeDelta(_) => colors::LARGE_DELTA,
            SpanKind::SingleGap => colors::GAP,
            SpanKind::Gap(_) => colors::GAP,
        }
    }

    fn label(&self) -> String {
        match self {
            SpanKind::HeaderBaseTs(v) => format!("base_ts: {v}"),
            SpanKind::HeaderDuration(v) => format!("duration: {v}"),
            SpanKind::HeaderCount(v) => format!("count: {v}"),
            SpanKind::HeaderFirstValue(v) => format!("first_val: {v}"),
            SpanKind::HeaderInterval(v) => format!("interval: {v}s"),
            SpanKind::Zero => "=0".to_string(),
            SpanKind::ZeroRun8_21(n) => format!("run={n}"),
            SpanKind::ZeroRun22_149(n) => format!("run={n}"),
            SpanKind::Delta1(d) => format!("{d:+}"),
            SpanKind::Delta2(d) => format!("{d:+}"),
            SpanKind::Delta3_10(d) => format!("{d:+}"),
            SpanKind::LargeDelta(d) => format!("{d:+}"),
            SpanKind::SingleGap => "gap=1".to_string(),
            SpanKind::Gap(n) => format!("gap={n}"),
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
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
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

    fn has_more(&self) -> bool {
        self.bit_pos / 8 < self.bytes.len()
    }

    fn pos(&self) -> usize {
        self.bit_pos
    }
}

fn parse_header(bytes: &[u8]) -> (Vec<BitSpan>, u64, u16, u16, i32, u16) {
    let mut spans = Vec::new();
    let mut id = 0;

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

    // duration: 2 bytes = 16 bits
    let duration = u16::from_le_bytes([bytes[4], bytes[5]]);
    spans.push(BitSpan {
        id,
        start_bit: 32,
        length: 16,
        kind: SpanKind::HeaderDuration(duration),
    });
    id += 1;

    // count: 2 bytes = 16 bits
    let count = u16::from_le_bytes([bytes[6], bytes[7]]);
    spans.push(BitSpan {
        id,
        start_bit: 48,
        length: 16,
        kind: SpanKind::HeaderCount(count),
    });
    id += 1;

    // first_value: 4 bytes = 32 bits
    let first_value = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    spans.push(BitSpan {
        id,
        start_bit: 64,
        length: 32,
        kind: SpanKind::HeaderFirstValue(first_value),
    });
    id += 1;

    // interval: 2 bytes = 16 bits
    let interval = u16::from_le_bytes([bytes[12], bytes[13]]);
    spans.push(BitSpan {
        id,
        start_bit: 96,
        length: 16,
        kind: SpanKind::HeaderInterval(interval),
    });

    (spans, base_ts, duration, count, first_value, interval)
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
    first_value: i32,
    interval: u16,
) -> Vec<Row> {
    let mut rows = Vec::new();

    // Header rows
    for span in header_spans {
        rows.push(Row {
            kind: RowKind::Header { span_id: span.id },
            left_text: span.kind.label(),
        });
    }

    // First reading (from header)
    rows.push(Row {
        kind: RowKind::Event { span_id: None, show_bits: false },
        left_text: format_event(0, base_ts, first_value, "(header)"),
    });

    // Data rows
    let mut value = first_value;
    let mut idx: u64 = 1;
    let mut event_num = 1;

    for span in data_spans {
        match &span.kind {
            SpanKind::Zero => {
                let ts = base_ts + idx * interval as u64;
                rows.push(Row {
                    kind: RowKind::Event { span_id: Some(span.id), show_bits: true },
                    left_text: format_event(event_num, ts, value, "(=0)"),
                });
                event_num += 1;
                idx += 1;
            }
            SpanKind::ZeroRun8_21(n) | SpanKind::ZeroRun22_149(n) => {
                for i in 0..*n {
                    let ts = base_ts + idx * interval as u64;
                    // Only show bits on first row of run
                    let show_bits = i == 0;
                    rows.push(Row {
                        kind: RowKind::Event { span_id: Some(span.id), show_bits },
                        left_text: format_event(event_num, ts, value, &format!("(run {}/{})", i + 1, n)),
                    });
                    event_num += 1;
                    idx += 1;
                }
            }
            SpanKind::Delta1(d) | SpanKind::Delta2(d) | SpanKind::Delta3_10(d) | SpanKind::LargeDelta(d) => {
                value += d;
                let ts = base_ts + idx * interval as u64;
                rows.push(Row {
                    kind: RowKind::Event { span_id: Some(span.id), show_bits: true },
                    left_text: format_event(event_num, ts, value, &format!("({d:+})")),
                });
                event_num += 1;
                idx += 1;
            }
            SpanKind::SingleGap => {
                rows.push(Row {
                    kind: RowKind::Gap { span_id: span.id },
                    left_text: "─── gap: 1 interval ───".to_string(),
                });
                idx += 1;
            }
            SpanKind::Gap(n) => {
                rows.push(Row {
                    kind: RowKind::Gap { span_id: span.id },
                    left_text: format!("─── gap: {} intervals ───", n),
                });
                idx += *n as u64;
            }
            _ => {}
        }
    }

    rows
}

fn format_event(num: usize, ts: u64, value: i32, delta: &str) -> String {
    let offset = ts.saturating_sub(EPOCH_BASE);
    let hours = offset / 3600;
    let mins = (offset % 3600) / 60;
    format!("#{:<3} {:02}:{:02}  val={:<4} {}", num, hours, mins, value, delta)
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

fn render_svg(bytes: &[u8], header_spans: &[BitSpan], data_spans: &[BitSpan], rows: &[Row]) -> String {
    let num_rows = rows.len();
    let legend_height = 460; // Fixed height for legend (9 encoding types)
    let content_height = MARGIN * 2 + num_rows * ROW_HEIGHT + 40;
    let total_height = content_height.max(legend_height + MARGIN * 2);
    let total_width = LEGEND_X + LEGEND_WIDTH + MARGIN;

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
  ]]></script>
"#);

    // Column headers
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" class="section-title">EVENTS</text>
  <text x="{}" y="{}" class="section-title">BINARY</text>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#,
        MARGIN, MARGIN + 12,
        RIGHT_X, MARGIN + 12,
        RIGHT_X - 10, MARGIN, RIGHT_X - 10, total_height - MARGIN, "#ccc"
    ));

    let start_y = MARGIN + 30;

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

        svg.push_str("  </g>\n");
    }

    // Render encoding rules legend
    svg.push_str(&render_legend());

    svg.push_str("</svg>\n");
    svg
}

fn render_legend() -> String {
    let mut legend = String::new();
    let x = LEGEND_X;
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
        r#"  <text x="{}" y="{}" class="legend-item" font-weight="bold">Header (14 bytes)</text>
"#,
        x, y
    ));
    y += 16;

    let header_fields = [
        ("base_ts_offset", "4 bytes", colors::HEADER),
        ("duration", "2 bytes", colors::HEADER),
        ("count", "2 bytes", colors::HEADER),
        ("first_value", "4 bytes", colors::HEADER),
        ("interval", "2 bytes", colors::HEADER),
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
        r#"  <text x="{}" y="{}" class="legend-item" font-weight="bold">Bit-Packed Data</text>
"#,
        x, y
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
        r#"  <text x="{}" y="{}" class="legend-item" font-weight="bold">Notes</text>
"#,
        x, y
    ));
    y += 16;

    let notes = [
        "• x = sign bit (0=+, 1=-)",
        "• Runs 1-7 use individual 0 bits",
        "• Large gaps use multiple markers",
    ];

    for note in notes {
        legend.push_str(&format!(
            r#"  <text x="{}" y="{}" class="legend-item">{}</text>
"#,
            x, y, note
        ));
        y += 14;
    }

    legend
}

fn main() {
    let args = Args::parse();

    let bytes = fs::read(&args.input).expect("Failed to read input file");

    if bytes.len() < HEADER_SIZE {
        eprintln!("Error: File too small to contain header");
        std::process::exit(1);
    }

    let (header_spans, base_ts, _duration, count, first_value, interval) = parse_header(&bytes);
    let data_spans = parse_data(&bytes, count as usize, header_spans.len());
    let rows = build_rows(&header_spans, &data_spans, base_ts, first_value, interval);

    let svg = render_svg(&bytes, &header_spans, &data_spans, &rows);

    let output = args.output.unwrap_or_else(|| {
        let mut p = args.input.clone();
        p.set_extension("svg");
        p
    });

    let mut file = File::create(&output).expect("Failed to create output file");
    file.write_all(svg.as_bytes())
        .expect("Failed to write SVG");

    println!("Generated: {}", output.display());
    println!("Rows: {} ({} header + {} events/gaps)", rows.len(), header_spans.len(), rows.len() - header_spans.len());
}
