//! Benchmark compression algorithms against nibblerun.
//!
//! Compares: nibblerun appendable, nibblerun freeze, tsz (gorilla), lz4, zstd-19, pco-4

use clap::Parser;
use std::collections::HashMap;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "nbl-bench-others")]
#[command(about = "Benchmark compression algorithms against nibblerun")]
struct Args {
    /// Directory containing CSV files
    #[arg(long)]
    dir: PathBuf,

    /// Maximum files to process (0 = all)
    #[arg(long, default_value = "1000")]
    max_files: usize,

    /// Generate HTML report with graphs at this path
    #[arg(long)]
    html_report: Option<PathBuf>,
}

/// Load readings from CSV files into a HashMap of device_id -> Vec<(timestamp, value)>
/// Filters out -1000 sentinel values.
fn load_readings(dir: &Path, max_files: usize) -> HashMap<u32, Vec<(u32, i8)>> {
    let entries: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "csv"))
        .collect();

    let mut result: HashMap<u32, Vec<(u32, i8)>> = HashMap::new();

    for (idx, entry) in entries.into_iter().enumerate() {
        if max_files > 0 && result.len() >= max_files {
            break;
        }

        let path = entry.path();

        // Parse device ID from filename (hex)
        let device_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| u32::from_str_radix(s.trim(), 16).ok())
            .unwrap_or(idx as u32);

        // Parse CSV
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let reader = BufReader::new(file);
        let mut readings = Vec::new();

        for line in reader.lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("ts") || trimmed.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = trimmed.split(',').collect();
            if parts.len() < 2 {
                continue;
            }

            let ts: u32 = match parts[0].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let value: i32 = match parts[1].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Filter out -1000 sentinel values
            if value == -1000 {
                continue;
            }

            let value_i8 = value.clamp(-128, 127) as i8;
            readings.push((ts, value_i8));
        }

        if !readings.is_empty() {
            result.insert(device_id, readings);
        }
    }

    result
}

/// Statistics for a set of sizes
#[derive(Debug)]
struct Stats {
    min: usize,
    p25: usize,
    p50: usize,
    p75: usize,
    p90: usize,
    p99: usize,
    max: usize,
    avg: f64,
    total: usize,
}

fn percentile(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p) as usize;
    sorted[idx]
}

fn compute_stats(sizes: &[usize]) -> Stats {
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable();

    let total: usize = sorted.iter().sum();
    let avg = if sorted.is_empty() {
        0.0
    } else {
        total as f64 / sorted.len() as f64
    };

    Stats {
        min: *sorted.first().unwrap_or(&0),
        p25: percentile(&sorted, 0.25),
        p50: percentile(&sorted, 0.50),
        p75: percentile(&sorted, 0.75),
        p90: percentile(&sorted, 0.90),
        p99: percentile(&sorted, 0.99),
        max: *sorted.last().unwrap_or(&0),
        avg,
        total,
    }
}

/// Benchmark result for one algorithm
struct BenchResult {
    name: &'static str,
    sizes: Vec<usize>,
    raw_sizes: Vec<usize>, // Raw size per device for ratio calculation
    encode_time: Duration,
    decode_time: Duration,
}

// ============================================================================
// Compression implementations
// ============================================================================

/// nibblerun appendable format
fn bench_nibblerun_appendable(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    use nibblerun::Encoder;

    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8); // (u32, i8) with padding = 8 bytes per reading
        let mut encoder: Encoder<i8, 300> = Encoder::new();
        for &(ts, val) in readings {
            let _ = encoder.append(ts, val);
        }
        encoded.push(encoder.to_bytes());
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let decoded = nibblerun::decode_appendable::<i8, 300>(buf);
        black_box(decoded);
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "nibblerun append",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// nibblerun frozen format
fn bench_nibblerun_freeze(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    use nibblerun::Encoder;

    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        let mut encoder: Encoder<i8, 300> = Encoder::new();
        for &(ts, val) in readings {
            let _ = encoder.append(ts, val);
        }
        let frozen = encoder.freeze();
        encoded.push(frozen);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let decoded = nibblerun::decode_frozen::<i8, 300>(buf);
        black_box(decoded);
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "nibblerun freeze",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// tsz (gorilla) compression
fn bench_tsz(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    use tsz::stream::{BufferedReader, BufferedWriter};
    use tsz::{DataPoint, Decode, Encode, StdDecoder, StdEncoder};

    // Encode - store as Box<[u8]> to avoid clone during decode
    let start = Instant::now();
    let mut encoded: Vec<Box<[u8]>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        let first_ts = readings[0].0 as u64;
        let w = BufferedWriter::new();
        let mut encoder = StdEncoder::new(first_ts, w);

        for &(ts, val) in readings {
            let dp = DataPoint::new(ts as u64, val as f64);
            encoder.encode(dp);
        }

        let bytes = encoder.close();
        encoded.push(bytes);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode - no clone needed since we stored as Box<[u8]>
    let start = Instant::now();
    for buf in encoded {
        let r = BufferedReader::new(buf);
        let mut decoder = StdDecoder::new(r);
        while let Ok(dp) = decoder.next() {
            black_box(dp);
        }
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "tsz (gorilla)",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// lz4 compression (columnar format with delta-encoded timestamps)
fn bench_lz4(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        // Format: [count:u32][base_ts:u32][ts_deltas as u16 for readings[1..]...][values...]
        let count = readings.len() as u32;
        let base_ts = readings[0].0;
        let mut raw = Vec::with_capacity(8 + (readings.len() - 1) * 2 + readings.len());
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&base_ts.to_le_bytes());
        // Delta-encode timestamps starting from second reading
        let mut prev_ts = base_ts;
        for &(ts, _) in &readings[1..] {
            let delta = (ts - prev_ts) as u16;
            raw.extend_from_slice(&delta.to_le_bytes());
            prev_ts = ts;
        }
        for &(_, val) in readings {
            raw.push(val as u8);
        }
        let compressed = lz4_flex::compress_prepend_size(&raw);
        encoded.push(compressed);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let decompressed = lz4_flex::decompress_size_prepended(buf).unwrap();
        let count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        let base_ts = u32::from_le_bytes(decompressed[4..8].try_into().unwrap());
        let mut timestamps = Vec::with_capacity(count);
        timestamps.push(base_ts);
        let mut ts = base_ts;
        for i in 0..(count - 1) {
            let delta_offset = 8 + i * 2;
            let delta = u16::from_le_bytes(decompressed[delta_offset..delta_offset + 2].try_into().unwrap());
            ts = ts.wrapping_add(delta as u32);
            timestamps.push(ts);
        }
        let values_offset = 8 + (count - 1) * 2;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(decompressed[values_offset + i] as i8);
        }
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "lz4",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// zstd level 3 compression (columnar format with delta-encoded timestamps)
fn bench_zstd(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        // Format: [count:u32][base_ts:u32][ts_deltas as u16 for readings[1..]...][values...]
        let count = readings.len() as u32;
        let base_ts = readings[0].0;
        let mut raw = Vec::with_capacity(8 + (readings.len() - 1) * 2 + readings.len());
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&base_ts.to_le_bytes());
        // Delta-encode timestamps starting from second reading
        let mut prev_ts = base_ts;
        for &(ts, _) in &readings[1..] {
            let delta = (ts - prev_ts) as u16;
            raw.extend_from_slice(&delta.to_le_bytes());
            prev_ts = ts;
        }
        for &(_, val) in readings {
            raw.push(val as u8);
        }
        let compressed = zstd::encode_all(&raw[..], 3).unwrap();
        encoded.push(compressed);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let decompressed = zstd::decode_all(&buf[..]).unwrap();
        let count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        let base_ts = u32::from_le_bytes(decompressed[4..8].try_into().unwrap());
        let mut timestamps = Vec::with_capacity(count);
        timestamps.push(base_ts);
        let mut ts = base_ts;
        for i in 0..(count - 1) {
            let delta_offset = 8 + i * 2;
            let delta = u16::from_le_bytes(decompressed[delta_offset..delta_offset + 2].try_into().unwrap());
            ts = ts.wrapping_add(delta as u32);
            timestamps.push(ts);
        }
        let values_offset = 8 + (count - 1) * 2;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(decompressed[values_offset + i] as i8);
        }
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "zstd-3",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// pco level 4 compression (columnar format, designed for this)
/// Note: pco doesn't support i8, so we use i16 for values
fn bench_pco(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    use pco::standalone::{simple_decompress, simpler_compress};

    const PCO_LEVEL: usize = 4;

    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        let timestamps: Vec<u32> = readings.iter().map(|&(ts, _)| ts).collect();
        // pco doesn't support i8, use i16
        let values: Vec<i16> = readings.iter().map(|&(_, val)| val as i16).collect();

        // pco compresses each array separately, we concat them with length prefix
        let ts_compressed = simpler_compress(&timestamps, PCO_LEVEL).unwrap();
        let val_compressed = simpler_compress(&values, PCO_LEVEL).unwrap();

        let mut buf = Vec::with_capacity(8 + ts_compressed.len() + val_compressed.len());
        buf.extend_from_slice(&(ts_compressed.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(val_compressed.len() as u32).to_le_bytes());
        buf.extend_from_slice(&ts_compressed);
        buf.extend_from_slice(&val_compressed);
        encoded.push(buf);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let ts_len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let val_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let ts_data = &buf[8..8 + ts_len];
        let val_data = &buf[8 + ts_len..8 + ts_len + val_len];

        let timestamps: Vec<u32> = simple_decompress(ts_data).unwrap();
        let values: Vec<i16> = simple_decompress(val_data).unwrap();
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "pco-4",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// Snappy compression (columnar format with delta-encoded timestamps)
fn bench_snappy(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        // Format: [count:u32][base_ts:u32][ts_deltas as u16 for readings[1..]...][values...]
        let count = readings.len() as u32;
        let base_ts = readings[0].0;
        let mut raw = Vec::with_capacity(8 + (readings.len() - 1) * 2 + readings.len());
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&base_ts.to_le_bytes());
        let mut prev_ts = base_ts;
        for &(ts, _) in &readings[1..] {
            let delta = (ts - prev_ts) as u16;
            raw.extend_from_slice(&delta.to_le_bytes());
            prev_ts = ts;
        }
        for &(_, val) in readings {
            raw.push(val as u8);
        }
        let mut encoder = snap::raw::Encoder::new();
        let compressed = encoder.compress_vec(&raw).unwrap();
        encoded.push(compressed);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let mut decoder = snap::raw::Decoder::new();
        let decompressed = decoder.decompress_vec(buf).unwrap();
        let count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        let base_ts = u32::from_le_bytes(decompressed[4..8].try_into().unwrap());
        let mut timestamps = Vec::with_capacity(count);
        timestamps.push(base_ts);
        let mut ts = base_ts;
        for i in 0..(count - 1) {
            let delta_offset = 8 + i * 2;
            let delta = u16::from_le_bytes(decompressed[delta_offset..delta_offset + 2].try_into().unwrap());
            ts = ts.wrapping_add(delta as u32);
            timestamps.push(ts);
        }
        let values_offset = 8 + (count - 1) * 2;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(decompressed[values_offset + i] as i8);
        }
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "snappy",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// Raw delta encoding (no compression, just delta-encoded timestamps)
fn bench_raw_delta(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        // Format: [count:u32][base_ts:u32][ts_deltas as u16 for readings[1..]...][values...]
        // No compression - just store the delta-encoded data
        let count = readings.len() as u32;
        let base_ts = readings[0].0;
        let mut raw = Vec::with_capacity(8 + (readings.len() - 1) * 2 + readings.len());
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&base_ts.to_le_bytes());
        let mut prev_ts = base_ts;
        for &(ts, _) in &readings[1..] {
            let delta = (ts - prev_ts) as u16;
            raw.extend_from_slice(&delta.to_le_bytes());
            prev_ts = ts;
        }
        for &(_, val) in readings {
            raw.push(val as u8);
        }
        encoded.push(raw);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let base_ts = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let mut timestamps = Vec::with_capacity(count);
        timestamps.push(base_ts);
        let mut ts = base_ts;
        for i in 0..(count - 1) {
            let delta_offset = 8 + i * 2;
            let delta = u16::from_le_bytes(buf[delta_offset..delta_offset + 2].try_into().unwrap());
            ts = ts.wrapping_add(delta as u32);
            timestamps.push(ts);
        }
        let values_offset = 8 + (count - 1) * 2;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(buf[values_offset + i] as i8);
        }
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "raw delta",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// Brotli compression (columnar format with delta-encoded timestamps)
fn bench_brotli(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    use std::io::Read;

    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        // Format: [count:u32][base_ts:u32][ts_deltas as u16 for readings[1..]...][values...]
        let count = readings.len() as u32;
        let base_ts = readings[0].0;
        let mut raw = Vec::with_capacity(8 + (readings.len() - 1) * 2 + readings.len());
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&base_ts.to_le_bytes());
        let mut prev_ts = base_ts;
        for &(ts, _) in &readings[1..] {
            let delta = (ts - prev_ts) as u16;
            raw.extend_from_slice(&delta.to_le_bytes());
            prev_ts = ts;
        }
        for &(_, val) in readings {
            raw.push(val as u8);
        }
        // Brotli quality 4 (balanced speed/compression)
        let mut compressed = Vec::new();
        let mut compressor = brotli::CompressorReader::new(&raw[..], 4096, 4, 22);
        compressor.read_to_end(&mut compressed).unwrap();
        encoded.push(compressed);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let mut decompressed = Vec::new();
        let mut decompressor = brotli::Decompressor::new(&buf[..], 4096);
        decompressor.read_to_end(&mut decompressed).unwrap();
        let count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        let base_ts = u32::from_le_bytes(decompressed[4..8].try_into().unwrap());
        let mut timestamps = Vec::with_capacity(count);
        timestamps.push(base_ts);
        let mut ts = base_ts;
        for i in 0..(count - 1) {
            let delta_offset = 8 + i * 2;
            let delta = u16::from_le_bytes(decompressed[delta_offset..delta_offset + 2].try_into().unwrap());
            ts = ts.wrapping_add(delta as u32);
            timestamps.push(ts);
        }
        let values_offset = 8 + (count - 1) * 2;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(decompressed[values_offset + i] as i8);
        }
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "brotli-4",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// Deflate compression (columnar format with delta-encoded timestamps)
fn bench_deflate(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    use flate2::read::{DeflateDecoder, DeflateEncoder};
    use flate2::Compression;
    use std::io::Read;

    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        // Format: [count:u32][base_ts:u32][ts_deltas as u16 for readings[1..]...][values...]
        let count = readings.len() as u32;
        let base_ts = readings[0].0;
        let mut raw = Vec::with_capacity(8 + (readings.len() - 1) * 2 + readings.len());
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&base_ts.to_le_bytes());
        let mut prev_ts = base_ts;
        for &(ts, _) in &readings[1..] {
            let delta = (ts - prev_ts) as u16;
            raw.extend_from_slice(&delta.to_le_bytes());
            prev_ts = ts;
        }
        for &(_, val) in readings {
            raw.push(val as u8);
        }
        let mut compressed = Vec::new();
        let mut encoder = DeflateEncoder::new(&raw[..], Compression::default());
        encoder.read_to_end(&mut compressed).unwrap();
        encoded.push(compressed);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let mut decompressed = Vec::new();
        let mut decoder = DeflateDecoder::new(&buf[..]);
        decoder.read_to_end(&mut decompressed).unwrap();
        let count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        let base_ts = u32::from_le_bytes(decompressed[4..8].try_into().unwrap());
        let mut timestamps = Vec::with_capacity(count);
        timestamps.push(base_ts);
        let mut ts = base_ts;
        for i in 0..(count - 1) {
            let delta_offset = 8 + i * 2;
            let delta = u16::from_le_bytes(decompressed[delta_offset..delta_offset + 2].try_into().unwrap());
            ts = ts.wrapping_add(delta as u32);
            timestamps.push(ts);
        }
        let values_offset = 8 + (count - 1) * 2;
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(decompressed[values_offset + i] as i8);
        }
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "deflate",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

/// Delta-of-delta encoding (Prometheus/Gorilla style) with lz4 compression
/// For regular intervals, delta-of-delta is mostly 0 which compresses very well
fn bench_delta_of_delta(data: &HashMap<u32, Vec<(u32, i8)>>) -> BenchResult {
    // Encode
    let start = Instant::now();
    let mut encoded: Vec<Vec<u8>> = Vec::with_capacity(data.len());
    let mut raw_sizes: Vec<usize> = Vec::with_capacity(data.len());
    for readings in data.values() {
        if readings.is_empty() {
            continue;
        }
        raw_sizes.push(readings.len() * 8);
        // Format: [count:u32][base_ts:u32][first_delta:i16][delta_of_deltas as i16...][values...]
        // Delta-of-delta: for regular 300s intervals, most dod will be 0
        let count = readings.len() as u32;
        let base_ts = readings[0].0;
        let mut raw = Vec::with_capacity(8 + readings.len() * 3);
        raw.extend_from_slice(&count.to_le_bytes());
        raw.extend_from_slice(&base_ts.to_le_bytes());

        if readings.len() >= 2 {
            let first_delta = (readings[1].0 as i32 - readings[0].0 as i32) as i16;
            raw.extend_from_slice(&first_delta.to_le_bytes());

            let mut prev_delta = first_delta as i32;
            let mut prev_ts = readings[1].0;
            for &(ts, _) in &readings[2..] {
                let delta = ts as i32 - prev_ts as i32;
                let dod = (delta - prev_delta) as i16;
                raw.extend_from_slice(&dod.to_le_bytes());
                prev_delta = delta;
                prev_ts = ts;
            }
        }

        for &(_, val) in readings {
            raw.push(val as u8);
        }
        let compressed = lz4_flex::compress_prepend_size(&raw);
        encoded.push(compressed);
    }
    let encode_time = start.elapsed();
    black_box(&encoded);

    let sizes: Vec<usize> = encoded.iter().map(|b| b.len()).collect();

    // Decode
    let start = Instant::now();
    for buf in &encoded {
        let decompressed = lz4_flex::decompress_size_prepended(buf).unwrap();
        let count = u32::from_le_bytes(decompressed[0..4].try_into().unwrap()) as usize;
        let base_ts = u32::from_le_bytes(decompressed[4..8].try_into().unwrap());
        let mut timestamps = Vec::with_capacity(count);
        timestamps.push(base_ts);

        if count >= 2 {
            let first_delta = i16::from_le_bytes(decompressed[8..10].try_into().unwrap()) as i32;
            let second_ts = (base_ts as i32 + first_delta) as u32;
            timestamps.push(second_ts);

            let mut prev_delta = first_delta;
            let mut prev_ts = second_ts;
            for i in 2..count {
                let dod_offset = 10 + (i - 2) * 2;
                let dod = i16::from_le_bytes(decompressed[dod_offset..dod_offset + 2].try_into().unwrap()) as i32;
                let delta = prev_delta + dod;
                let ts = (prev_ts as i32 + delta) as u32;
                timestamps.push(ts);
                prev_delta = delta;
                prev_ts = ts;
            }
        }

        // Values start after header (8) + first_delta (2) + dod array ((count-2)*2)
        let values_offset = if count >= 2 { 8 + 2 + (count - 2) * 2 } else { 8 };
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            values.push(decompressed[values_offset + i] as i8);
        }
        black_box((timestamps, values));
    }
    let decode_time = start.elapsed();

    BenchResult {
        name: "dod+lz4",
        sizes,
        raw_sizes,
        encode_time,
        decode_time,
    }
}

// ============================================================================
// HTML Report Generation
// ============================================================================

/// Embedded Plotly library for standalone HTML reports
const PLOTLY_JS: &str = include_str!("plotly-basic.min.js");

/// Generate an HTML report with interactive Plotly charts
fn generate_html_report(
    results: &[BenchResult],
    raw_sizes_per_device: &[usize],
    raw_size: usize,
    total_readings: usize,
    num_devices: usize,
    output_path: &Path,
) -> std::io::Result<()> {
    use std::io::Write;

    // Build JSON data for the charts
    let mut algorithms = Vec::new();
    let mut sizes_data = Vec::new();
    let mut ratios_data = Vec::new();
    let mut summary_data = Vec::new();

    for result in results {
        let stats = compute_stats(&result.sizes);
        let ratio = raw_size as f64 / stats.total as f64;
        let encode_mbs = raw_size as f64 / result.encode_time.as_secs_f64() / 1_000_000.0;
        let decode_mbs = raw_size as f64 / result.decode_time.as_secs_f64() / 1_000_000.0;

        // Per-device ratios
        let ratios: Vec<f64> = result
            .raw_sizes
            .iter()
            .zip(result.sizes.iter())
            .map(|(&raw, &compressed)| raw as f64 / compressed as f64)
            .collect();

        algorithms.push(format!("\"{}\"", result.name));

        // Sizes as JSON array
        let sizes_json: String = result
            .sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        sizes_data.push(format!("\"{}\":[{}]", result.name, sizes_json));

        // Ratios as JSON array
        let ratios_json: String = ratios
            .iter()
            .map(|r| format!("{:.2}", r))
            .collect::<Vec<_>>()
            .join(",");
        ratios_data.push(format!("\"{}\":[{}]", result.name, ratios_json));

        // Summary stats
        summary_data.push(format!(
            "\"{}\": {{\"total_size\":{},\"ratio\":{:.2},\"encode_mbs\":{:.1},\"decode_mbs\":{:.1},\"percentiles\":{{\"min\":{},\"p25\":{},\"p50\":{},\"p75\":{},\"p90\":{},\"p99\":{},\"max\":{}}}}}",
            result.name,
            stats.total,
            ratio,
            encode_mbs,
            decode_mbs,
            stats.min,
            stats.p25,
            stats.p50,
            stats.p75,
            stats.p90,
            stats.p99,
            stats.max
        ));
    }

    // Raw baseline sizes as JSON
    let raw_sizes_json: String = raw_sizes_per_device
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compression Benchmark Report</title>
    <script>{plotly_js}</script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0d1117;
            color: #e6edf3;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 32px; }}
        header {{ margin-bottom: 32px; }}
        h1 {{ font-size: 28px; font-weight: 600; color: #f0f6fc; margin-bottom: 8px; }}
        .subtitle {{ color: #7d8590; font-size: 14px; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }}
        .metric {{
            background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: #58a6ff;
            font-variant-numeric: tabular-nums;
        }}
        .metric-value.highlight {{ color: #f85149; }}
        .metric-value.green {{ color: #3fb950; }}
        .metric-label {{
            font-size: 12px;
            color: #7d8590;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
        @media (max-width: 1000px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        .panel {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }}
        .panel.wide {{ grid-column: span 2; }}
        @media (max-width: 1000px) {{ .panel.wide {{ grid-column: span 1; }} }}
        .panel-title {{
            font-size: 14px;
            font-weight: 600;
            color: #e6edf3;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #21262d;
        }}
        .plot {{ height: 400px; }}
        .plot-tall {{ height: 500px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #21262d;
        }}
        th {{
            font-weight: 600;
            color: #7d8590;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{ color: #e6edf3; }}
        tr:hover {{ background: #1c2128; }}
        .highlight-row {{ background: rgba(248, 81, 73, 0.1); }}
        .mono {{ font-family: 'SF Mono', Monaco, monospace; }}
        .best {{ color: #3fb950; font-weight: 600; }}
        footer {{
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #21262d;
            text-align: center;
            color: #484f58;
            font-size: 12px;
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Compression Benchmark Report</h1>
        <p class="subtitle">{num_devices} devices · {total_readings} readings · {raw_size_mb:.2} MB raw (8 bytes/reading)</p>
    </header>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value highlight" id="best-ratio"></div>
            <div class="metric-label">Best Compression</div>
        </div>
        <div class="metric">
            <div class="metric-value green" id="best-size"></div>
            <div class="metric-label">Smallest Avg Size</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="best-encode"></div>
            <div class="metric-label">Fastest Encode</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="best-decode"></div>
            <div class="metric-label">Fastest Decode</div>
        </div>
    </div>

    <div class="grid">
        <div class="panel">
            <div class="panel-title">Compression Ratio (higher = better)</div>
            <div id="bar-ratio" class="plot"></div>
        </div>

        <div class="panel">
            <div class="panel-title">Encode / Decode Speed (MB/s)</div>
            <div id="bar-speed" class="plot"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Compressed Size Distribution (per device)</div>
            <div id="box-sizes" class="plot-tall"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Compression Ratio Distribution (per device)</div>
            <div id="box-ratios" class="plot-tall"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Size Percentiles</div>
            <div id="line-percentiles" class="plot"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Full Results</div>
            <table id="results-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Algorithm</th>
                        <th>Ratio</th>
                        <th>Total Size</th>
                        <th>Avg Size</th>
                        <th>p50 Size</th>
                        <th>Encode MB/s</th>
                        <th>Decode MB/s</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <footer>Generated by nbl-bench-others</footer>
</div>

<script>
    const benchmarkData = {{
        algorithms: [{algorithms}],
        sizes: {{{sizes_data}}},
        ratios: {{{ratios_data}}},
        summary: {{{summary_data}}},
        rawSizes: [{raw_sizes_json}],
        metadata: {{
            totalReadings: {total_readings},
            numDevices: {num_devices},
            rawSize: {raw_size}
        }}
    }};

    // Dark theme layout for Plotly
    const darkLayout = {{
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#161b22',
        font: {{ color: '#e6edf3' }},
        xaxis: {{ gridcolor: '#30363d', zerolinecolor: '#30363d' }},
        yaxis: {{ gridcolor: '#30363d', zerolinecolor: '#30363d' }},
        margin: {{ t: 20, r: 20, b: 60, l: 60 }}
    }};

    // Color palette
    const colors = [
        '#f85149', '#ff7b72', '#3fb950', '#58a6ff', '#a371f7',
        '#79c0ff', '#ffa657', '#7ee787', '#ff9bce', '#d2a8ff', '#8b949e'
    ];

    // Find best values for metrics
    const algos = benchmarkData.algorithms;
    let bestRatio = {{ algo: '', val: 0 }};
    let bestSize = {{ algo: '', val: Infinity }};
    let bestEncode = {{ algo: '', val: 0 }};
    let bestDecode = {{ algo: '', val: 0 }};

    algos.forEach(a => {{
        const s = benchmarkData.summary[a];
        if (s.ratio > bestRatio.val) bestRatio = {{ algo: a, val: s.ratio }};
        if (s.percentiles.p50 < bestSize.val) bestSize = {{ algo: a, val: s.percentiles.p50 }};
        if (s.encode_mbs > bestEncode.val) bestEncode = {{ algo: a, val: s.encode_mbs }};
        if (s.decode_mbs > bestDecode.val) bestDecode = {{ algo: a, val: s.decode_mbs }};
    }});

    document.getElementById('best-ratio').textContent = bestRatio.val.toFixed(1) + 'x';
    document.getElementById('best-size').textContent = bestSize.val + 'B';
    document.getElementById('best-encode').textContent = bestEncode.val.toFixed(0) + ' MB/s';
    document.getElementById('best-decode').textContent = bestDecode.val.toFixed(0) + ' MB/s';

    // 1. Bar Chart: Compression Ratio (sorted by ratio, with annotations) - include raw baseline
    const barSortedByRatio = [...algos, 'raw'].sort((a, b) => {{
        const ratioA = a === 'raw' ? 1.0 : benchmarkData.summary[a].ratio;
        const ratioB = b === 'raw' ? 1.0 : benchmarkData.summary[b].ratio;
        return ratioB - ratioA;
    }});
    const rawAvgSize = Math.round(benchmarkData.rawSizes.reduce((a, b) => a + b, 0) / benchmarkData.metadata.numDevices);
    Plotly.newPlot('bar-ratio', [{{
        x: barSortedByRatio,
        y: barSortedByRatio.map(a => a === 'raw' ? 1.0 : benchmarkData.summary[a].ratio),
        type: 'bar',
        marker: {{ color: barSortedByRatio.map((a, i) => a === 'raw' ? '#484f58' : colors[i % colors.length]) }},
        hoverinfo: 'x+y'
    }}], {{
        ...darkLayout,
        xaxis: {{ ...darkLayout.xaxis, tickangle: 45 }},
        yaxis: {{ ...darkLayout.yaxis, title: 'Compression Ratio (x)' }},
        margin: {{ t: 40, r: 20, b: 100, l: 60 }},
        showlegend: false,
        annotations: barSortedByRatio.map((algo, i) => {{
            const ratio = algo === 'raw' ? 1.0 : benchmarkData.summary[algo].ratio;
            const avgSize = algo === 'raw' ? rawAvgSize : Math.round(benchmarkData.summary[algo].total_size / benchmarkData.metadata.numDevices);
            return [
                {{ x: algo, y: ratio, yanchor: 'bottom', yshift: 22, text: ratio.toFixed(1) + 'x', showarrow: false, font: {{ color: '#e6edf3', size: 11 }} }},
                {{ x: algo, y: ratio, yanchor: 'bottom', yshift: 6, text: avgSize + 'B', showarrow: false, font: {{ color: '#7d8590', size: 10 }} }}
            ];
        }}).flat()
    }}, {{ responsive: true }});

    // 2. Grouped Bar: Encode/Decode Speed
    Plotly.newPlot('bar-speed', [
        {{
            x: algos,
            y: algos.map(a => benchmarkData.summary[a].encode_mbs),
            type: 'bar',
            name: 'Encode',
            marker: {{ color: '#58a6ff' }}
        }},
        {{
            x: algos,
            y: algos.map(a => benchmarkData.summary[a].decode_mbs),
            type: 'bar',
            name: 'Decode',
            marker: {{ color: '#3fb950' }}
        }}
    ], {{
        ...darkLayout,
        barmode: 'group',
        yaxis: {{ ...darkLayout.yaxis, title: 'Speed (MB/s)' }},
        legend: {{ orientation: 'h', y: -0.2, font: {{ color: '#e6edf3' }} }}
    }}, {{ responsive: true }});

    // 3. Box Plot: Compressed Sizes (sorted by ratio, with annotations) - include raw baseline
    const boxSortedByRatio = [...algos, 'raw'].sort((a, b) => {{
        const ratioA = a === 'raw' ? 1.0 : benchmarkData.summary[a].ratio;
        const ratioB = b === 'raw' ? 1.0 : benchmarkData.summary[b].ratio;
        return ratioB - ratioA;
    }});

    Plotly.newPlot('box-sizes', boxSortedByRatio.map((algo, i) => ({{
        y: algo === 'raw' ? benchmarkData.rawSizes : benchmarkData.sizes[algo],
        type: 'box',
        name: algo,
        marker: {{ color: algo === 'raw' ? '#484f58' : colors[i % colors.length] }},
        boxpoints: false,
        hoverinfo: 'name'
    }})), {{
        ...darkLayout,
        xaxis: {{ ...darkLayout.xaxis, tickangle: 45 }},
        yaxis: {{ ...darkLayout.yaxis, title: 'Compressed Size (bytes)', type: 'log' }},
        margin: {{ t: 40, r: 20, b: 100, l: 60 }},
        showlegend: false,
        annotations: boxSortedByRatio.flatMap((algo, i) => {{
            const sizes = algo === 'raw' ? benchmarkData.rawSizes : benchmarkData.sizes[algo];
            const ratio = algo === 'raw' ? 1.0 : benchmarkData.summary[algo].ratio;
            const avgSize = algo === 'raw' ? rawAvgSize : Math.round(benchmarkData.summary[algo].total_size / benchmarkData.metadata.numDevices);
            const maxVal = Math.max(...sizes);
            return [
                {{ x: i, y: maxVal, yanchor: 'bottom', yshift: 22, text: ratio.toFixed(1) + 'x', showarrow: false, font: {{ color: '#e6edf3', size: 11 }} }},
                {{ x: i, y: maxVal, yanchor: 'bottom', yshift: 6, text: avgSize + 'B', showarrow: false, font: {{ color: '#7d8590', size: 10 }} }}
            ];
        }})
    }}, {{ responsive: true }});

    // 4. Box Plot: Compression Ratios (sorted by ratio, with annotations) - include raw baseline
    const rawRatios = benchmarkData.rawSizes.map(() => 1.0);
    Plotly.newPlot('box-ratios', boxSortedByRatio.map((algo, i) => ({{
        y: algo === 'raw' ? rawRatios : benchmarkData.ratios[algo],
        type: 'box',
        name: algo,
        marker: {{ color: algo === 'raw' ? '#484f58' : colors[i % colors.length] }},
        boxpoints: false,
        hoverinfo: 'name'
    }})), {{
        ...darkLayout,
        xaxis: {{ ...darkLayout.xaxis, tickangle: 45 }},
        yaxis: {{ ...darkLayout.yaxis, title: 'Compression Ratio (x)' }},
        margin: {{ t: 40, r: 20, b: 100, l: 60 }},
        showlegend: false,
        annotations: boxSortedByRatio.flatMap((algo, i) => {{
            const ratios = algo === 'raw' ? rawRatios : benchmarkData.ratios[algo];
            const ratio = algo === 'raw' ? 1.0 : benchmarkData.summary[algo].ratio;
            const avgSize = algo === 'raw' ? rawAvgSize : Math.round(benchmarkData.summary[algo].total_size / benchmarkData.metadata.numDevices);
            const maxVal = Math.max(...ratios);
            return [
                {{ x: i, y: maxVal, yanchor: 'bottom', yshift: 22, text: ratio.toFixed(1) + 'x', showarrow: false, font: {{ color: '#e6edf3', size: 11 }} }},
                {{ x: i, y: maxVal, yanchor: 'bottom', yshift: 6, text: avgSize + 'B', showarrow: false, font: {{ color: '#7d8590', size: 10 }} }}
            ];
        }})
    }}, {{ responsive: true }});

    // 5. Line Chart: Percentiles
    const pLabels = ['min', 'p25', 'p50', 'p75', 'p90', 'p99', 'max'];
    Plotly.newPlot('line-percentiles', algos.map((algo, i) => {{
        const p = benchmarkData.summary[algo].percentiles;
        return {{
            x: pLabels,
            y: [p.min, p.p25, p.p50, p.p75, p.p90, p.p99, p.max],
            type: 'scatter',
            mode: 'lines+markers',
            name: algo,
            line: {{ color: colors[i % colors.length] }}
        }};
    }}), {{
        ...darkLayout,
        yaxis: {{ ...darkLayout.yaxis, title: 'Compressed Size (bytes)', type: 'log' }},
        xaxis: {{ ...darkLayout.xaxis, title: 'Percentile' }},
        legend: {{ orientation: 'h', y: -0.2, font: {{ color: '#e6edf3' }} }}
    }}, {{ responsive: true }});

    // Populate results table (sorted by ratio)
    const sorted = [...algos].sort((a, b) => benchmarkData.summary[b].ratio - benchmarkData.summary[a].ratio);
    const tbody = document.querySelector('#results-table tbody');
    sorted.forEach((algo, i) => {{
        const s = benchmarkData.summary[algo];
        const avgSize = (s.total_size / benchmarkData.metadata.numDevices).toFixed(0);
        const isBest = i === 0;
        const row = document.createElement('tr');
        if (isBest) row.classList.add('highlight-row');
        row.innerHTML = `
            <td>${{i + 1}}</td>
            <td>${{algo}}</td>
            <td class="mono ${{isBest ? 'best' : ''}}">${{s.ratio.toFixed(1)}}x</td>
            <td class="mono">${{s.total_size.toLocaleString()}}</td>
            <td class="mono">${{avgSize}}B</td>
            <td class="mono">${{s.percentiles.p50}}B</td>
            <td class="mono">${{s.encode_mbs.toFixed(1)}}</td>
            <td class="mono">${{s.decode_mbs.toFixed(1)}}</td>
        `;
        tbody.appendChild(row);
    }});
</script>
</body>
</html>"##,
        plotly_js = PLOTLY_JS,
        num_devices = num_devices,
        total_readings = total_readings,
        raw_size_mb = raw_size as f64 / (1024.0 * 1024.0),
        raw_size = raw_size,
        algorithms = algorithms.join(","),
        sizes_data = sizes_data.join(","),
        ratios_data = ratios_data.join(","),
        summary_data = summary_data.join(","),
        raw_sizes_json = raw_sizes_json,
    );

    let mut file = File::create(output_path)?;
    file.write_all(html.as_bytes())?;
    println!("\nHTML report written to: {}", output_path.display());
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args = Args::parse();

    // Expand ~ in path
    let dir = if args.dir.starts_with("~") {
        let home = std::env::var("HOME").expect("HOME not set");
        PathBuf::from(home).join(args.dir.strip_prefix("~").unwrap())
    } else {
        args.dir.clone()
    };

    println!("Loading readings from {}...", dir.display());
    let data = load_readings(&dir, args.max_files);
    let num_devices = data.len();
    let total_readings: usize = data.values().map(|v| v.len()).sum();
    println!(
        "Loaded {} readings from {} devices",
        total_readings, num_devices
    );

    // Raw size: (u32, i8) tuple with padding = 8 bytes per reading
    let raw_size = total_readings * 8;
    println!(
        "Raw size: {:.2} MB ({} readings * 8 bytes)\n",
        raw_size as f64 / (1024.0 * 1024.0),
        total_readings
    );

    // Compute raw sizes per device for baseline
    let raw_sizes_per_device: Vec<usize> = data.values().map(|v| v.len() * 8).collect();
    let raw_stats = compute_stats(&raw_sizes_per_device);

    // Run benchmarks
    let results = vec![
        bench_nibblerun_appendable(&data),
        bench_nibblerun_freeze(&data),
        bench_tsz(&data),
        bench_lz4(&data),
        bench_snappy(&data),
        bench_zstd(&data),
        bench_pco(&data),
        bench_brotli(&data),
        bench_deflate(&data),
        bench_delta_of_delta(&data),
        bench_raw_delta(&data),
    ];

    // Print header
    println!(
        "{:<18} | {:>8} | {:>8} | {:>10} | {:>7} | {:>11} | {:>11}",
        "Algorithm", "Size p50", "Size avg", "Total", "Ratio", "Encode MB/s", "Decode MB/s"
    );
    println!("{}", "-".repeat(100));

    // Print baseline (raw) first
    println!(
        "{:<18} | {:>8} | {:>8.1} | {:>10} | {:>6.1}x | {:>11} | {:>11}",
        "raw (baseline)",
        raw_stats.p50,
        raw_stats.avg,
        raw_stats.total,
        1.0,
        "-",
        "-"
    );

    // Print results
    for result in &results {
        let stats = compute_stats(&result.sizes);
        let ratio = raw_size as f64 / stats.total as f64;
        let encode_mbs = raw_size as f64 / result.encode_time.as_secs_f64() / 1_000_000.0;
        let decode_mbs = raw_size as f64 / result.decode_time.as_secs_f64() / 1_000_000.0;

        println!(
            "{:<18} | {:>8} | {:>8.1} | {:>10} | {:>6.1}x | {:>11.1} | {:>11.1}",
            result.name,
            stats.p50,
            stats.avg,
            stats.total,
            ratio,
            encode_mbs,
            decode_mbs
        );
    }

    // Print per-device compressed size percentiles
    println!("\n=== Per-Device Compressed Size (bytes) ===");
    println!("{:<18} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}",
        "Algorithm", "min", "p25", "p50", "p75", "p90", "p99", "max");
    println!("{}", "-".repeat(85));

    // Baseline raw sizes
    println!(
        "{:<18} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}",
        "raw (baseline)", raw_stats.min, raw_stats.p25, raw_stats.p50, raw_stats.p75, raw_stats.p90, raw_stats.p99, raw_stats.max
    );

    for result in &results {
        let stats = compute_stats(&result.sizes);
        println!(
            "{:<18} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}",
            result.name, stats.min, stats.p25, stats.p50, stats.p75, stats.p90, stats.p99, stats.max
        );
    }

    // Print per-device compression ratio percentiles
    println!("\n=== Per-Device Compression Ratio ===");
    println!("{:<18} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}",
        "Algorithm", "min", "p25", "p50", "p75", "p90", "p99", "max");
    println!("{}", "-".repeat(85));

    for result in &results {
        // Calculate per-device ratios
        let ratios: Vec<f64> = result.raw_sizes.iter()
            .zip(result.sizes.iter())
            .map(|(&raw, &compressed)| raw as f64 / compressed as f64)
            .collect();

        let mut sorted_ratios = ratios.clone();
        sorted_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let ratio_percentile = |p: f64| -> f64 {
            if sorted_ratios.is_empty() { return 0.0; }
            let idx = ((sorted_ratios.len() - 1) as f64 * p) as usize;
            sorted_ratios[idx]
        };

        println!(
            "{:<18} | {:>6.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>6.1} | {:>6.1}",
            result.name,
            ratio_percentile(0.0),  // min
            ratio_percentile(0.25),
            ratio_percentile(0.50),
            ratio_percentile(0.75),
            ratio_percentile(0.90),
            ratio_percentile(0.99),
            ratio_percentile(1.0),  // max
        );
    }

    // Generate HTML report if requested
    if let Some(html_path) = args.html_report {
        if let Err(e) = generate_html_report(
            &results,
            &raw_sizes_per_device,
            raw_size,
            total_readings,
            num_devices,
            &html_path,
        ) {
            eprintln!("Failed to generate HTML report: {}", e);
        }
    }
}
