//! Benchmark nibblerun encoding performance using real sensor data.
//!
//! Measures append throughput, decode throughput, and memory footprint
//! across CSV files containing timestamp,value pairs.

use ahash::AHashMap;
use cap::Cap;
use clap::Parser;
use nibblerun::{appendable, decode, Encoder};
use rand::Rng;
use std::alloc;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::MAX);

#[derive(Parser)]
#[command(name = "nbl-bench")]
#[command(about = "Benchmark nibblerun encoding performance")]
struct Args {
    /// Directory containing CSV files
    dir: PathBuf,

    /// Maximum files to process (0 = all)
    #[arg(short, long, default_value = "0")]
    max_files: usize,
}

/// Per-file statistics
#[derive(Clone)]
struct FileStats {
    reading_count: usize,
    encoded_bytes: usize,
}

/// Statistics calculator for a collection of values
struct Stats {
    values: Vec<usize>,
}

impl Stats {
    fn new(values: Vec<usize>) -> Self {
        let mut values = values;
        values.sort_unstable();
        Self { values }
    }

    fn sum(&self) -> usize {
        self.values.iter().sum()
    }

    fn min(&self) -> usize {
        self.values.first().copied().unwrap_or(0)
    }

    fn max(&self) -> usize {
        self.values.last().copied().unwrap_or(0)
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum() as f64 / self.values.len() as f64
        }
    }

    fn median(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mid = self.values.len() / 2;
        if self.values.len() % 2 == 0 {
            (self.values[mid - 1] + self.values[mid]) as f64 / 2.0
        } else {
            self.values[mid] as f64
        }
    }

    fn percentile(&self, p: f64) -> usize {
        if self.values.is_empty() {
            return 0;
        }
        let idx = ((p / 100.0) * (self.values.len() - 1) as f64).round() as usize;
        self.values[idx.min(self.values.len() - 1)]
    }

    fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self.values.iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / (self.values.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Aggregated benchmark results
#[derive(Default)]
struct BenchResults {
    file_stats: Vec<FileStats>,
    append_duration: Duration,
    decode_duration: Duration,
    total_ops: usize,  // total encode/decode operations across all iterations
}

impl BenchResults {
    fn files_processed(&self) -> usize {
        self.file_stats.len()
    }

    fn total_readings(&self) -> usize {
        self.file_stats.iter().map(|f| f.reading_count).sum()
    }

    fn total_encoded_bytes(&self) -> usize {
        self.file_stats.iter().map(|f| f.encoded_bytes).sum()
    }

    fn raw_bytes(&self) -> usize {
        // 8 bytes for timestamp + 1 byte for i8 value
        self.total_readings() * 9
    }

    fn compression_ratio(&self) -> f64 {
        let encoded = self.total_encoded_bytes();
        if encoded == 0 {
            0.0
        } else {
            self.raw_bytes() as f64 / encoded as f64
        }
    }

    fn append_throughput(&self) -> f64 {
        if self.append_duration.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.total_ops as f64 / self.append_duration.as_secs_f64()
        }
    }

    fn decode_throughput(&self) -> f64 {
        if self.decode_duration.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.total_ops as f64 / self.decode_duration.as_secs_f64()
        }
    }

    fn append_ns_per_reading(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.append_duration.as_nanos() as f64 / self.total_ops as f64
        }
    }

    fn decode_ns_per_reading(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.decode_duration.as_nanos() as f64 / self.total_ops as f64
        }
    }

    fn bytes_per_reading(&self) -> f64 {
        let total = self.total_readings();
        if total == 0 {
            0.0
        } else {
            self.total_encoded_bytes() as f64 / total as f64
        }
    }

    fn encoded_bytes_stats(&self) -> Stats {
        Stats::new(self.file_stats.iter().map(|f| f.encoded_bytes).collect())
    }

    fn reading_count_stats(&self) -> Stats {
        Stats::new(self.file_stats.iter().map(|f| f.reading_count).collect())
    }
}

/// Parse a CSV file to Vec<(timestamp, value)>
fn parse_csv(path: &Path) -> Vec<(u64, i8)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = BufReader::new(file);
    let mut readings = Vec::new();

    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        // Skip header and empty lines
        if trimmed.is_empty() || trimmed.starts_with("ts") || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let ts: u64 = match parts[0].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let value: i32 = match parts[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Skip sentinel values (-1000 = gap marker)
        if value == -1000 {
            continue;
        }

        // Clamp value to i8 range
        let value_i8 = value.clamp(-128, 127) as i8;
        readings.push((ts, value_i8));
    }

    readings
}

/// Benchmark decoding from encoded bytes
fn bench_decode(bytes: &[u8]) -> Duration {
    let start = Instant::now();
    let decoded = decode::<i8, 300>(bytes);
    black_box(decoded);
    start.elapsed()
}

/// Format a number with thousands separators
fn format_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format bytes in human-readable form
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format bytes as float
fn format_bytes_f64(bytes: f64) -> String {
    if bytes >= 1024.0 * 1024.0 {
        format!("{:.2} MB", bytes / (1024.0 * 1024.0))
    } else if bytes >= 1024.0 {
        format!("{:.2} KB", bytes / 1024.0)
    } else {
        format!("{:.1} B", bytes)
    }
}

/// Format throughput in human-readable form
fn format_throughput(readings_per_sec: f64) -> String {
    if readings_per_sec >= 1_000_000.0 {
        format!("{:.2}M readings/sec", readings_per_sec / 1_000_000.0)
    } else if readings_per_sec >= 1_000.0 {
        format!("{:.2}K readings/sec", readings_per_sec / 1_000.0)
    } else {
        format!("{:.0} readings/sec", readings_per_sec)
    }
}

fn main() {
    let args = Args::parse();

    // Expand ~ in path
    let dir = if args.dir.starts_with("~") {
        let home = std::env::var("HOME").expect("HOME not set");
        PathBuf::from(home).join(args.dir.strip_prefix("~").unwrap())
    } else {
        args.dir.clone()
    };

    println!("NibbleRun Benchmark");
    println!("===================");
    println!();
    println!("Directory: {}", dir.display());
    println!();

    // Collect all CSV files
    let entries: Vec<_> = fs::read_dir(&dir)
        .expect("Failed to read directory")
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "csv"))
        .collect();

    let total_files = if args.max_files > 0 {
        args.max_files.min(entries.len())
    } else {
        entries.len()
    };

    println!(
        "Found {} CSV files, processing {}...",
        entries.len(),
        total_files
    );

    // Parse all CSV files into pending readings per device
    println!("Parsing CSV files...");
    println!("  Reading {} files from disk", total_files);
    let parse_start = Instant::now();
    let mut pending: AHashMap<u32, VecDeque<(u64, i8)>> = AHashMap::new();
    let mut file_reading_counts: AHashMap<u32, usize> = AHashMap::new();
    let mut empty_files = 0;

    for (idx, entry) in entries.into_iter().take(total_files).enumerate() {
        let path = entry.path();
        let readings = parse_csv(&path);
        if readings.is_empty() {
            empty_files += 1;
            continue;
        }

        let baby_uid = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| u32::from_str_radix(s.trim(), 16).ok())
            .unwrap_or(idx as u32);

        file_reading_counts.insert(baby_uid, readings.len());
        pending.insert(baby_uid, VecDeque::from(readings));
    }

    let parse_elapsed = parse_start.elapsed();
    let num_devices = pending.len();
    let total_readings: usize = pending.values().map(|v| v.len()).sum();
    println!(
        "  Parsed {} files in {:.2}s ({} empty/skipped)",
        total_files,
        parse_elapsed.as_secs_f64(),
        empty_files
    );
    println!("  Devices with data: {}", num_devices);
    println!("  Total readings: {}", format_num(total_readings));
    println!();

    // Benchmark phase: interleaved encoding (simulates real-world usage)
    println!("Running benchmark (interleaved encoding)...");
    println!("  Simulating real-world usage: randomly pick device, append next reading");
    let mut results = BenchResults::default();
    let mut rng = rand::rng();

    // Storage: raw encoded byte buffers per device
    let mut storage: AHashMap<u32, Vec<u8>> = AHashMap::new();

    // Build a list of device UIDs for random selection (avoids repeated allocation)
    let mut device_uids: Vec<u32> = pending.keys().copied().collect();

    // Interleaved encoding
    while !device_uids.is_empty() {
        // Pick random device index (not timed - test harness overhead)
        let idx = rng.random_range(0..device_uids.len());
        let uid = device_uids[idx];

        // Pop next reading (not timed - test harness overhead)
        let readings = pending.get_mut(&uid).unwrap();
        let (ts, val) = readings.pop_front().unwrap();

        // Timed: hashmap lookup + append/insert (this is the real workload)
        let start = Instant::now();
        if let Some(buf) = storage.get_mut(&uid) {
            let _ = appendable::append::<i8, 300>(buf, ts, val);
        } else {
            let buf = appendable::create::<i8, 300>(ts, val);
            storage.insert(uid, buf);
        }
        results.append_duration += start.elapsed();
        results.total_ops += 1;

        // Remove device from list if done (swap-remove for O(1), not timed)
        if readings.is_empty() {
            pending.remove(&uid);
            device_uids.swap_remove(idx);
        }
    }
    println!("  Encoded {} readings across {} devices in {:.2}ms",
        format_num(results.total_ops),
        storage.len(),
        results.append_duration.as_secs_f64() * 1000.0
    );

    // Measure storage memory after pending is fully consumed
    print!("  Measuring memory usage...");
    let alloc_before = ALLOCATOR.allocated();
    // Clone storage to measure its size (clone allocates same amount)
    let storage_clone: AHashMap<u32, Vec<u8>> = storage.iter()
        .map(|(&k, v)| (k, v.clone()))
        .collect();
    let alloc_after = ALLOCATOR.allocated();
    let total_memory = alloc_after.saturating_sub(alloc_before);
    drop(storage_clone);

    storage.shrink_to_fit();
    println!(" {}", format_bytes(total_memory));

    // Collect file stats from final storage
    print!("  Collecting stats from {} buffers...", storage.len());
    for (&uid, buf) in &storage {
        results.file_stats.push(FileStats {
            reading_count: file_reading_counts.get(&uid).copied().unwrap_or(0),
            encoded_bytes: buf.len(),
        });
    }
    println!(" done");

    // Decode phase: decode all buffers
    println!();
    println!("Running decode benchmark...");
    println!("  Decoding {} buffers sequentially", storage.len());
    let decode_start = Instant::now();
    for (_uid, bytes) in &storage {
        let decode_time = bench_decode(bytes);
        results.decode_duration += decode_time;
    }
    let decode_elapsed = decode_start.elapsed();
    println!("  Decoded {} readings in {:.2}ms",
        format_num(results.total_readings()),
        decode_elapsed.as_secs_f64() * 1000.0
    );

    // Print results
    println!();
    println!("Results");
    println!("=======");
    println!();
    println!(
        "Files processed:  {}",
        format_num(results.files_processed())
    );
    println!("Total readings:   {}", format_num(results.total_readings()));
    println!();

    println!("Encode Performance:");
    println!(
        "  Total time:     {:.2} ms",
        results.append_duration.as_secs_f64() * 1000.0
    );
    println!(
        "  Throughput:     {}",
        format_throughput(results.append_throughput())
    );
    println!(
        "  Per reading:    {:.1} ns",
        results.append_ns_per_reading()
    );
    println!();

    println!("Decode Performance:");
    println!(
        "  Total time:     {:.2} ms",
        results.decode_duration.as_secs_f64() * 1000.0
    );
    println!(
        "  Throughput:     {}",
        format_throughput(results.decode_throughput())
    );
    println!(
        "  Per reading:    {:.1} ns",
        results.decode_ns_per_reading()
    );
    println!();

    println!("Memory Footprint (Total):");
    println!(
        "  Raw size:       {} (9 bytes/reading)",
        format_bytes(results.raw_bytes())
    );
    println!(
        "  Encoded size:   {}",
        format_bytes(results.total_encoded_bytes())
    );
    println!("  Compression:    {:.1}x", results.compression_ratio());
    println!(
        "  Per reading:    {:.2} bytes",
        results.bytes_per_reading()
    );
    println!();

    // Per-file statistics
    let bytes_stats = results.encoded_bytes_stats();
    let readings_stats = results.reading_count_stats();

    println!("Per-File Encoded Size:");
    println!("  Min:            {}", format_bytes(bytes_stats.min()));
    println!("  Max:            {}", format_bytes(bytes_stats.max()));
    println!("  Mean:           {}", format_bytes_f64(bytes_stats.mean()));
    println!("  Median:         {}", format_bytes_f64(bytes_stats.median()));
    println!("  Std Dev:        {}", format_bytes_f64(bytes_stats.std_dev()));
    println!("  p25:            {}", format_bytes(bytes_stats.percentile(25.0)));
    println!("  p75:            {}", format_bytes(bytes_stats.percentile(75.0)));
    println!("  p90:            {}", format_bytes(bytes_stats.percentile(90.0)));
    println!("  p95:            {}", format_bytes(bytes_stats.percentile(95.0)));
    println!("  p99:            {}", format_bytes(bytes_stats.percentile(99.0)));
    println!();

    println!("Per-File Reading Count:");
    println!("  Min:            {}", readings_stats.min());
    println!("  Max:            {}", readings_stats.max());
    println!("  Mean:           {:.1}", readings_stats.mean());
    println!("  Median:         {:.1}", readings_stats.median());
    println!();

    let total_encoded: usize = storage.values().map(|v| v.len()).sum();
    println!("Memory Usage:");
    println!("  Total:          {}", format_bytes(total_memory));
    println!("  Encoded data:   {}", format_bytes(total_encoded));
    println!("  Overhead:       {} ({:.1}%)",
        format_bytes(total_memory.saturating_sub(total_encoded)),
        100.0 * (total_memory.saturating_sub(total_encoded)) as f64 / total_memory as f64
    );
    if !storage.is_empty() {
        println!("  Per entry:      {:.1} B", total_memory as f64 / storage.len() as f64);
        println!("  Overhead/entry: {:.1} B", (total_memory.saturating_sub(total_encoded)) as f64 / storage.len() as f64);
    }
    println!();

    // Demonstrate lookup
    if let Some((&sample_uid, sample_data)) = storage.iter().next() {
        println!("Example lookup:");
        println!("  baby_uid:       0x{:08x}", sample_uid);
        println!("  len:            {} bytes", sample_data.len());
        println!("  data[0..4]:     {:?}", &sample_data[..4.min(sample_data.len())]);
    }
    println!();

    // Validation: decode each buffer, re-encode it, and verify the bytes match
    // This tests roundtrip integrity without worrying about CSV gap semantics
    println!("Validation");
    println!("==========");
    println!("Testing roundtrip integrity: decode -> re-encode -> compare bytes");
    println!("  Processing {} buffers...", storage.len());

    let mut validated = 0;
    let mut failed = 0;
    let mut mismatches: Vec<String> = Vec::new();

    for (&baby_uid, encoded_bytes) in &storage {
        let decoded = decode::<i8, 300>(encoded_bytes);

        if decoded.is_empty() {
            mismatches.push(format!("0x{:08x}: decoded to empty", baby_uid));
            failed += 1;
            continue;
        }

        // Re-encode the decoded readings
        let mut encoder = Encoder::<i8>::new();
        for reading in &decoded {
            let _ = encoder.append(reading.ts, reading.value);
        }
        let re_encoded = encoder.to_bytes();

        // Compare the bytes
        if encoded_bytes.len() != re_encoded.len() {
            mismatches.push(format!(
                "0x{:08x}: size mismatch (original={}, re-encoded={})",
                baby_uid, encoded_bytes.len(), re_encoded.len()
            ));
            failed += 1;
            continue;
        }

        if encoded_bytes != re_encoded.as_slice() {
            // Find first differing byte
            let diff_idx = encoded_bytes.iter()
                .zip(re_encoded.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            mismatches.push(format!(
                "0x{:08x}: bytes differ at offset {} (original=0x{:02x}, re-encoded=0x{:02x})",
                baby_uid, diff_idx,
                encoded_bytes.get(diff_idx).unwrap_or(&0),
                re_encoded.get(diff_idx).unwrap_or(&0)
            ));
            failed += 1;
            continue;
        }

        validated += 1;
    }

    println!("  Validated {} buffers", validated);
    println!();
    println!("Results:");
    println!("  Passed:         {}", validated);
    println!("  Failed:         {}", failed);

    if !mismatches.is_empty() {
        println!();
        println!("  Mismatches:");
        for (i, msg) in mismatches.iter().take(10).enumerate() {
            println!("    {}: {}", i + 1, msg);
        }
        if mismatches.len() > 10 {
            println!("    ... and {} more", mismatches.len() - 10);
        }
    }

    if failed == 0 && validated > 0 {
        println!();
        println!("  All {} files validated successfully ✓", validated);
    }

    // Final summary
    println!();
    println!("================================================================================");
    println!("SUMMARY");
    println!("================================================================================");
    println!();
    println!("Data:");
    println!("  Files:          {}", format_num(results.files_processed()));
    println!("  Readings:       {}", format_num(results.total_readings()));
    println!("  Raw size:       {}", format_bytes(results.raw_bytes()));
    println!("  Encoded size:   {}", format_bytes(results.total_encoded_bytes()));
    println!("  Compression:    {:.1}x", results.compression_ratio());
    println!("  Bytes/reading:  {:.2}", results.bytes_per_reading());
    println!();
    println!("Encode:");
    println!("  Throughput:     {}", format_throughput(results.append_throughput()));
    println!("  Per reading:    {:.1} ns", results.append_ns_per_reading());
    println!();
    println!("Decode:");
    println!("  Throughput:     {}", format_throughput(results.decode_throughput()));
    println!("  Per reading:    {:.1} ns", results.decode_ns_per_reading());
    println!();
    println!("Memory:");
    println!("  Total:          {}", format_bytes(total_memory));
    println!("  Data:           {}", format_bytes(total_encoded));
    println!("  Overhead:       {} ({:.1}%)",
        format_bytes(total_memory.saturating_sub(total_encoded)),
        100.0 * (total_memory.saturating_sub(total_encoded)) as f64 / total_memory as f64
    );
    println!();
    println!("Validation:       {} passed, {} failed", validated, failed);

    // Prevent optimizations
    black_box(&storage);
}
