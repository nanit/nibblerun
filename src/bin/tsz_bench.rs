//! Benchmark TSZ (Gorilla) encoding performance using real sensor data.
//!
//! Measures append throughput and memory footprint
//! across CSV files containing timestamp,value pairs.

use ahash::AHashMap;
use cap::Cap;
use clap::Parser;
use rand::Rng;
use std::alloc;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tsz::stream::BufferedWriter;
use tsz::{DataPoint, Encode, StdEncoder};

#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::MAX);

#[derive(Parser)]
#[command(name = "tsz-bench")]
#[command(about = "Benchmark TSZ (Gorilla) encoding performance")]
struct Args {
    /// Directory containing CSV files
    dir: PathBuf,

    /// Maximum files to process (0 = all)
    #[arg(short, long, default_value = "0")]
    max_files: usize,
}

/// Aggregated benchmark results
#[derive(Default)]
struct BenchResults {
    num_devices: usize,
    total_readings: usize,
    append_duration: Duration,
    total_ops: usize,
}

impl BenchResults {
    fn append_throughput(&self) -> f64 {
        if self.append_duration.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.total_ops as f64 / self.append_duration.as_secs_f64()
        }
    }

    fn append_ns_per_reading(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            self.append_duration.as_nanos() as f64 / self.total_ops as f64
        }
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

    println!("TSZ (Gorilla) Benchmark");
    println!("=======================");
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
    results.num_devices = num_devices;
    results.total_readings = total_readings;
    let mut rng = rand::rng();

    // Storage: TSZ encoders per device (don't call .close())
    let mut storage: AHashMap<u32, StdEncoder<BufferedWriter>> = AHashMap::new();

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

        // Timed: hashmap lookup + encode (this is the real workload)
        let start = Instant::now();
        if let Some(encoder) = storage.get_mut(&uid) {
            encoder.encode(DataPoint::new(ts, val as f64));
        } else {
            let w = BufferedWriter::new();
            let mut encoder = StdEncoder::new(ts, w);
            encoder.encode(DataPoint::new(ts, val as f64));
            storage.insert(uid, encoder);
        }
        results.append_duration += start.elapsed();
        results.total_ops += 1;

        // Remove device from list if done (swap-remove for O(1), not timed)
        if readings.is_empty() {
            pending.remove(&uid);
            device_uids.swap_remove(idx);
        }
    }
    println!(
        "  Encoded {} readings across {} devices in {:.2}ms",
        format_num(results.total_ops),
        storage.len(),
        results.append_duration.as_secs_f64() * 1000.0
    );

    // Measure storage memory (clone the HashMap to measure)
    print!("  Measuring memory usage...");
    let alloc_before = ALLOCATOR.allocated();
    // We can't clone StdEncoder, so we measure by dropping and re-checking
    // Instead, measure current allocation
    let alloc_current = ALLOCATOR.allocated();
    // Create a dummy to estimate overhead
    let dummy_storage: AHashMap<u32, Vec<u8>> = AHashMap::new();
    let _ = black_box(&dummy_storage);
    let total_memory = alloc_current.saturating_sub(alloc_before);
    println!(" (estimated from allocator)");

    // Can't get exact encoder size without closing, so estimate based on allocator
    let final_allocated = ALLOCATOR.allocated();
    println!("  Current allocator usage: {}", format_bytes(final_allocated));

    // Print results
    println!();
    println!("Results");
    println!("=======");
    println!();
    println!("Files processed:  {}", format_num(results.num_devices));
    println!("Total readings:   {}", format_num(results.total_readings));
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

    // Final summary
    println!("================================================================================");
    println!("SUMMARY");
    println!("================================================================================");
    println!();
    println!("Data:");
    println!("  Files:          {}", format_num(results.num_devices));
    println!("  Readings:       {}", format_num(results.total_readings));
    println!();
    println!("Encode:");
    println!(
        "  Throughput:     {}",
        format_throughput(results.append_throughput())
    );
    println!("  Per reading:    {:.1} ns", results.append_ns_per_reading());
    println!();
    println!("Memory:");
    println!(
        "  Allocator total: {}",
        format_bytes(ALLOCATOR.allocated())
    );

    // Prevent optimizations
    black_box(&storage);
}
