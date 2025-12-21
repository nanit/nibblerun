//! Benchmark memory usage of different storage methods for time series data.

use ahash::AHashMap;
use clap::{Parser, ValueEnum};
use rand::Rng;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tikv_jemalloc_ctl::{epoch, stats};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Debug, Clone, ValueEnum)]
enum BenchmarkType {
    Naive,
    NibblerunHashmap,
    NibblerunAhash,
    NibblerunFxhash,
    NibblerunBtreemap,
    NibblerunFrozen,
}

#[derive(Parser)]
#[command(name = "nbl-bench")]
#[command(about = "Benchmark memory usage of different storage methods")]
struct Args {
    /// Directory containing CSV files
    dir: PathBuf,

    /// Maximum files to process (0 = all)
    #[arg(short, long, default_value = "0")]
    max_files: usize,

    /// Benchmark to run
    #[arg(short, long, value_enum)]
    benchmark: BenchmarkType,
}

fn get_allocated() -> usize {
    epoch::advance().unwrap();
    stats::allocated::read().unwrap()
}

/// Load readings from CSV files into a HashMap of device_id -> VecDeque<(timestamp, value)>
/// Filters out -1000 sentinel values.
/// Keeps loading files until `max_files` files with valid data are loaded (0 = all).
fn load_readings(dir: &Path, max_files: usize) -> HashMap<u32, VecDeque<(u32, i8)>> {
    let entries: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "csv"))
        .collect();

    let mut result: HashMap<u32, VecDeque<(u32, i8)>> = HashMap::new();

    for (idx, entry) in entries.into_iter().enumerate() {
        // Stop if we've reached the requested sample size
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
        let mut readings = VecDeque::new();

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
            readings.push_back((ts, value_i8));
        }

        if !readings.is_empty() {
            result.insert(device_id, readings);
        }
    }

    result
}

/// Interleave readings from multiple devices to simulate real-world arrival order.
/// Randomly picks a device, pops one reading, appends to result.
/// Returns Vec<(device_id, timestamp, value)>
fn interleave_readings(mut pending: HashMap<u32, VecDeque<(u32, i8)>>) -> Vec<(u32, u32, i8)> {
    let mut result = Vec::new();
    let mut rng = rand::rng();

    // Pre-calculate total size for allocation
    let total: usize = pending.values().map(|v| v.len()).sum();
    result.reserve(total);

    // Build list of device IDs for random selection
    let mut device_ids: Vec<u32> = pending.keys().copied().collect();

    while !device_ids.is_empty() {
        // Pick random device
        let idx = rng.random_range(0..device_ids.len());
        let device_id = device_ids[idx];

        // Pop next reading
        let readings = pending.get_mut(&device_id).unwrap();
        let (ts, val) = readings.pop_front().unwrap();
        result.push((device_id, ts, val));

        // Remove device if exhausted
        if readings.is_empty() {
            pending.remove(&device_id);
            device_ids.swap_remove(idx);
        }
    }

    result
}

/// Naive storage: HashMap<u32, Vec<(u32, i8)>>
/// Returns the storage to keep it alive for memory measurement
fn run_naive(readings: &[(u32, u32, i8)]) -> HashMap<u32, Vec<(u32, i8)>> {
    let mut storage: HashMap<u32, Vec<(u32, i8)>> = HashMap::new();
    for &(device_id, ts, val) in readings {
        storage.entry(device_id).or_default().push((ts, val));
    }
    println!("  Stored {} devices", storage.len());
    storage
}

/// NibbleRun storage: HashMap<u32, Vec<u8>>
/// Returns the storage and compressed data size
fn run_nibblerun_hashmap(readings: &[(u32, u32, i8)]) -> (HashMap<u32, Vec<u8>>, usize) {
    let mut storage: HashMap<u32, Vec<u8>> = HashMap::new();
    for &(device_id, ts, val) in readings {
        if let Some(buf) = storage.get_mut(&device_id) {
            let _ = nibblerun::appendable::append::<i8, 300>(buf, ts as u64, val);
        } else {
            let buf = nibblerun::appendable::create::<i8, 300>(ts as u64, val);
            storage.insert(device_id, buf);
        }
    }

    // Calculate actual compressed data size and capacity
    let total_data: usize = storage.values().map(|v| v.len()).sum();
    let total_capacity: usize = storage.values().map(|v| v.capacity()).sum();
    let avg_data = total_data / storage.len();
    let avg_capacity = total_capacity / storage.len();

    println!("  Stored {} devices (HashMap capacity: {})", storage.len(), storage.capacity());
    println!("  Compressed data: {:.1} MB total, {} bytes avg per device", total_data as f64 / (1024.0 * 1024.0), avg_data);
    println!("  Vec capacity: {:.1} MB total, {} bytes avg per device", total_capacity as f64 / (1024.0 * 1024.0), avg_capacity);

    (storage, total_data)
}

/// NibbleRun storage: AHashMap<u32, Vec<u8>>
/// Returns the storage and compressed data size
fn run_nibblerun_ahash(readings: &[(u32, u32, i8)]) -> (AHashMap<u32, Vec<u8>>, usize) {
    let mut storage: AHashMap<u32, Vec<u8>> = AHashMap::new();
    for &(device_id, ts, val) in readings {
        if let Some(buf) = storage.get_mut(&device_id) {
            let _ = nibblerun::appendable::append::<i8, 300>(buf, ts as u64, val);
        } else {
            let buf = nibblerun::appendable::create::<i8, 300>(ts as u64, val);
            storage.insert(device_id, buf);
        }
    }

    // Calculate actual compressed data size and capacity
    let total_data: usize = storage.values().map(|v| v.len()).sum();
    let total_capacity: usize = storage.values().map(|v| v.capacity()).sum();
    let avg_data = total_data / storage.len();
    let avg_capacity = total_capacity / storage.len();

    println!("  Stored {} devices (AHashMap capacity: {})", storage.len(), storage.capacity());
    println!("  Compressed data: {:.1} MB total, {} bytes avg per device", total_data as f64 / (1024.0 * 1024.0), avg_data);
    println!("  Vec capacity: {:.1} MB total, {} bytes avg per device", total_capacity as f64 / (1024.0 * 1024.0), avg_capacity);

    (storage, total_data)
}

/// NibbleRun storage: FxHashMap<u32, Vec<u8>>
/// Returns the storage and compressed data size
fn run_nibblerun_fxhash(readings: &[(u32, u32, i8)]) -> (FxHashMap<u32, Vec<u8>>, usize) {
    let mut storage: FxHashMap<u32, Vec<u8>> = FxHashMap::default();
    for &(device_id, ts, val) in readings {
        if let Some(buf) = storage.get_mut(&device_id) {
            let _ = nibblerun::appendable::append::<i8, 300>(buf, ts as u64, val);
        } else {
            let buf = nibblerun::appendable::create::<i8, 300>(ts as u64, val);
            storage.insert(device_id, buf);
        }
    }

    // Calculate actual compressed data size and capacity
    let total_data: usize = storage.values().map(|v| v.len()).sum();
    let total_capacity: usize = storage.values().map(|v| v.capacity()).sum();
    let avg_data = total_data / storage.len();
    let avg_capacity = total_capacity / storage.len();

    println!("  Stored {} devices (FxHashMap capacity: {})", storage.len(), storage.capacity());
    println!("  Compressed data: {:.1} MB total, {} bytes avg per device", total_data as f64 / (1024.0 * 1024.0), avg_data);
    println!("  Vec capacity: {:.1} MB total, {} bytes avg per device", total_capacity as f64 / (1024.0 * 1024.0), avg_capacity);

    (storage, total_data)
}

/// NibbleRun storage: BTreeMap<u32, Vec<u8>>
/// Returns the storage and compressed data size
fn run_nibblerun_btreemap(readings: &[(u32, u32, i8)]) -> (BTreeMap<u32, Vec<u8>>, usize) {
    let mut storage: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    for &(device_id, ts, val) in readings {
        if let Some(buf) = storage.get_mut(&device_id) {
            let _ = nibblerun::appendable::append::<i8, 300>(buf, ts as u64, val);
        } else {
            let buf = nibblerun::appendable::create::<i8, 300>(ts as u64, val);
            storage.insert(device_id, buf);
        }
    }

    // Calculate actual compressed data size and capacity
    let total_data: usize = storage.values().map(|v| v.len()).sum();
    let total_capacity: usize = storage.values().map(|v| v.capacity()).sum();
    let avg_data = total_data / storage.len();
    let avg_capacity = total_capacity / storage.len();

    println!("  Stored {} devices", storage.len());
    println!("  Compressed data: {:.1} MB total, {} bytes avg per device", total_data as f64 / (1024.0 * 1024.0), avg_data);
    println!("  Vec capacity: {:.1} MB total, {} bytes avg per device", total_capacity as f64 / (1024.0 * 1024.0), avg_capacity);

    (storage, total_data)
}

/// NibbleRun frozen storage: AHashMap<u32, Vec<u8>>
/// Encodes then freezes for compact read-only storage.
/// Returns the storage and frozen data size
fn run_nibblerun_frozen(readings: &[(u32, u32, i8)]) -> (AHashMap<u32, Vec<u8>>, usize) {
    // First encode to appendable format
    let mut temp_storage: AHashMap<u32, Vec<u8>> = AHashMap::new();
    for &(device_id, ts, val) in readings {
        if let Some(buf) = temp_storage.get_mut(&device_id) {
            let _ = nibblerun::appendable::append::<i8, 300>(buf, ts as u64, val);
        } else {
            let buf = nibblerun::appendable::create::<i8, 300>(ts as u64, val);
            temp_storage.insert(device_id, buf);
        }
    }

    // Then freeze all buffers
    let mut storage: AHashMap<u32, Vec<u8>> = AHashMap::new();
    for (device_id, buf) in temp_storage {
        let frozen = nibblerun::appendable::freeze::<i8, 300>(&buf);
        storage.insert(device_id, frozen);
    }

    // Calculate actual frozen data size
    let total_data: usize = storage.values().map(|v| v.len()).sum();
    let total_capacity: usize = storage.values().map(|v| v.capacity()).sum();
    let avg_data = total_data / storage.len();
    let avg_capacity = total_capacity / storage.len();

    println!("  Stored {} devices (AHashMap capacity: {})", storage.len(), storage.capacity());
    println!("  Frozen data: {:.1} MB total, {} bytes avg per device", total_data as f64 / (1024.0 * 1024.0), avg_data);
    println!("  Vec capacity: {:.1} MB total, {} bytes avg per device", total_capacity as f64 / (1024.0 * 1024.0), avg_capacity);

    (storage, total_data)
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

    println!("Loading readings from {}...", dir.display());
    let pending = load_readings(&dir, args.max_files);
    let num_devices = pending.len();
    let total_readings: usize = pending.values().map(|v| v.len()).sum();
    println!("Loaded {} readings from {} devices", total_readings, num_devices);

    println!("Interleaving readings...");
    let interleaved = interleave_readings(pending);
    println!("Interleaved {} readings", interleaved.len());
    println!();

    // Expected size: each reading is (u32 ts + i8 val + padding) = 8 bytes per reading
    // But stored as (u32, i8) tuple which is 8 bytes with padding
    let expected_bytes = interleaved.len() * 8;
    println!("Expected naive size: {:.1} MB ({} readings * 8 bytes)", expected_bytes as f64 / (1024.0 * 1024.0), interleaved.len());
    println!();

    let before = get_allocated();
    let start = Instant::now();

    let (name, compressed_data): (&str, Option<usize>) = match args.benchmark {
        BenchmarkType::Naive => {
            println!("Running naive benchmark...");
            let storage = run_naive(&interleaved);
            let encode_duration = start.elapsed();
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("naive", expected_bytes, allocated, None, interleaved.len(), encode_duration);
            black_box(storage);
            return;
        }
        BenchmarkType::NibblerunHashmap => {
            println!("Running nibblerun-hashmap benchmark...");
            let (storage, data_size) = run_nibblerun_hashmap(&interleaved);
            let encode_duration = start.elapsed();
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("nibblerun-hashmap", expected_bytes, allocated, Some(data_size), interleaved.len(), encode_duration);
            black_box(storage);
            return;
        }
        BenchmarkType::NibblerunAhash => {
            println!("Running nibblerun-ahash benchmark...");
            let (storage, data_size) = run_nibblerun_ahash(&interleaved);
            let encode_duration = start.elapsed();
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("nibblerun-ahash", expected_bytes, allocated, Some(data_size), interleaved.len(), encode_duration);
            black_box(storage);
            return;
        }
        BenchmarkType::NibblerunFxhash => {
            println!("Running nibblerun-fxhash benchmark...");
            let (storage, data_size) = run_nibblerun_fxhash(&interleaved);
            let encode_duration = start.elapsed();
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("nibblerun-fxhash", expected_bytes, allocated, Some(data_size), interleaved.len(), encode_duration);
            black_box(storage);
            return;
        }
        BenchmarkType::NibblerunBtreemap => {
            println!("Running nibblerun-btreemap benchmark...");
            let (storage, data_size) = run_nibblerun_btreemap(&interleaved);
            let encode_duration = start.elapsed();
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("nibblerun-btreemap", expected_bytes, allocated, Some(data_size), interleaved.len(), encode_duration);
            black_box(storage);
            return;
        }
        BenchmarkType::NibblerunFrozen => {
            println!("Running nibblerun-frozen benchmark...");
            let (storage, data_size) = run_nibblerun_frozen(&interleaved);
            let encode_duration = start.elapsed();
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("nibblerun-frozen", expected_bytes, allocated, Some(data_size), interleaved.len(), encode_duration);
            black_box(storage);
            return;
        }
    };
}

fn print_results(name: &str, expected_bytes: usize, allocated: usize, compressed_data: Option<usize>, total_readings: usize, encode_duration: Duration) {
    let ratio = expected_bytes as f64 / allocated as f64;
    let encode_rate_m = total_readings as f64 / encode_duration.as_secs_f64() / 1_000_000.0;
    let expected_mb = expected_bytes as f64 / (1024.0 * 1024.0);
    let allocated_mb = allocated as f64 / (1024.0 * 1024.0);
    println!();

    if let Some(data_size) = compressed_data {
        let overhead = allocated.saturating_sub(data_size);
        let data_ratio = expected_bytes as f64 / data_size as f64;
        let data_mb = data_size as f64 / (1024.0 * 1024.0);
        let overhead_mb = overhead as f64 / (1024.0 * 1024.0);
        println!("Benchmark: {} | Expected: {:.1} MB | Allocated: {:.1} MB | Ratio: {:.1}x",
                 name, expected_mb, allocated_mb, ratio);
        println!("Compressed data: {:.1} MB | Overhead: {:.1} MB | Data ratio: {:.1}x",
                 data_mb, overhead_mb, data_ratio);
    } else {
        println!("Benchmark: {} | Expected: {:.1} MB | Allocated: {:.1} MB | Ratio: {:.1}x",
                 name, expected_mb, allocated_mb, ratio);
    }
    println!("Encode rate: {:.1}M readings/sec ({:.2}s for {} readings)",
             encode_rate_m, encode_duration.as_secs_f64(), total_readings);
}
