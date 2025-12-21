//! Benchmark memory usage of different storage methods for time series data.

use clap::{Parser, ValueEnum};
use rand::Rng;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tikv_jemalloc_ctl::{epoch, stats};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Debug, Clone, ValueEnum)]
enum BenchmarkType {
    Naive,
    NibblerunHashmap,
    NibblerunBtreemap,
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
    println!("  Compressed data: {} KB total, {} bytes avg per device", total_data / 1024, avg_data);
    println!("  Vec capacity: {} KB total, {} bytes avg per device", total_capacity / 1024, avg_capacity);

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
    println!("  Compressed data: {} KB total, {} bytes avg per device", total_data / 1024, avg_data);
    println!("  Vec capacity: {} KB total, {} bytes avg per device", total_capacity / 1024, avg_capacity);

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
    println!("Expected naive size: {} KB ({} readings * 8 bytes)", expected_bytes / 1024, interleaved.len());
    println!();

    let before = get_allocated();

    let (name, compressed_data): (&str, Option<usize>) = match args.benchmark {
        BenchmarkType::Naive => {
            println!("Running naive benchmark...");
            let storage = run_naive(&interleaved);
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("naive", expected_bytes, allocated, None);
            black_box(storage);
            return;
        }
        BenchmarkType::NibblerunHashmap => {
            println!("Running nibblerun-hashmap benchmark...");
            let (storage, data_size) = run_nibblerun_hashmap(&interleaved);
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("nibblerun-hashmap", expected_bytes, allocated, Some(data_size));
            black_box(storage);
            return;
        }
        BenchmarkType::NibblerunBtreemap => {
            println!("Running nibblerun-btreemap benchmark...");
            let (storage, data_size) = run_nibblerun_btreemap(&interleaved);
            let after = get_allocated();
            let allocated = after.saturating_sub(before);
            print_results("nibblerun-btreemap", expected_bytes, allocated, Some(data_size));
            black_box(storage);
            return;
        }
    };
}

fn print_results(name: &str, expected_bytes: usize, allocated: usize, compressed_data: Option<usize>) {
    let ratio = expected_bytes as f64 / allocated as f64;
    println!();

    if let Some(data_size) = compressed_data {
        let overhead = allocated.saturating_sub(data_size);
        let data_ratio = expected_bytes as f64 / data_size as f64;
        println!("Benchmark: {} | Expected: {} KB | Allocated: {} KB | Ratio: {:.1}x",
                 name, expected_bytes / 1024, allocated / 1024, ratio);
        println!("Compressed data: {} KB | Overhead: {} KB | Data ratio: {:.1}x",
                 data_size / 1024, overhead / 1024, data_ratio);
    } else {
        println!("Benchmark: {} | Expected: {} KB | Allocated: {} KB | Ratio: {:.1}x",
                 name, expected_bytes / 1024, allocated / 1024, ratio);
    }
}
