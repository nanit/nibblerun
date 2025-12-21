//! Benchmark memory usage of different storage methods for time series data.

use clap::{Parser, ValueEnum};
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tikv_jemalloc_ctl::{epoch, stats};

/// Storage result from benchmarks
enum Storage {
    Naive(HashMap<u32, Vec<(u32, i8)>>),
    NibblerunHashmap(HashMap<u32, Vec<u8>>),
}

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Debug, Clone, ValueEnum)]
enum BenchmarkType {
    Naive,
    NibblerunHashmap,
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
/// Returns the storage to keep it alive for memory measurement
fn run_nibblerun_hashmap(readings: &[(u32, u32, i8)]) -> HashMap<u32, Vec<u8>> {
    let mut storage: HashMap<u32, Vec<u8>> = HashMap::new();
    for &(device_id, ts, val) in readings {
        if let Some(buf) = storage.get_mut(&device_id) {
            let _ = nibblerun::appendable::append::<i8, 300>(buf, ts as u64, val);
        } else {
            let buf = nibblerun::appendable::create::<i8, 300>(ts as u64, val);
            storage.insert(device_id, buf);
        }
    }
    println!("  Stored {} devices", storage.len());
    storage
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

    let (name, storage) = match args.benchmark {
        BenchmarkType::Naive => {
            println!("Running naive benchmark...");
            ("naive", Storage::Naive(run_naive(&interleaved)))
        }
        BenchmarkType::NibblerunHashmap => {
            println!("Running nibblerun-hashmap benchmark...");
            ("nibblerun-hashmap", Storage::NibblerunHashmap(run_nibblerun_hashmap(&interleaved)))
        }
    };

    let after = get_allocated();
    let allocated = after.saturating_sub(before);
    let ratio = expected_bytes as f64 / allocated as f64;
    println!();
    println!("Benchmark: {} | Expected: {} KB | Allocated: {} KB | Ratio: {:.1}x",
             name, expected_bytes / 1024, allocated / 1024, ratio);
    black_box(storage);
}
