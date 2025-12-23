//! Generate sample nibblerun time series data.

use clap::Parser;
use nibblerun::Encoder;
use rand::Rng;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nbl-gen")]
#[command(about = "Generate sample nibblerun time series data")]
#[command(after_help = "CSV FORMAT:\n  \
    When using --csv, the file should contain lines with: timestamp,value\n  \
    - Lines starting with # are comments\n  \
    - Empty lines are skipped\n  \
    - First line 'timestamp,value' is treated as header and skipped\n  \
    - Timestamps are Unix seconds\n  \
    - Values are i32 sensor readings\n\n\
SUPPORTED INTERVALS:\n  \
    1, 60, 300 (default), 600, 3600")]
struct Args {
    /// Output file path
    output: PathBuf,

    /// Input CSV file with timestamp,value pairs (overrides random generation)
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Number of readings to generate (default: 288 = 24h at 5-min intervals)
    #[arg(short, long, default_value = "288")]
    readings: usize,

    /// Include random gaps (sensor offline periods)
    #[arg(long)]
    gaps: bool,

    /// Include occasional large temperature changes
    #[arg(long)]
    spikes: bool,

    /// Base temperature in Celsius (default: 22)
    #[arg(long, default_value = "22")]
    base_temp: i32,

    /// Interval in seconds (supported: 1, 60, 300, 600, 3600)
    #[arg(long, default_value = "300")]
    interval: u16,
}

/// Read timestamp,value pairs from a CSV file
fn read_csv(path: &PathBuf) -> Result<Vec<(u32, i32)>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open CSV: {e}"))?;
    let reader = BufReader::new(file);
    let mut readings = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| format!("Failed to read line {}: {}", line_num + 1, e))?;
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Skip header line
        if trimmed.to_lowercase().starts_with("timestamp") {
            continue;
        }

        // Parse timestamp,value
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() < 2 {
            return Err(format!(
                "Line {}: expected 'timestamp,value', got '{}'",
                line_num + 1,
                trimmed
            ));
        }

        let ts: u32 = parts[0]
            .trim()
            .parse()
            .map_err(|e| format!("Line {}: invalid timestamp '{}': {}", line_num + 1, parts[0], e))?;

        let value: i32 = parts[1]
            .trim()
            .parse()
            .map_err(|e| format!("Line {}: invalid value '{}': {}", line_num + 1, parts[1], e))?;

        readings.push((ts, value));
    }

    Ok(readings)
}

/// Generate random readings based on args
fn generate_readings(args: &Args) -> Vec<(u32, i32)> {
    let mut rng = rand::rng();
    let mut readings = Vec::new();

    // Base timestamp: start of a day at EPOCH_BASE
    let base_ts: u32 = 1_760_000_000;
    let interval = u32::from(args.interval);

    let mut current_idx: u32 = 0;

    for i in 0..args.readings {
        // Skip some intervals if gaps enabled (roughly 5% chance)
        if args.gaps && i > 0 && rng.random_range(0..100) < 5 {
            let gap_size = rng.random_range(1..=5);
            current_idx += gap_size;
        }

        let ts = base_ts + current_idx * interval;

        // Calculate temperature based on time of day
        // Simulates: cooler at night, warmer during day
        let hour = (current_idx * u32::from(args.interval) / 3600) % 24;
        let hour_f = hour as f64 + (current_idx as f64 * f64::from(args.interval) % 3600.0) / 3600.0;

        // Temperature curve: min at 5am, max at 3pm
        // Using sine wave: base + amplitude * sin((hour - 5) * π / 12 - π/2)
        let temp_variation = 4.0 * ((hour_f - 5.0) * PI / 12.0 - PI / 2.0).sin();
        let base = f64::from(args.base_temp) + temp_variation;

        // Add small random jitter (±1-2°C)
        let jitter: f64 = rng.random_range(-2.0..=2.0);

        let mut temp = (base + jitter).round() as i32;

        // Occasional spike if enabled (roughly 2% chance)
        if args.spikes && rng.random_range(0..100) < 2 {
            let spike: i32 = rng.random_range(-10..=10);
            temp += spike;
        }

        readings.push((ts, temp));
        current_idx += 1;
    }

    readings
}

/// Encode readings and write to file
macro_rules! encode_and_write {
    ($interval:expr, $readings:expr, $output:expr, $from_csv:expr) => {{
        let mut enc = Encoder::<i32, $interval>::new();

        for (i, (ts, value)) in $readings.iter().enumerate() {
            if let Err(e) = enc.append(*ts, *value) {
                eprintln!("Warning: Failed to append reading {i} (ts={ts}, value={value}): {e}");
            }
        }

        let bytes = enc.to_bytes();

        let mut file = File::create(&$output).expect("Failed to create output file");
        file.write_all(&bytes).expect("Failed to write data");

        if $from_csv {
            println!("Encoded {} readings from CSV", enc.count());
        } else {
            println!("Generated {} readings", enc.count());
        }
        println!("Output: {} ({} bytes)", $output.display(), bytes.len());
        println!(
            "Compression: {:.1}x",
            (enc.count() * 12) as f64 / bytes.len() as f64
        );
    }};
}

fn main() {
    let args = Args::parse();

    // Get readings from CSV or generate randomly
    let (readings, from_csv) = if let Some(csv_path) = &args.csv {
        match read_csv(csv_path) {
            Ok(r) => {
                println!("Reading {} entries from CSV...", r.len());
                (r, true)
            }
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        (generate_readings(&args), false)
    };

    // Match on supported intervals (const generics require compile-time values)
    match args.interval {
        1 => encode_and_write!(1, readings, args.output, from_csv),
        60 => encode_and_write!(60, readings, args.output, from_csv),
        300 => encode_and_write!(300, readings, args.output, from_csv),
        600 => encode_and_write!(600, readings, args.output, from_csv),
        3600 => encode_and_write!(3600, readings, args.output, from_csv),
        other => {
            eprintln!("Error: Unsupported interval {other}. Supported values: 1, 60, 300, 600, 3600");
            std::process::exit(1);
        }
    }
}
