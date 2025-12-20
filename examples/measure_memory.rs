use nibblerun::Encoder;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::mem;

fn main() {
    // Print struct size (without heap allocations)
    println!("=== Encoder<i8, 300> Memory Analysis ===\n");
    println!("Stack size (struct itself): {} bytes", mem::size_of::<Encoder<i8, 300>>());
    println!("Alignment: {} bytes", mem::align_of::<Encoder<i8, 300>>());

    // Create encoder and load data
    let mut encoder: Encoder<i8, 300> = Encoder::new();

    let file = File::open("/Users/quatrix/workspace/random/compacting_time_series/data/out_csv/6de38c22.csv")
        .expect("Failed to open CSV");
    let reader = BufReader::new(file);

    let mut count = 0;
    for line in reader.lines().skip(1) {
        let line = line.expect("Failed to read line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 2 {
            let ts: u64 = parts[0].parse().expect("Invalid timestamp");
            let temp: i8 = parts[1].parse().expect("Invalid temperature");
            encoder.append(ts, temp).expect("Failed to append");
            count += 1;
        }
    }

    println!("\nLoaded {} readings (1 day at 5-min intervals)", count);

    // Measure the Vec<u8> data buffer
    // We need to access internal state - let's use the encoded output size
    let encoded = encoder.to_bytes();

    println!("\n=== Memory Breakdown ===");
    println!("Stack size: {} bytes", mem::size_of::<Encoder<i8, 300>>());
    println!("Encoded output size: {} bytes", encoded.len());
    println!("Reading count: {}", encoder.count());

    // Calculate approximate heap usage
    // Vec<u8> has: pointer (8) + capacity (8) + len (8) = 24 bytes overhead
    // Plus the actual data
    println!("\n=== Estimated Total Memory ===");
    println!("Stack: {} bytes", mem::size_of::<Encoder<i8, 300>>());
    println!("Heap (data buffer, estimated): ~{} bytes", encoded.len().saturating_sub(7)); // subtract header

    // Let's also print field-by-field breakdown based on the struct definition
    println!("\n=== Field-by-Field Stack Breakdown ===");
    println!("pending_avg: {} bytes (u64)", mem::size_of::<u64>());
    println!("bit_accum: {} bytes (u32)", mem::size_of::<u32>());
    println!("data: {} bytes (Vec<u8> - pointer+len+cap)", mem::size_of::<Vec<u8>>());
    println!("base_ts_offset: {} bytes (u32)", mem::size_of::<u32>());
    println!("first_value: {} bytes (i8)", mem::size_of::<i8>());
    println!("prev_value: {} bytes (i8)", mem::size_of::<i8>());
    println!("prev_logical_idx: {} bytes (u16)", mem::size_of::<u16>());
    println!("count: {} bytes (u16)", mem::size_of::<u16>());
    println!("bit_count: {} bytes (u8)", mem::size_of::<u8>());
    println!("_marker: {} bytes (PhantomData)", mem::size_of::<std::marker::PhantomData<i8>>());
}
