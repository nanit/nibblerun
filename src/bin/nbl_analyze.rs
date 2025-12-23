//! Analyze delta frequencies across real sensor data.
//!
//! This tool processes CSV files with timestamp,temperature pairs and computes
//! statistics to help optimize the nibblerun encoding scheme.

use clap::Parser;
use nibblerun::Encoder;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nbl-analyze")]
#[command(about = "Analyze sensor data for encoding optimization")]
struct Args {
    /// Directory containing CSV files
    dir: PathBuf,

    /// Maximum files to process (0 = all)
    #[arg(short, long, default_value = "0")]
    max_files: usize,

    /// Show progress every N files
    #[arg(short, long, default_value = "10000")]
    progress: usize,

    /// Number of threads (0 = auto)
    #[arg(short, long, default_value = "0")]
    threads: usize,

    /// Output HTML report with interactive charts
    #[arg(long)]
    html: Option<PathBuf>,

    /// Label for the data type (e.g., "temperature", "humidity")
    #[arg(short, long, default_value = "temperature")]
    label: String,
}

#[derive(Default)]
struct Stats {
    files_processed: u64,
    total_readings: u64,
    // Delta counts (raw values)
    delta_counts: HashMap<i32, u64>,
    // Zero-run length histogram
    zero_run_lengths: HashMap<u32, u64>,
    // Gap counts (number of intervals skipped)
    gap_counts: HashMap<u32, u64>,
    // Value distribution (temperature, humidity, etc.)
    value_counts: HashMap<i32, u64>,
    max_zero_run: u32,
    max_delta: i32,
    min_delta: i32,
    // Value range
    max_value: i32,
    min_value: i32,
    // Actual compression stats
    actual_compressed_bytes: u64,
    raw_input_bytes: u64,
    // Data type label
    label: String,
    // Per-file compression ratios for distribution analysis
    compression_ratios: Vec<f32>,
}

impl Stats {
    fn merge(&mut self, other: &Self) {
        self.files_processed += other.files_processed;
        self.total_readings += other.total_readings;

        for (k, v) in &other.delta_counts {
            *self.delta_counts.entry(*k).or_insert(0) += v;
        }

        for (k, v) in &other.zero_run_lengths {
            *self.zero_run_lengths.entry(*k).or_insert(0) += v;
        }

        for (k, v) in &other.gap_counts {
            *self.gap_counts.entry(*k).or_insert(0) += v;
        }

        for (k, v) in &other.value_counts {
            *self.value_counts.entry(*k).or_insert(0) += v;
        }

        self.max_zero_run = self.max_zero_run.max(other.max_zero_run);
        self.max_delta = self.max_delta.max(other.max_delta);
        self.min_delta = self.min_delta.min(other.min_delta);
        self.max_value = self.max_value.max(other.max_value);
        self.min_value = self.min_value.min(other.min_value);
        self.actual_compressed_bytes += other.actual_compressed_bytes;
        self.raw_input_bytes += other.raw_input_bytes;
        self.compression_ratios.extend(&other.compression_ratios);
    }

    fn print_report(&self) {
        let label_upper = self.label.to_uppercase();
        let label_title = format!("{}{}", &self.label[..1].to_uppercase(), &self.label[1..]);
        let unit = if self.label == "temperature" { "°C" } else if self.label == "humidity" { "%" } else { "" };

        println!("\n{}", "=".repeat(70));
        println!("{} DELTA FREQUENCY ANALYSIS", label_upper);
        println!("{}", "=".repeat(70));
        println!();
        println!(
            "Analyzed: {} files, {} readings",
            self.files_processed,
            format_num(self.total_readings)
        );
        println!(
            "{} range: {}{} to {}{}",
            label_title, self.min_value, unit, self.max_value, unit
        );
        println!("Delta range: {} to {}", self.min_delta, self.max_delta);
        println!();

        // Delta distribution by encoding tier
        println!("{}", "-".repeat(70));
        println!("DELTA DISTRIBUTION BY ENCODING TIER");
        println!("{}", "-".repeat(70));

        let total = self.total_readings as f64;

        // Count by tier
        let zeros: u64 = *self.delta_counts.get(&0).unwrap_or(&0);
        let plus_one: u64 = *self.delta_counts.get(&1).unwrap_or(&0);
        let minus_one: u64 = *self.delta_counts.get(&-1).unwrap_or(&0);
        let plus_two: u64 = *self.delta_counts.get(&2).unwrap_or(&0);
        let minus_two: u64 = *self.delta_counts.get(&-2).unwrap_or(&0);

        let mut tier_3_10: u64 = 0;
        let mut tier_11_plus: u64 = 0;

        for (delta, count) in &self.delta_counts {
            let abs_delta = delta.abs();
            if (3..=10).contains(&abs_delta) {
                tier_3_10 += count;
            } else if abs_delta >= 11 {
                tier_11_plus += count;
            }
        }

        let pm1 = plus_one + minus_one;
        let pm2 = plus_two + minus_two;

        println!(
            "  delta=0:     {:>10} ({:>6.2}%) - 1 bit each (zero-run)",
            format_num(zeros),
            100.0 * zeros as f64 / total
        );
        println!(
            "  delta=±1:    {:>10} ({:>6.2}%) - 3 bits each",
            format_num(pm1),
            100.0 * pm1 as f64 / total
        );
        println!(
            "  delta=±2:    {:>10} ({:>6.2}%) - 5 bits each",
            format_num(pm2),
            100.0 * pm2 as f64 / total
        );
        println!(
            "  delta=±3-10: {:>10} ({:>6.2}%) - 11 bits each",
            format_num(tier_3_10),
            100.0 * tier_3_10 as f64 / total
        );
        println!(
            "  delta=±11+:  {:>10} ({:>6.2}%) - 19 bits each",
            format_num(tier_11_plus),
            100.0 * tier_11_plus as f64 / total
        );

        // Detailed delta breakdown
        println!();
        println!("{}", "-".repeat(70));
        println!("DETAILED DELTA BREAKDOWN (top 20)");
        println!("{}", "-".repeat(70));

        let mut sorted_deltas: Vec<_> = self.delta_counts.iter().collect();
        sorted_deltas.sort_by(|a, b| b.1.cmp(a.1));

        for (delta, count) in sorted_deltas.iter().take(20) {
            let pct = 100.0 * **count as f64 / total;
            println!(
                "  delta={:>4}: {:>12} ({:>6.3}%)",
                delta,
                format_num(**count),
                pct
            );
        }

        // Zero-run length distribution
        println!();
        println!("{}", "-".repeat(70));
        println!("ZERO-RUN LENGTH DISTRIBUTION");
        println!("{}", "-".repeat(70));
        println!("Max zero-run observed: {}", self.max_zero_run);

        let total_runs: u64 = self.zero_run_lengths.values().sum();
        let total_zeros_in_runs: u64 = self
            .zero_run_lengths
            .iter()
            .map(|(len, count)| u64::from(*len) * count)
            .sum();

        println!("Total zero-runs: {}", format_num(total_runs));
        println!(
            "Total zeros in runs: {} (matches delta=0 above: {})",
            format_num(total_zeros_in_runs),
            if total_zeros_in_runs == zeros {
                "✓"
            } else {
                "✗"
            }
        );
        println!();

        // Bucket by encoding tier
        let mut tier_1_7: u64 = 0;
        let mut tier_8_21: u64 = 0;
        let mut tier_22_149: u64 = 0;
        let mut tier_150_plus: u64 = 0;

        for (len, count) in &self.zero_run_lengths {
            match *len {
                1..=7 => tier_1_7 += count,
                8..=21 => tier_8_21 += count,
                22..=149 => tier_22_149 += count,
                _ => tier_150_plus += count,
            }
        }

        let tr = total_runs as f64;
        println!(
            "  1-7 (individual bits): {:>10} ({:>6.2}%)",
            format_num(tier_1_7),
            100.0 * tier_1_7 as f64 / tr
        );
        println!(
            "  8-21 (9-bit encoding): {:>10} ({:>6.2}%)",
            format_num(tier_8_21),
            100.0 * tier_8_21 as f64 / tr
        );
        println!(
            "  22-149 (13-bit enc.):  {:>10} ({:>6.2}%)",
            format_num(tier_22_149),
            100.0 * tier_22_149 as f64 / tr
        );
        println!(
            "  150+ (multiple 13b):   {:>10} ({:>6.2}%)",
            format_num(tier_150_plus),
            100.0 * tier_150_plus as f64 / tr
        );

        // Gap analysis
        println!();
        println!("{}", "-".repeat(70));
        println!("GAP ANALYSIS");
        println!("{}", "-".repeat(70));

        let total_gaps: u64 = self.gap_counts.values().sum();
        println!("Total gaps: {}", format_num(total_gaps));

        if !self.gap_counts.is_empty() {
            // Detailed gap size distribution
            let single_gaps = *self.gap_counts.get(&1).unwrap_or(&0);
            let mut multi_gaps_2_65: u64 = 0;
            let mut multi_gaps_66_plus: u64 = 0;

            for (gap_size, count) in &self.gap_counts {
                match *gap_size {
                    1 => {} // already counted
                    2..=65 => multi_gaps_2_65 += count,
                    _ => multi_gaps_66_plus += count,
                }
            }

            let tg = total_gaps as f64;
            println!();
            println!("Gap size distribution:");
            println!(
                "  1 interval (3 bits):     {:>10} ({:>6.2}%)",
                format_num(single_gaps),
                100.0 * single_gaps as f64 / tg
            );
            println!(
                "  2-65 intervals (14 bits):{:>10} ({:>6.2}%)",
                format_num(multi_gaps_2_65),
                100.0 * multi_gaps_2_65 as f64 / tg
            );
            println!(
                "  66+ intervals (multi):   {:>10} ({:>6.2}%)",
                format_num(multi_gaps_66_plus),
                100.0 * multi_gaps_66_plus as f64 / tg
            );

            // Detailed breakdown by gap size
            println!();
            println!("Detailed gap sizes (top 20):");
            let mut sorted_gaps: Vec<_> = self.gap_counts.iter().collect();
            sorted_gaps.sort_by(|a, b| a.0.cmp(b.0)); // Sort by gap size
            for (gap_size, count) in sorted_gaps.iter().take(20) {
                let bits = match **gap_size {
                    1 => 3,
                    2..=65 => 14,
                    n => 14 * u64::from(n).div_ceil(64), // multiple 14-bit chunks
                };
                println!(
                    "  {:>3} intervals: {:>10} ({:>6.3}%) - {} bits each",
                    gap_size,
                    format_num(**count),
                    100.0 * **count as f64 / tg,
                    bits
                );
            }
        }

        // Bit cost analysis
        println!();
        println!("{}", "-".repeat(70));
        println!("BIT COST ANALYSIS (current encoding)");
        println!("{}", "-".repeat(70));

        // Calculate bits for zero runs (NEW encoding)
        let mut zero_run_bits: u64 = 0;
        for (len, count) in &self.zero_run_lengths {
            let bits_per_run = match *len {
                1..=7 => u64::from(*len), // individual bits
                8..=21 => 9,          // 11110 + 4 bits
                22..=149 => 13,       // 111110 + 7 bits
                _ => 13 * u64::from(*len).div_ceil(128), // multiple 13-bit chunks
            };
            zero_run_bits += bits_per_run * count;
        }

        // Calculate gap bits (NEW encoding: single gap = 3 bits, multi = 14 bits)
        let single_gaps_count = *self.gap_counts.get(&1).unwrap_or(&0);
        let mut multi_gaps_count: u64 = 0;
        let mut multi_gaps_bits: u64 = 0;
        for (gap_size, count) in &self.gap_counts {
            match *gap_size {
                1 => {} // already counted
                2..=65 => {
                    multi_gaps_count += count;
                    multi_gaps_bits += count * 14;
                }
                n => {
                    multi_gaps_count += count;
                    multi_gaps_bits += count * 14 * u64::from(n).div_ceil(64);
                }
            }
        }
        let gap_bits: u64 = single_gaps_count * 3 + multi_gaps_bits;

        let pm1_bits = pm1 * 3;
        let pm2_bits = pm2 * 5;           // NEW: 5 bits
        let tier_3_10_bits = tier_3_10 * 11;  // NEW: 11 bits
        let tier_11_plus_bits = tier_11_plus * 19;  // NEW: 19 bits

        let total_bits = zero_run_bits + pm1_bits + pm2_bits + tier_3_10_bits + tier_11_plus_bits + gap_bits;

        println!(
            "  Zero runs:   {:>12} bits ({:>5.1}%)",
            format_num(zero_run_bits),
            100.0 * zero_run_bits as f64 / total_bits as f64
        );
        println!(
            "  ±1 deltas:   {:>12} bits ({:>5.1}%)",
            format_num(pm1_bits),
            100.0 * pm1_bits as f64 / total_bits as f64
        );
        println!(
            "  ±2 deltas:   {:>12} bits ({:>5.1}%)",
            format_num(pm2_bits),
            100.0 * pm2_bits as f64 / total_bits as f64
        );
        println!(
            "  ±3-10:       {:>12} bits ({:>5.1}%)",
            format_num(tier_3_10_bits),
            100.0 * tier_3_10_bits as f64 / total_bits as f64
        );
        println!(
            "  ±11+:        {:>12} bits ({:>5.1}%)",
            format_num(tier_11_plus_bits),
            100.0 * tier_11_plus_bits as f64 / total_bits as f64
        );
        println!(
            "  Gaps:        {:>12} bits ({:>5.1}%)",
            format_num(gap_bits),
            100.0 * gap_bits as f64 / total_bits as f64
        );
        println!("  ─────────────────────────────────");
        println!(
            "  TOTAL:       {:>12} bits ({} bytes)",
            format_num(total_bits),
            format_num(total_bits.div_ceil(8))
        );

        // Alternative schemes (compared to NEW encoding)
        println!();
        println!("{}", "-".repeat(70));
        println!("ALTERNATIVE ENCODING ANALYSIS");
        println!("{}", "-".repeat(70));

        // Alternative 1: Remove ±2 tier, merge into ±3-10
        let alt1_pm2_bits = pm2 * 11; // ±2 uses 11 bits instead of 5
        let alt1_total = zero_run_bits + pm1_bits + alt1_pm2_bits + tier_3_10_bits + tier_11_plus_bits + gap_bits;
        let alt1_diff = alt1_total as i64 - total_bits as i64;

        println!("Alternative 1: Remove ±2 tier (merge into ±3-10)");
        println!(
            "  ±2 would use 11 bits instead of 5: {} bits ({:+} bits, {:+.2}%)",
            format_num(alt1_total),
            alt1_diff,
            100.0 * alt1_diff as f64 / total_bits as f64
        );

        // Alternative 2: If all gaps used 14 bits (no single-gap optimization)
        let old_gap_bits = total_gaps * 14; // old scheme: all gaps 14 bits
        let alt2_total = zero_run_bits + pm1_bits + pm2_bits + tier_3_10_bits + tier_11_plus_bits + old_gap_bits;
        let alt2_diff = alt2_total as i64 - total_bits as i64;

        println!();
        println!("Alternative 2: No single-gap optimization (all gaps 14 bits)");
        println!(
            "  All gaps 14 bits: {} bits ({:+} bits, {:+.2}%)",
            format_num(alt2_total),
            alt2_diff,
            100.0 * alt2_diff as f64 / total_bits as f64
        );

        // Show breakdown of gap savings
        let single_gap_savings = single_gaps_count * (14 - 3);
        println!(
            "  Single-gap optimization saves: {} bits ({} single gaps × 11 bits)",
            format_num(single_gap_savings),
            format_num(single_gaps_count)
        );

        // Summary table
        println!();
        println!("{}", "-".repeat(70));
        println!("ENCODING SUMMARY (all events)");
        println!("{}", "-".repeat(70));

        let total_events = self.total_readings + total_gaps;
        println!(
            "  Total events: {} ({} readings + {} gaps)",
            format_num(total_events),
            format_num(self.total_readings),
            format_num(total_gaps)
        );
        println!();
        println!("  Event Type      Count          %      Bits/ea   Total Bits      Bit %");
        println!("  ─────────────────────────────────────────────────────────────────────");

        // Calculate average bits per zero (accounting for RLE)
        let avg_bits_per_zero = if zeros > 0 { zero_run_bits as f64 / zeros as f64 } else { 1.0 };

        println!(
            "  Zero deltas   {:>12}   {:>6.2}%    ~{:.2}      {:>12}   {:>5.1}%",
            format_num(zeros),
            100.0 * zeros as f64 / total_events as f64,
            avg_bits_per_zero,
            format_num(zero_run_bits),
            100.0 * zero_run_bits as f64 / total_bits as f64
        );
        println!(
            "  ±1 deltas     {:>12}   {:>6.2}%      3       {:>12}   {:>5.1}%",
            format_num(pm1),
            100.0 * pm1 as f64 / total_events as f64,
            format_num(pm1_bits),
            100.0 * pm1_bits as f64 / total_bits as f64
        );
        println!(
            "  ±2 deltas     {:>12}   {:>6.2}%      5       {:>12}   {:>5.1}%",
            format_num(pm2),
            100.0 * pm2 as f64 / total_events as f64,
            format_num(pm2_bits),
            100.0 * pm2_bits as f64 / total_bits as f64
        );
        println!(
            "  ±3-10 deltas  {:>12}   {:>6.2}%     11       {:>12}   {:>5.1}%",
            format_num(tier_3_10),
            100.0 * tier_3_10 as f64 / total_events as f64,
            format_num(tier_3_10_bits),
            100.0 * tier_3_10_bits as f64 / total_bits as f64
        );
        println!(
            "  ±11+ deltas   {:>12}   {:>6.2}%     19       {:>12}   {:>5.1}%",
            format_num(tier_11_plus),
            100.0 * tier_11_plus as f64 / total_events as f64,
            format_num(tier_11_plus_bits),
            100.0 * tier_11_plus_bits as f64 / total_bits as f64
        );
        // Calculate avg bits per gap
        let avg_gap_bits = if total_gaps > 0 { gap_bits as f64 / total_gaps as f64 } else { 3.0 };
        println!(
            "  Gaps          {:>12}   {:>6.2}%   ~{:.1}       {:>12}   {:>5.1}%",
            format_num(total_gaps),
            100.0 * total_gaps as f64 / total_events as f64,
            avg_gap_bits,
            format_num(gap_bits),
            100.0 * gap_bits as f64 / total_bits as f64
        );
        println!("  ─────────────────────────────────────────────────────────────────────");
        println!(
            "  TOTAL         {:>12}   100.00%            {:>12}   100.0%",
            format_num(total_events),
            format_num(total_bits)
        );
        println!();
        println!(
            "  Compressed size: {} bytes ({:.1}x compression vs 12 bytes/reading)",
            format_num(total_bits.div_ceil(8)),
            (self.total_readings * 12) as f64 / total_bits.div_ceil(8) as f64
        );

        // Recommendations
        println!();
        println!("{}", "=".repeat(70));
        println!("CURRENT ENCODING EFFECTIVENESS");
        println!("{}", "=".repeat(70));

        let pm2_pct = 100.0 * pm2 as f64 / total_events as f64;
        let pm1_pct = 100.0 * pm1 as f64 / total_events as f64;
        let zero_pct = 100.0 * zeros as f64 / total_events as f64;
        let gap_pct = 100.0 * total_gaps as f64 / total_events as f64;
        let single_gap_pct = if total_gaps > 0 { 100.0 * single_gaps_count as f64 / total_gaps as f64 } else { 0.0 };

        println!(
            "• Zero deltas dominate ({zero_pct:.2}%). RLE scheme is effective."
        );
        println!(
            "• ±1 deltas are common ({pm1_pct:.2}%). 3-bit encoding is appropriate."
        );
        println!(
            "• ±2 deltas are rare ({pm2_pct:.2}%). 5-bit encoding keeps prefix tree balanced."
        );
        println!(
            "• Gaps ({gap_pct:.2}%): {single_gap_pct:.1}% are single-interval (3 bits), rest use 14 bits."
        );
        if single_gap_pct > 90.0 && total_gaps > 0 {
            println!(
                "  → Single-gap optimization is highly effective: saves {} bits",
                format_num(single_gap_savings)
            );
        }

        // Actual compression stats
        println!();
        println!("{}", "=".repeat(70));
        println!("ACTUAL COMPRESSION (nibblerun)");
        println!("{}", "=".repeat(70));
        let actual_ratio = self.raw_input_bytes as f64 / self.actual_compressed_bytes as f64;
        let estimated_ratio = (self.total_readings * 12) as f64 / total_bits.div_ceil(8) as f64;
        println!(
            "  Raw input size:        {} bytes ({} readings × 12 bytes)",
            format_num(self.raw_input_bytes),
            format_num(self.total_readings)
        );
        println!(
            "  Actual compressed:     {} bytes",
            format_num(self.actual_compressed_bytes)
        );
        println!(
            "  Actual compression:    {:.1}x",
            actual_ratio
        );
        println!(
            "  Estimated (bit math):  {:.1}x",
            estimated_ratio
        );
        println!(
            "  Estimation accuracy:   {:.1}%",
            (estimated_ratio / actual_ratio) * 100.0
        );
    }

    fn generate_html(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Calculate all the statistics we need for charts
        let total = self.total_readings as f64;
        let zeros: u64 = *self.delta_counts.get(&0).unwrap_or(&0);
        let plus_one: u64 = *self.delta_counts.get(&1).unwrap_or(&0);
        let minus_one: u64 = *self.delta_counts.get(&-1).unwrap_or(&0);
        let plus_two: u64 = *self.delta_counts.get(&2).unwrap_or(&0);
        let minus_two: u64 = *self.delta_counts.get(&-2).unwrap_or(&0);

        let mut tier_3_10: u64 = 0;
        let mut tier_11_plus: u64 = 0;
        for (delta, count) in &self.delta_counts {
            let abs_delta = delta.abs();
            if (3..=10).contains(&abs_delta) {
                tier_3_10 += count;
            } else if abs_delta >= 11 {
                tier_11_plus += count;
            }
        }

        let pm1 = plus_one + minus_one;
        let pm2 = plus_two + minus_two;
        let total_gaps: u64 = self.gap_counts.values().sum();

        // Bit costs
        let mut zero_run_bits: u64 = 0;
        for (len, count) in &self.zero_run_lengths {
            let bits_per_run = match *len {
                1..=7 => u64::from(*len),
                8..=21 => 9,
                22..=149 => 13,
                _ => 13 * u64::from(*len).div_ceil(128),
            };
            zero_run_bits += bits_per_run * count;
        }

        let single_gaps_count = *self.gap_counts.get(&1).unwrap_or(&0);
        let mut multi_gaps_count: u64 = 0;
        let mut multi_gaps_bits: u64 = 0;
        for (gap_size, count) in &self.gap_counts {
            match *gap_size {
                1 => {}
                2..=65 => {
                    multi_gaps_count += count;
                    multi_gaps_bits += count * 14;
                }
                n => {
                    multi_gaps_count += count;
                    multi_gaps_bits += count * 14 * u64::from(n).div_ceil(64);
                }
            }
        }
        let single_gap_bits = single_gaps_count * 3;
        let gap_bits: u64 = single_gap_bits + multi_gaps_bits;

        let pm1_bits = pm1 * 3;
        let pm2_bits = pm2 * 5;
        let tier_3_10_bits = tier_3_10 * 11;
        let tier_11_plus_bits = tier_11_plus * 19;
        let total_bits = zero_run_bits + pm1_bits + pm2_bits + tier_3_10_bits + tier_11_plus_bits + gap_bits;

        // Delta distribution data (sorted by delta value, -15 to +15 to show tier boundaries)
        let mut delta_labels = Vec::new();
        let mut delta_values = Vec::new();
        for d in -15..=15 {
            delta_labels.push(d.to_string());
            delta_values.push(*self.delta_counts.get(&d).unwrap_or(&0));
        }

        // Zero-run length distribution (1-160 to show tier boundaries at 7, 21, 149)
        let mut zr_labels = Vec::new();
        let mut zr_values = Vec::new();
        for len in 1..=160 {
            zr_labels.push(len.to_string());
            zr_values.push(*self.zero_run_lengths.get(&len).unwrap_or(&0));
        }

        // Gap size distribution (1-70 to show tier boundary at 65)
        let mut gap_labels = Vec::new();
        let mut gap_values = Vec::new();
        for g in 1..=70 {
            gap_labels.push(g.to_string());
            gap_values.push(*self.gap_counts.get(&g).unwrap_or(&0));
        }

        // Value distribution (use actual range from data)
        let value_min = self.min_value.max(-50); // Clamp for display
        let value_max = self.max_value.min(100);
        let unit = if self.label == "temperature" { "°" } else if self.label == "humidity" { "%" } else { "" };
        let mut value_labels = Vec::new();
        let mut value_values = Vec::new();
        for v in value_min..=value_max {
            value_labels.push(format!("{}{}", v, unit));
            value_values.push(*self.value_counts.get(&v).unwrap_or(&0));
        }

        // Cumulative delta distribution (for CDF chart)
        let mut sorted_deltas: Vec<_> = self.delta_counts.iter().collect();
        sorted_deltas.sort_by_key(|(d, _)| *d);
        let mut cumulative_delta_labels = Vec::new();
        let mut cumulative_delta_values = Vec::new();
        let mut cumsum: u64 = 0;
        for (delta, count) in &sorted_deltas {
            cumsum += *count;
            cumulative_delta_labels.push(delta.to_string());
            cumulative_delta_values.push((cumsum as f64 / total * 100.0) as f64);
        }

        // Cumulative zero-run distribution
        let mut sorted_zr: Vec<_> = self.zero_run_lengths.iter().collect();
        sorted_zr.sort_by_key(|(len, _)| *len);
        let total_runs: u64 = self.zero_run_lengths.values().sum();
        let mut cumulative_zr_labels = Vec::new();
        let mut cumulative_zr_values = Vec::new();
        let mut zr_cumsum: u64 = 0;
        for (len, count) in &sorted_zr {
            zr_cumsum += *count;
            cumulative_zr_labels.push(len.to_string());
            cumulative_zr_values.push((zr_cumsum as f64 / total_runs as f64 * 100.0) as f64);
        }

        // Calculate actual compression ratio
        let actual_ratio = self.raw_input_bytes as f64 / self.actual_compressed_bytes as f64;
        let raw_mb = self.raw_input_bytes as f64 / 1_000_000.0;
        let compressed_mb = self.actual_compressed_bytes as f64 / 1_000_000.0;
        let label_title = format!("{}{}", &self.label[..1].to_uppercase(), &self.label[1..]);
        let unit_full = if self.label == "temperature" { "°C" } else if self.label == "humidity" { "%" } else { "" };

        // Compression ratio distribution
        let mut sorted_ratios = self.compression_ratios.clone();
        sorted_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile = |p: usize| -> f32 {
            if sorted_ratios.is_empty() { return 0.0; }
            let idx = (p * sorted_ratios.len() / 100).min(sorted_ratios.len() - 1);
            sorted_ratios[idx]
        };

        let p1 = percentile(1);
        let p5 = percentile(5);
        let p10 = percentile(10);
        let p25 = percentile(25);
        let p50 = percentile(50);
        let p75 = percentile(75);
        let p90 = percentile(90);
        let p95 = percentile(95);
        let p99 = percentile(99);
        let min_ratio = sorted_ratios.first().copied().unwrap_or(0.0);
        let max_ratio = sorted_ratios.last().copied().unwrap_or(0.0);

        // Create histogram buckets for compression ratio
        let bucket_size = 5.0_f32; // 5x per bucket
        let min_bucket = (min_ratio / bucket_size).floor() * bucket_size;
        let max_bucket = (max_ratio / bucket_size).ceil() * bucket_size;
        let num_buckets = ((max_bucket - min_bucket) / bucket_size).ceil() as usize + 1;

        let mut histogram_labels: Vec<String> = Vec::new();
        let mut histogram_values: Vec<u64> = vec![0; num_buckets];

        for i in 0..num_buckets {
            let bucket_start = min_bucket + (i as f32 * bucket_size);
            histogram_labels.push(format!("{:.0}x", bucket_start));
        }

        for &ratio in &sorted_ratios {
            let bucket_idx = ((ratio - min_bucket) / bucket_size).floor() as usize;
            if bucket_idx < histogram_values.len() {
                histogram_values[bucket_idx] += 1;
            }
        }

        write!(file, r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NibbleRun {label_title} Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
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
        .plot {{ height: 300px; }}
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
        <h1>NibbleRun {label_title} Analysis</h1>
        <p class="subtitle">{files} sensor files · {readings} total readings</p>
    </header>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value green">{compression_ratio:.1}x</div>
            <div class="metric-label">Compression Ratio</div>
        </div>
        <div class="metric">
            <div class="metric-value">{raw_mb:.1} MB</div>
            <div class="metric-label">Raw Size</div>
        </div>
        <div class="metric">
            <div class="metric-value green">{compressed_mb:.2} MB</div>
            <div class="metric-label">Compressed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{zero_pct:.1}%</div>
            <div class="metric-label">Zero Deltas</div>
        </div>
        <div class="metric">
            <div class="metric-value">{value_range}</div>
            <div class="metric-label">{label_title} Range</div>
        </div>
    </div>

    <div class="grid">
        <div class="panel wide">
            <div class="panel-title">Delta Value Distribution</div>
            <div id="deltaChart" class="plot"></div>
        </div>

        <div class="panel">
            <div class="panel-title">Bit Cost by Encoding Tier</div>
            <div id="bitCostChart" class="plot"></div>
        </div>

        <div class="panel">
            <div class="panel-title">Event Type Distribution</div>
            <div id="eventChart" class="plot"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Zero-Run Length Distribution</div>
            <div id="zeroRunChart" class="plot"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">{label_title} Distribution</div>
            <div id="valueChart" class="plot"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Gap Size Distribution</div>
            <div id="gapChart" class="plot"></div>
        </div>

        <div class="panel">
            <div class="panel-title">Cumulative Delta Distribution</div>
            <div id="cdfDeltaChart" class="plot"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Encoding Efficiency: Events vs Bits by Tier</div>
            <div id="efficiencyChart" class="plot" style="height: 450px;"></div>
        </div>

        <div class="panel wide">
            <div class="panel-title">Compression Ratio Distribution (per file)</div>
            <div id="compressionDistChart" class="plot"></div>
        </div>

        <div class="panel">
            <div class="panel-title">Compression Percentiles</div>
            <div id="percentilesChart" class="plot"></div>
        </div>
    </div>

    <footer>Generated by nbl-analyze · NibbleRun Time Series Compression</footer>
</div>

<script>
const darkLayout = {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{ color: '#7d8590', size: 11 }},
    margin: {{ l: 50, r: 20, t: 20, b: 40 }},
    xaxis: {{ gridcolor: '#21262d', zerolinecolor: '#30363d' }},
    yaxis: {{ gridcolor: '#21262d', zerolinecolor: '#30363d' }},
    hoverlabel: {{ bgcolor: '#1c2128', bordercolor: '#30363d', font: {{ color: '#e6edf3' }} }}
}};

const config = {{ displayModeBar: false, responsive: true }};

// Delta distribution with tier coloring
// Tiers: 0=1bit (green), ±1=3bits (blue), ±2=5bits (light blue), ±3-10=11bits (orange), ±11+=19bits (red)
const deltaColors = {delta_labels:?}.map(d => {{
    const v = parseInt(d);
    if (v === 0) return '#3fb950';      // 1 bit - green
    if (Math.abs(v) === 1) return '#58a6ff';  // 3 bits - blue
    if (Math.abs(v) === 2) return '#79c0ff';  // 5 bits - light blue
    if (Math.abs(v) <= 10) return '#f0883e';  // 11 bits - orange
    return '#f85149';                    // 19 bits - red
}});
Plotly.newPlot('deltaChart', [{{
    x: {delta_labels:?},
    y: {delta_values:?},
    type: 'bar',
    marker: {{ color: deltaColors }},
    hovertemplate: 'Delta %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
}}], {{
    ...darkLayout,
    yaxis: {{ ...darkLayout.yaxis, type: 'log', title: {{ text: 'Count (log)', font: {{ size: 11 }} }} }},
    xaxis: {{ ...darkLayout.xaxis, title: {{ text: 'Delta Value', font: {{ size: 11 }} }} }},
    shapes: [
        {{ type: 'line', x0: -1.5, x1: -1.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#484f58', width: 1, dash: 'dot' }} }},
        {{ type: 'line', x0: 1.5, x1: 1.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#484f58', width: 1, dash: 'dot' }} }},
        {{ type: 'line', x0: -2.5, x1: -2.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#484f58', width: 1, dash: 'dot' }} }},
        {{ type: 'line', x0: 2.5, x1: 2.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#484f58', width: 1, dash: 'dot' }} }},
        {{ type: 'line', x0: -10.5, x1: -10.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#484f58', width: 1, dash: 'dot' }} }},
        {{ type: 'line', x0: 10.5, x1: 10.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#484f58', width: 1, dash: 'dot' }} }}
    ],
    annotations: [
        {{ x: 0, y: 1.02, yref: 'paper', text: '1 bit', showarrow: false, font: {{ size: 9, color: '#3fb950' }} }},
        {{ x: 1, y: 1.08, yref: 'paper', text: '3 bits', showarrow: false, font: {{ size: 9, color: '#58a6ff' }} }},
        {{ x: 2, y: 1.02, yref: 'paper', text: '5b', showarrow: false, font: {{ size: 9, color: '#79c0ff' }} }},
        {{ x: 6, y: 1.08, yref: 'paper', text: '11 bits', showarrow: false, font: {{ size: 9, color: '#f0883e' }} }},
        {{ x: 13, y: 1.02, yref: 'paper', text: '19 bits', showarrow: false, font: {{ size: 9, color: '#f85149' }} }}
    ]
}}, config);

// Bit cost pie
Plotly.newPlot('bitCostChart', [{{
    values: [{zero_run_bits}, {pm1_bits}, {pm2_bits}, {tier_3_10_bits}, {tier_11_plus_bits}, {gap_bits}],
    labels: ['Zero runs', '±1', '±2', '±3-10', '±11+', 'Gaps'],
    type: 'pie',
    hole: 0.4,
    marker: {{ colors: ['#3fb950', '#58a6ff', '#79c0ff', '#a5d6ff', '#f85149', '#a371f7'] }},
    textinfo: 'percent',
    textfont: {{ color: '#e6edf3', size: 11 }},
    hovertemplate: '%{{label}}<br>%{{value:,.0f}} bits<br>%{{percent}}<extra></extra>'
}}], {{ ...darkLayout, showlegend: true, legend: {{ font: {{ size: 10 }}, x: 1, y: 0.5 }} }}, config);

// Event distribution pie
Plotly.newPlot('eventChart', [{{
    values: [{zeros}, {pm1}, {pm2}, {tier_3_10}, {tier_11_plus}, {total_gaps}],
    labels: ['Zero (δ=0)', '±1', '±2', '±3-10', '±11+', 'Gaps'],
    type: 'pie',
    hole: 0.4,
    marker: {{ colors: ['#3fb950', '#58a6ff', '#79c0ff', '#a5d6ff', '#f85149', '#a371f7'] }},
    textinfo: 'percent',
    textfont: {{ color: '#e6edf3', size: 11 }},
    hovertemplate: '%{{label}}<br>%{{value:,.0f}} events<br>%{{percent}}<extra></extra>'
}}], {{ ...darkLayout, showlegend: true, legend: {{ font: {{ size: 10 }}, x: 1, y: 0.5 }} }}, config);

// Zero-run distribution with tier coloring
// Tiers: 1-7=individual bits, 8-21=9bits, 22-149=13bits, 150+=multiple 13-bit
const zrColors = {zr_labels:?}.map(d => {{
    const v = parseInt(d);
    if (v <= 7) return '#3fb950';       // 1-7: individual bits (most efficient)
    if (v <= 21) return '#58a6ff';      // 8-21: 9 bits
    if (v <= 149) return '#f0883e';     // 22-149: 13 bits
    return '#f85149';                    // 150+: multiple chunks
}});
Plotly.newPlot('zeroRunChart', [{{
    x: {zr_labels:?},
    y: {zr_values:?},
    type: 'bar',
    marker: {{ color: zrColors }},
    hovertemplate: 'Run length %{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
}}], {{
    ...darkLayout,
    yaxis: {{ ...darkLayout.yaxis, type: 'log', title: {{ text: 'Count (log)', font: {{ size: 11 }} }} }},
    xaxis: {{ ...darkLayout.xaxis, title: {{ text: 'Run Length', font: {{ size: 11 }} }}, dtick: 20 }},
    shapes: [
        {{ type: 'line', x0: 7.5, x1: 7.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#58a6ff', width: 2, dash: 'dash' }} }},
        {{ type: 'line', x0: 21.5, x1: 21.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#f0883e', width: 2, dash: 'dash' }} }},
        {{ type: 'line', x0: 149.5, x1: 149.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#f85149', width: 2, dash: 'dash' }} }}
    ],
    annotations: [
        {{ x: 4, y: 1.05, yref: 'paper', text: '1-7: n bits each', showarrow: false, font: {{ size: 10, color: '#3fb950' }} }},
        {{ x: 14, y: 1.05, yref: 'paper', text: '8-21: 9 bits', showarrow: false, font: {{ size: 10, color: '#58a6ff' }} }},
        {{ x: 85, y: 1.05, yref: 'paper', text: '22-149: 13 bits', showarrow: false, font: {{ size: 10, color: '#f0883e' }} }},
        {{ x: 155, y: 1.05, yref: 'paper', text: '150+', showarrow: false, font: {{ size: 10, color: '#f85149' }} }}
    ]
}}, config);

// Value distribution
Plotly.newPlot('valueChart', [{{
    x: {value_labels:?},
    y: {value_values:?},
    type: 'bar',
    marker: {{ color: '#f78166' }},
    hovertemplate: '%{{x}}<br>Count: %{{y:,.0f}}<extra></extra>'
}}], {{
    ...darkLayout,
    xaxis: {{ ...darkLayout.xaxis, title: {{ text: '{label_title}', font: {{ size: 11 }} }} }},
    yaxis: {{ ...darkLayout.yaxis, title: {{ text: 'Count', font: {{ size: 11 }} }} }}
}}, config);

// Gap distribution with tier coloring
// Tiers: 1=3bits (single gap), 2-65=14bits, 66+=multiple 14-bit chunks
const gapColors = {gap_labels:?}.map(d => {{
    const v = parseInt(d);
    if (v === 1) return '#3fb950';       // 1: 3 bits (optimized single gap)
    if (v <= 65) return '#a371f7';       // 2-65: 14 bits
    return '#f85149';                     // 66+: multiple chunks
}});
Plotly.newPlot('gapChart', [{{
    x: {gap_labels:?},
    y: {gap_values:?},
    type: 'bar',
    marker: {{ color: gapColors }},
    hovertemplate: '%{{x}} intervals<br>Count: %{{y:,.0f}}<extra></extra>'
}}], {{
    ...darkLayout,
    yaxis: {{ ...darkLayout.yaxis, type: 'log', title: {{ text: 'Count (log)', font: {{ size: 11 }} }} }},
    xaxis: {{ ...darkLayout.xaxis, title: {{ text: 'Gap Size (intervals)', font: {{ size: 11 }} }}, dtick: 10 }},
    shapes: [
        {{ type: 'line', x0: 1.5, x1: 1.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#a371f7', width: 2, dash: 'dash' }} }},
        {{ type: 'line', x0: 65.5, x1: 65.5, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#f85149', width: 2, dash: 'dash' }} }}
    ],
    annotations: [
        {{ x: 1, y: 1.05, yref: 'paper', text: '1: 3 bits', showarrow: false, font: {{ size: 10, color: '#3fb950' }} }},
        {{ x: 33, y: 1.05, yref: 'paper', text: '2-65: 14 bits', showarrow: false, font: {{ size: 10, color: '#a371f7' }} }},
        {{ x: 68, y: 1.05, yref: 'paper', text: '66+', showarrow: false, font: {{ size: 10, color: '#f85149' }} }}
    ]
}}, config);

// CDF
Plotly.newPlot('cdfDeltaChart', [{{
    x: {cumulative_delta_labels:?},
    y: {cumulative_delta_values:?},
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    line: {{ color: '#58a6ff', width: 2 }},
    fillcolor: 'rgba(88, 166, 255, 0.1)',
    hovertemplate: 'δ ≤ %{{x}}<br>%{{y:.1f}}% of deltas<extra></extra>'
}}], {{
    ...darkLayout,
    xaxis: {{ ...darkLayout.xaxis, title: {{ text: 'Delta Value', font: {{ size: 11 }} }} }},
    yaxis: {{ ...darkLayout.yaxis, title: {{ text: 'Cumulative %', font: {{ size: 11 }} }}, range: [0, 100] }}
}}, config);

// Combined Efficiency Chart - Events % vs Bits % for each tier
const tiers = [
    {{ name: 'δ=0 (RLE)', events: {zeros}, bits: {zero_run_bits}, color: '#3fb950', bpv: ({zero_run_bits}/{zeros}).toFixed(2) }},
    {{ name: 'δ=±1 (3 bits)', events: {pm1}, bits: {pm1_bits}, color: '#58a6ff', bpv: '3.00' }},
    {{ name: 'δ=±2 (5 bits)', events: {pm2}, bits: {pm2_bits}, color: '#79c0ff', bpv: '5.00' }},
    {{ name: 'δ=±3-10 (11 bits)', events: {tier_3_10}, bits: {tier_3_10_bits}, color: '#f0883e', bpv: '11.00' }},
    {{ name: 'δ=±11+ (19 bits)', events: {tier_11_plus}, bits: {tier_11_plus_bits}, color: '#f85149', bpv: '19.00' }},
    {{ name: 'Gap single (3 bits)', events: {single_gaps_count}, bits: {single_gaps_count} * 3, color: '#a371f7', bpv: '3.00' }},
    {{ name: 'Gap 2-65 (14 bits)', events: {multi_gaps_count}, bits: {multi_gaps_count} * 14, color: '#d2a8ff', bpv: '14.00' }}
];

const totalEvents = tiers.reduce((sum, t) => sum + t.events, 0);
const totalBits = tiers.reduce((sum, t) => sum + t.bits, 0);

const eventPcts = tiers.map(t => t.events / totalEvents * 100);
const bitPcts = tiers.map(t => t.bits / totalBits * 100);

Plotly.newPlot('efficiencyChart', [
    {{
        y: tiers.map(t => t.name),
        x: eventPcts,
        name: 'Events %',
        type: 'bar',
        orientation: 'h',
        marker: {{ color: tiers.map(t => t.color) }},
        text: eventPcts.map(p => p >= 0.5 ? p.toFixed(1) + '%' : ''),
        textposition: 'inside',
        textfont: {{ color: '#fff', size: 11 }},
        hovertemplate: '%{{y}}<br>Events: %{{x:.2f}}%<br>Count: %{{customdata:,.0f}}<extra></extra>',
        customdata: tiers.map(t => t.events)
    }},
    {{
        y: tiers.map(t => t.name),
        x: bitPcts,
        name: 'Bits %',
        type: 'bar',
        orientation: 'h',
        marker: {{ color: tiers.map(t => t.color + '66'), pattern: {{ shape: '/' }} }},
        text: bitPcts.map(p => p >= 0.5 ? p.toFixed(1) + '%' : ''),
        textposition: 'inside',
        textfont: {{ color: '#e6edf3', size: 11 }},
        hovertemplate: '%{{y}}<br>Bits: %{{x:.2f}}%<br>Bits/value: %{{customdata}}<extra></extra>',
        customdata: tiers.map(t => t.bpv)
    }}
], {{
    ...darkLayout,
    barmode: 'group',
    bargap: 0.2,
    bargroupgap: 0.1,
    margin: {{ l: 140, r: 30, t: 40, b: 50 }},
    xaxis: {{
        ...darkLayout.xaxis,
        title: {{ text: 'Percentage', font: {{ size: 11 }} }},
        range: [0, 100],
        ticksuffix: '%'
    }},
    yaxis: {{ ...darkLayout.yaxis }},
    showlegend: true,
    legend: {{ x: 0.75, y: 1.15, orientation: 'h', font: {{ size: 11 }} }},
    title: {{ text: 'Solid = % of events, Striped = % of bits (wider solid bar = more efficient)', font: {{ size: 11, color: '#7d8590' }}, y: 0.98 }}
}}, config);

// Compression ratio distribution histogram
Plotly.newPlot('compressionDistChart', [{{
    x: {histogram_labels:?},
    y: {histogram_values:?},
    type: 'bar',
    marker: {{ color: '#3fb950' }},
    hovertemplate: '%{{x}}<br>Files: %{{y:,.0f}}<extra></extra>'
}}], {{
    ...darkLayout,
    xaxis: {{ ...darkLayout.xaxis, title: {{ text: 'Compression Ratio', font: {{ size: 11 }} }} }},
    yaxis: {{ ...darkLayout.yaxis, title: {{ text: 'Number of Files', font: {{ size: 11 }} }} }},
    shapes: [
        {{ type: 'line', x0: '{p50:.0}x', x1: '{p50:.0}x', y0: 0, y1: 1, yref: 'paper', line: {{ color: '#58a6ff', width: 2, dash: 'dash' }} }}
    ],
    annotations: [
        {{ x: '{p50:.0}x', y: 1.05, yref: 'paper', text: 'Median: {p50:.1}x', showarrow: false, font: {{ size: 10, color: '#58a6ff' }} }}
    ]
}}, config);

// Percentiles chart
Plotly.newPlot('percentilesChart', [{{
    x: ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'],
    y: [{p1:.1}, {p5:.1}, {p10:.1}, {p25:.1}, {p50:.1}, {p75:.1}, {p90:.1}, {p95:.1}, {p99:.1}],
    type: 'scatter',
    mode: 'lines+markers',
    line: {{ color: '#58a6ff', width: 2 }},
    marker: {{ size: 8, color: '#58a6ff' }},
    hovertemplate: '%{{x}}: %{{y:.1f}}x<extra></extra>'
}}], {{
    ...darkLayout,
    xaxis: {{ ...darkLayout.xaxis, title: {{ text: 'Percentile', font: {{ size: 11 }} }} }},
    yaxis: {{ ...darkLayout.yaxis, title: {{ text: 'Compression Ratio', font: {{ size: 11 }} }} }},
    annotations: [
        {{ x: 'p50', y: {p50:.1}, text: '  {p50:.1}x (median)', showarrow: false, xanchor: 'left', font: {{ size: 10, color: '#3fb950' }} }},
        {{ x: 'p1', y: {p1:.1}, text: '  {p1:.1}x (worst 1%)', showarrow: false, xanchor: 'left', font: {{ size: 10, color: '#f85149' }} }},
        {{ x: 'p99', y: {p99:.1}, text: '{p99:.1}x (best 1%)  ', showarrow: false, xanchor: 'right', font: {{ size: 10, color: '#3fb950' }} }}
    ]
}}, config);
</script>
</body>
</html>
"##,
            files = format_num(self.files_processed),
            readings = format_num(self.total_readings),
            compression_ratio = actual_ratio,
            raw_mb = raw_mb,
            compressed_mb = compressed_mb,
            zero_pct = 100.0 * zeros as f64 / total,
            value_range = format!("{}{} to {}{}", self.min_value, unit_full, self.max_value, unit_full),
            label_title = label_title,
            delta_labels = delta_labels,
            delta_values = delta_values,
            zero_run_bits = zero_run_bits,
            pm1_bits = pm1_bits,
            pm2_bits = pm2_bits,
            tier_3_10_bits = tier_3_10_bits,
            tier_11_plus_bits = tier_11_plus_bits,
            gap_bits = gap_bits,
            zr_labels = zr_labels,
            zr_values = zr_values,
            zeros = zeros,
            pm1 = pm1,
            pm2 = pm2,
            tier_3_10 = tier_3_10,
            tier_11_plus = tier_11_plus,
            total_gaps = total_gaps,
            gap_labels = gap_labels,
            gap_values = gap_values,
            value_labels = value_labels,
            value_values = value_values,
            cumulative_delta_labels = cumulative_delta_labels,
            cumulative_delta_values = cumulative_delta_values,
            // Efficiency chart data
            single_gaps_count = single_gaps_count,
            multi_gaps_count = multi_gaps_count,
            // Compression distribution data
            histogram_labels = histogram_labels,
            histogram_values = histogram_values,
            p1 = p1,
            p5 = p5,
            p10 = p10,
            p25 = p25,
            p50 = p50,
            p75 = p75,
            p90 = p90,
            p95 = p95,
            p99 = p99,
        )?;

        Ok(())
    }
}

fn format_num(n: u64) -> String {
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

fn process_file(path: &PathBuf) -> Option<Stats> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);

    let mut stats = Stats {
        min_value: i32::MAX,
        max_value: i32::MIN,
        min_delta: i32::MAX,
        max_delta: i32::MIN,
        ..Default::default()
    };

    // Collect readings for actual compression
    let mut readings: Vec<(u32, i32)> = Vec::new();

    let mut prev_temp: Option<i32> = None;
    let mut prev_ts: Option<u64> = None;
    let mut current_zero_run: u32 = 0;

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
        let temp: i32 = match parts[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Sentinel value -1000 represents a gap in original data
        if temp == -1000 {
            // Flush any pending zero run before the gap
            if current_zero_run > 0 {
                *stats.zero_run_lengths.entry(current_zero_run).or_insert(0) += 1;
                stats.max_zero_run = stats.max_zero_run.max(current_zero_run);
                current_zero_run = 0;
            }
            // Count this as a gap (1 interval skipped per -1000 marker)
            *stats.gap_counts.entry(1).or_insert(0) += 1;
            // Don't reset prev_temp - the gap is between readings
            // but do update prev_ts to track consecutive gaps
            prev_ts = Some(ts as u64);
            continue;
        }

        // Collect for compression
        readings.push((ts, temp));

        stats.total_readings += 1;
        stats.max_value = stats.max_value.max(temp);
        stats.min_value = stats.min_value.min(temp);
        *stats.value_counts.entry(temp).or_insert(0) += 1;

        if let Some(pt) = prev_temp {
            let delta = temp - pt;
            stats.max_delta = stats.max_delta.max(delta);
            stats.min_delta = stats.min_delta.min(delta);
            *stats.delta_counts.entry(delta).or_insert(0) += 1;

            if delta == 0 {
                current_zero_run += 1;
            } else if current_zero_run > 0 {
                *stats
                    .zero_run_lengths
                    .entry(current_zero_run)
                    .or_insert(0) += 1;
                stats.max_zero_run = stats.max_zero_run.max(current_zero_run);
                current_zero_run = 0;
            }
        }

        // Check for timestamp gaps (assuming ~300 second intervals)
        if let Some(pt) = prev_ts {
            let ts_diff = (ts as u64).saturating_sub(pt);
            if ts_diff > 350 {
                // More than one interval (with some slack)
                let gap_intervals = (ts_diff / 300).saturating_sub(1) as u32;
                if gap_intervals > 0 {
                    *stats.gap_counts.entry(gap_intervals).or_insert(0) += 1;
                }
            }
        }

        prev_temp = Some(temp);
        prev_ts = Some(ts as u64);
    }

    // Flush remaining zero run
    if current_zero_run > 0 {
        *stats
            .zero_run_lengths
            .entry(current_zero_run)
            .or_insert(0) += 1;
        stats.max_zero_run = stats.max_zero_run.max(current_zero_run);
    }

    // Actually compress the data with nibblerun
    if !readings.is_empty() {
        let mut encoder: Encoder<i32, 300> = Encoder::new();
        for (ts, temp) in &readings {
            let _ = encoder.append(*ts, *temp);
        }
        let compressed = encoder.to_bytes();
        stats.actual_compressed_bytes = compressed.len() as u64;
        // Raw size: 8 bytes timestamp + 4 bytes i32 temp per reading
        stats.raw_input_bytes = (readings.len() * 12) as u64;
        // Track per-file compression ratio
        if !compressed.is_empty() {
            let ratio = stats.raw_input_bytes as f32 / compressed.len() as f32;
            stats.compression_ratios.push(ratio);
        }
    }

    stats.files_processed = 1;
    Some(stats)
}

fn main() {
    let args = Args::parse();

    println!("Scanning directory: {}", args.dir.display());

    // Collect all CSV files
    let entries: Vec<_> = fs::read_dir(&args.dir)
        .expect("Failed to read directory")
        .filter_map(std::result::Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "csv")
        })
        .collect();

    let total_files = if args.max_files > 0 {
        args.max_files.min(entries.len())
    } else {
        entries.len()
    };

    println!("Found {} CSV files, processing {}...", entries.len(), total_files);
    println!();

    let start = Instant::now();
    let processed = AtomicU64::new(0);
    let global_stats = Mutex::new(Stats {
        min_value: i32::MAX,
        max_value: i32::MIN,
        min_delta: i32::MAX,
        max_delta: i32::MIN,
        label: args.label.clone(),
        ..Default::default()
    });

    // Process files (single-threaded for now, can add rayon later)
    let files_to_process: Vec<_> = entries.into_iter().take(total_files).collect();

    for (i, entry) in files_to_process.iter().enumerate() {
        if let Some(file_stats) = process_file(&entry.path()) {
            let mut gs = global_stats.lock().unwrap();
            gs.merge(&file_stats);
        }

        let count = processed.fetch_add(1, Ordering::Relaxed) + 1;
        if args.progress > 0 && count.is_multiple_of(args.progress as u64) {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = count as f64 / elapsed;
            let eta = (total_files as u64 - count) as f64 / rate;
            eprint!(
                "\rProcessed {count}/{total_files} files ({rate:.0}/s, ETA: {eta:.0}s)  "
            );
        }

        // Also update on every file for first 100
        if i < 100 && i % 10 == 0 {
            eprint!("\rProcessed {i}/{total_files} files  ");
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\rProcessed {} files in {:.2}s ({:.0} files/s)",
        total_files,
        elapsed.as_secs_f64(),
        total_files as f64 / elapsed.as_secs_f64()
    );

    let stats = global_stats.lock().unwrap();
    stats.print_report();

    // Generate HTML report if requested
    if let Some(html_path) = &args.html {
        match stats.generate_html(html_path) {
            Ok(()) => println!("\nHTML report written to: {}", html_path.display()),
            Err(e) => eprintln!("\nFailed to write HTML report: {}", e),
        }
    }
}
