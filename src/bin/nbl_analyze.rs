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
#[command(about = "Analyze temperature sensor data for encoding optimization")]
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
    // Temperature distribution
    temp_counts: HashMap<i32, u64>,
    max_zero_run: u32,
    max_delta: i32,
    min_delta: i32,
    // Temperature range
    max_temp: i32,
    min_temp: i32,
    // Actual compression stats
    actual_compressed_bytes: u64,
    raw_input_bytes: u64,
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

        for (k, v) in &other.temp_counts {
            *self.temp_counts.entry(*k).or_insert(0) += v;
        }

        self.max_zero_run = self.max_zero_run.max(other.max_zero_run);
        self.max_delta = self.max_delta.max(other.max_delta);
        self.min_delta = self.min_delta.min(other.min_delta);
        self.max_temp = self.max_temp.max(other.max_temp);
        self.min_temp = self.min_temp.min(other.min_temp);
        self.actual_compressed_bytes += other.actual_compressed_bytes;
        self.raw_input_bytes += other.raw_input_bytes;
    }

    fn print_report(&self) {
        println!("\n{}", "=".repeat(70));
        println!("DELTA FREQUENCY ANALYSIS");
        println!("{}", "=".repeat(70));
        println!();
        println!(
            "Analyzed: {} files, {} readings",
            self.files_processed,
            format_num(self.total_readings)
        );
        println!(
            "Temperature range: {}°C to {}°C",
            self.min_temp, self.max_temp
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
        let mut multi_gaps_bits: u64 = 0;
        for (gap_size, count) in &self.gap_counts {
            match *gap_size {
                1 => {}
                2..=65 => multi_gaps_bits += count * 14,
                n => multi_gaps_bits += count * 14 * u64::from(n).div_ceil(64),
            }
        }
        let gap_bits: u64 = single_gaps_count * 3 + multi_gaps_bits;

        let pm1_bits = pm1 * 3;
        let pm2_bits = pm2 * 5;
        let tier_3_10_bits = tier_3_10 * 11;
        let tier_11_plus_bits = tier_11_plus * 19;
        let total_bits = zero_run_bits + pm1_bits + pm2_bits + tier_3_10_bits + tier_11_plus_bits + gap_bits;

        // Delta distribution data (sorted by delta value, -20 to +20)
        let mut delta_labels = Vec::new();
        let mut delta_values = Vec::new();
        for d in -20..=20 {
            delta_labels.push(d.to_string());
            delta_values.push(*self.delta_counts.get(&d).unwrap_or(&0));
        }

        // Zero-run length distribution (1-50)
        let mut zr_labels = Vec::new();
        let mut zr_values = Vec::new();
        for len in 1..=50 {
            zr_labels.push(len.to_string());
            zr_values.push(*self.zero_run_lengths.get(&len).unwrap_or(&0));
        }

        // Gap size distribution (1-20)
        let mut gap_labels = Vec::new();
        let mut gap_values = Vec::new();
        for g in 1..=20 {
            gap_labels.push(g.to_string());
            gap_values.push(*self.gap_counts.get(&g).unwrap_or(&0));
        }

        // Temperature distribution (use actual range from data)
        let temp_min = self.min_temp.max(-50); // Clamp for display
        let temp_max = self.max_temp.min(100);
        let mut temp_labels = Vec::new();
        let mut temp_values = Vec::new();
        for t in temp_min..=temp_max {
            temp_labels.push(format!("{}°", t));
            temp_values.push(*self.temp_counts.get(&t).unwrap_or(&0));
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

        write!(file, r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NibbleRun Data Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #1976D2;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .stats-row {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: white;
            padding: 20px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1976D2;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .chart-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-card h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #1976D2;
            padding-bottom: 10px;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        .pie-container {{
            position: relative;
            height: 350px;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>NibbleRun Data Analysis</h1>
    <p class="subtitle">Compression statistics for {files} files, {readings} readings</p>

    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{compression_ratio:.1}x</div>
            <div class="stat-label">Compression Ratio</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{zero_pct:.1}%</div>
            <div class="stat-label">Zero Deltas</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{temp_range}</div>
            <div class="stat-label">Temperature Range</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_bytes}</div>
            <div class="stat-label">Compressed Size</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-card">
            <h3>Delta Distribution</h3>
            <div class="chart-container">
                <canvas id="deltaChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h3>Bit Cost by Encoding Tier</h3>
            <div class="pie-container">
                <canvas id="bitCostChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h3>Zero-Run Length Distribution</h3>
            <div class="chart-container">
                <canvas id="zeroRunChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h3>Event Distribution</h3>
            <div class="pie-container">
                <canvas id="eventChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h3>Gap Size Distribution</h3>
            <div class="chart-container">
                <canvas id="gapChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h3>Temperature Distribution</h3>
            <div class="chart-container">
                <canvas id="tempChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h3>Cumulative Delta Distribution (CDF)</h3>
            <div class="chart-container">
                <canvas id="cdfDeltaChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h3>Cumulative Zero-Run Distribution (CDF)</h3>
            <div class="chart-container">
                <canvas id="cdfZeroRunChart"></canvas>
            </div>
        </div>
    </div>

    <footer>
        Generated by nbl-analyze · NibbleRun Time Series Compression
    </footer>

    <script>
        // Delta distribution chart
        new Chart(document.getElementById('deltaChart'), {{
            type: 'bar',
            data: {{
                labels: {delta_labels:?},
                datasets: [{{
                    label: 'Count',
                    data: {delta_values:?},
                    backgroundColor: 'rgba(25, 118, 210, 0.7)',
                    borderColor: 'rgba(25, 118, 210, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `${{ctx.parsed.y.toLocaleString()}} occurrences`
                        }}
                    }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Delta Value' }} }},
                    y: {{
                        type: 'logarithmic',
                        title: {{ display: true, text: 'Count (log scale)' }},
                        min: 1
                    }}
                }}
            }}
        }});

        // Bit cost pie chart
        new Chart(document.getElementById('bitCostChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Zero runs', '±1 deltas', '±2 deltas', '±3-10 deltas', '±11+ deltas', 'Gaps'],
                datasets: [{{
                    data: [{zero_run_bits}, {pm1_bits}, {pm2_bits}, {tier_3_10_bits}, {tier_11_plus_bits}, {gap_bits}],
                    backgroundColor: [
                        '#4CAF50',
                        '#FF9800',
                        '#FFB74D',
                        '#FFCC80',
                        '#EF5350',
                        '#9C27B0'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right' }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => {{
                                const bits = ctx.parsed;
                                const pct = (bits / {total_bits} * 100).toFixed(1);
                                return `${{bits.toLocaleString()}} bits (${{pct}}%)`;
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Zero-run length chart
        new Chart(document.getElementById('zeroRunChart'), {{
            type: 'bar',
            data: {{
                labels: {zr_labels:?},
                datasets: [{{
                    label: 'Count',
                    data: {zr_values:?},
                    backgroundColor: 'rgba(76, 175, 80, 0.7)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `${{ctx.parsed.y.toLocaleString()}} runs`
                        }}
                    }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Run Length' }} }},
                    y: {{
                        type: 'logarithmic',
                        title: {{ display: true, text: 'Count (log scale)' }},
                        min: 1
                    }}
                }}
            }}
        }});

        // Event distribution pie chart
        new Chart(document.getElementById('eventChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['Zero (δ=0)', '±1', '±2', '±3-10', '±11+', 'Gaps'],
                datasets: [{{
                    data: [{zeros}, {pm1}, {pm2}, {tier_3_10}, {tier_11_plus}, {total_gaps}],
                    backgroundColor: [
                        '#4CAF50',
                        '#FF9800',
                        '#FFB74D',
                        '#FFCC80',
                        '#EF5350',
                        '#9C27B0'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right' }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => {{
                                const total = {total_events};
                                const pct = (ctx.parsed / total * 100).toFixed(2);
                                return `${{ctx.parsed.toLocaleString()}} (${{pct}}%)`;
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Gap size chart
        new Chart(document.getElementById('gapChart'), {{
            type: 'bar',
            data: {{
                labels: {gap_labels:?},
                datasets: [{{
                    label: 'Count',
                    data: {gap_values:?},
                    backgroundColor: 'rgba(156, 39, 176, 0.7)',
                    borderColor: 'rgba(156, 39, 176, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `${{ctx.parsed.y.toLocaleString()}} gaps`
                        }}
                    }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Gap Size (intervals)' }} }},
                    y: {{
                        type: 'logarithmic',
                        title: {{ display: true, text: 'Count (log scale)' }},
                        min: 1
                    }}
                }}
            }}
        }});

        // Temperature distribution chart
        new Chart(document.getElementById('tempChart'), {{
            type: 'bar',
            data: {{
                labels: {temp_labels:?},
                datasets: [{{
                    label: 'Count',
                    data: {temp_values:?},
                    backgroundColor: 'rgba(233, 30, 99, 0.7)',
                    borderColor: 'rgba(233, 30, 99, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `${{ctx.parsed.y.toLocaleString()}} readings`
                        }}
                    }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Temperature' }} }},
                    y: {{ title: {{ display: true, text: 'Count' }} }}
                }}
            }}
        }});

        // Cumulative delta distribution (CDF)
        new Chart(document.getElementById('cdfDeltaChart'), {{
            type: 'line',
            data: {{
                labels: {cumulative_delta_labels:?},
                datasets: [{{
                    label: 'Cumulative %',
                    data: {cumulative_delta_values:?},
                    borderColor: 'rgba(25, 118, 210, 1)',
                    backgroundColor: 'rgba(25, 118, 210, 0.1)',
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `${{ctx.parsed.y.toFixed(2)}}% of deltas ≤ ${{ctx.label}}`
                        }}
                    }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Delta Value' }} }},
                    y: {{
                        title: {{ display: true, text: 'Cumulative %' }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});

        // Cumulative zero-run distribution (CDF)
        new Chart(document.getElementById('cdfZeroRunChart'), {{
            type: 'line',
            data: {{
                labels: {cumulative_zr_labels:?},
                datasets: [{{
                    label: 'Cumulative %',
                    data: {cumulative_zr_values:?},
                    borderColor: 'rgba(76, 175, 80, 1)',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (ctx) => `${{ctx.parsed.y.toFixed(2)}}% of runs ≤ ${{ctx.label}} zeros`
                        }}
                    }}
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Zero-Run Length' }} }},
                    y: {{
                        title: {{ display: true, text: 'Cumulative %' }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"##,
            files = format_num(self.files_processed),
            readings = format_num(self.total_readings),
            compression_ratio = self.raw_input_bytes as f64 / self.actual_compressed_bytes as f64,
            zero_pct = 100.0 * zeros as f64 / total,
            temp_range = format!("{}°C to {}°C", self.min_temp, self.max_temp),
            total_bytes = format_num(self.actual_compressed_bytes),
            delta_labels = delta_labels,
            delta_values = delta_values,
            zero_run_bits = zero_run_bits,
            pm1_bits = pm1_bits,
            pm2_bits = pm2_bits,
            tier_3_10_bits = tier_3_10_bits,
            tier_11_plus_bits = tier_11_plus_bits,
            gap_bits = gap_bits,
            total_bits = total_bits,
            zr_labels = zr_labels,
            zr_values = zr_values,
            zeros = zeros,
            pm1 = pm1,
            pm2 = pm2,
            tier_3_10 = tier_3_10,
            tier_11_plus = tier_11_plus,
            total_gaps = total_gaps,
            total_events = self.total_readings + total_gaps,
            gap_labels = gap_labels,
            gap_values = gap_values,
            temp_labels = temp_labels,
            temp_values = temp_values,
            cumulative_delta_labels = cumulative_delta_labels,
            cumulative_delta_values = cumulative_delta_values,
            cumulative_zr_labels = cumulative_zr_labels,
            cumulative_zr_values = cumulative_zr_values,
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
        min_temp: i32::MAX,
        max_temp: i32::MIN,
        min_delta: i32::MAX,
        max_delta: i32::MIN,
        ..Default::default()
    };

    // Collect readings for actual compression
    let mut readings: Vec<(u64, i32)> = Vec::new();

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

        let ts: u64 = match parts[0].trim().parse() {
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
            prev_ts = Some(ts);
            continue;
        }

        // Collect for compression
        readings.push((ts, temp));

        stats.total_readings += 1;
        stats.max_temp = stats.max_temp.max(temp);
        stats.min_temp = stats.min_temp.min(temp);
        *stats.temp_counts.entry(temp).or_insert(0) += 1;

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
            let ts_diff = ts.saturating_sub(pt);
            if ts_diff > 350 {
                // More than one interval (with some slack)
                let gap_intervals = (ts_diff / 300).saturating_sub(1) as u32;
                if gap_intervals > 0 {
                    *stats.gap_counts.entry(gap_intervals).or_insert(0) += 1;
                }
            }
        }

        prev_temp = Some(temp);
        prev_ts = Some(ts);
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
        min_temp: i32::MAX,
        max_temp: i32::MIN,
        min_delta: i32::MAX,
        max_delta: i32::MIN,
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
