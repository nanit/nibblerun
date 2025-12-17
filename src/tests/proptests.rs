use crate::{decode, Encoder};
use proptest::prelude::*;

// Start well after EPOCH_BASE to ensure negative jitter doesn't underflow
const BASE_TS: u64 = 1_760_100_000;

/// Generate tests for a specific interval using a macro
macro_rules! proptest_interval {
    ($interval:expr, $mod_name:ident) => {
        mod $mod_name {
            use super::*;

            prop_compose! {
                /// Generate a sequence of readings with jitter relative to the interval
                /// Jitter is ±10% of the interval
                fn arb_readings()(
                    count in 0usize..500,
                )(
                    jitters in prop::collection::vec(-($interval as i64 / 10)..=($interval as i64 / 10), count),
                    temps in prop::collection::vec(-100i32..140, count),
                ) -> Vec<(u64, i32)> {
                    jitters.iter().zip(temps.iter()).enumerate()
                        .map(|(i, (&jitter, &temp))| {
                            let nominal_ts = BASE_TS + (i as u64) * ($interval as u64);
                            let jittered_ts = (nominal_ts as i64 + jitter).max(0) as u64;
                            (jittered_ts, temp)
                        })
                        .collect()
                }
            }

            proptest! {
                /// Property: size() must always equal to_bytes().len()
                #[test]
                fn prop_size_accuracy(readings in arb_readings()) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    for (ts, temp) in readings {
                        enc.append(ts, temp).unwrap();
                    }
                    prop_assert_eq!(enc.size(), enc.to_bytes().len());
                }

                /// Property: decoded length must equal count()
                #[test]
                fn prop_count_consistency(readings in arb_readings()) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    for (ts, temp) in readings {
                        enc.append(ts, temp).unwrap();
                    }
                    let decoded = enc.decode();
                    prop_assert_eq!(decoded.len(), enc.count());
                }

                /// Property: encode then decode via bytes equals direct decode
                #[test]
                fn prop_roundtrip_via_bytes(readings in arb_readings()) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    for (ts, temp) in readings {
                        enc.append(ts, temp).unwrap();
                    }

                    let direct = enc.decode();
                    let via_bytes = decode::<i32, $interval>(&enc.to_bytes());

                    prop_assert_eq!(direct.len(), via_bytes.len());
                    for (d, b) in direct.iter().zip(via_bytes.iter()) {
                        prop_assert_eq!(d.ts, b.ts);
                        prop_assert_eq!(d.value, b.value);
                    }
                }

                /// Property: decoded timestamps are strictly monotonic
                #[test]
                fn prop_monotonic_timestamps(readings in arb_readings()) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    for (ts, temp) in readings {
                        enc.append(ts, temp).unwrap();
                    }

                    let decoded = enc.decode();
                    for window in decoded.windows(2) {
                        prop_assert!(window[0].ts < window[1].ts,
                            "Timestamps not monotonic: {} >= {}", window[0].ts, window[1].ts);
                    }
                }

                /// Property: to_bytes() is idempotent (multiple calls return same result)
                #[test]
                fn prop_idempotent_serialization(readings in arb_readings()) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    for (ts, temp) in readings {
                        enc.append(ts, temp).unwrap();
                    }

                    let bytes1 = enc.to_bytes();
                    let bytes2 = enc.to_bytes();
                    prop_assert_eq!(bytes1, bytes2);
                }

                /// Property: all decoded timestamps are multiples of interval from base
                #[test]
                fn prop_timestamp_alignment(readings in arb_readings()) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    for (ts, temp) in readings {
                        enc.append(ts, temp).unwrap();
                    }

                    let decoded = enc.decode();
                    if let Some(first) = decoded.first() {
                        let base = first.ts;
                        for reading in &decoded {
                            let offset = reading.ts - base;
                            prop_assert_eq!(offset % ($interval as u64), 0,
                                "Timestamp {} not aligned to interval {} from base {}",
                                reading.ts, $interval, base);
                        }
                    }
                }

                /// Property: decoded readings are "close" to input readings
                #[test]
                fn prop_lossy_compression_bounds(readings in arb_readings()) {
                    if readings.is_empty() {
                        return Ok(());
                    }

                    let mut enc = Encoder::<i32, $interval>::new();
                    for &(ts, temp) in &readings {
                        enc.append(ts, temp).unwrap();
                    }

                    let decoded = enc.decode();
                    if decoded.is_empty() {
                        return Ok(());
                    }

                    // Group input readings by their quantized interval
                    let base_ts = readings[0].0;
                    let mut intervals: std::collections::BTreeMap<u64, Vec<i32>> = std::collections::BTreeMap::new();
                    let mut prev_idx = 0u64;

                    for &(ts, temp) in &readings {
                        if ts < base_ts {
                            continue;
                        }
                        let idx = (ts - base_ts) / ($interval as u64);
                        if intervals.is_empty() || idx >= prev_idx {
                            intervals.entry(idx).or_default().push(temp);
                            if idx > prev_idx || intervals.len() == 1 {
                                prev_idx = idx;
                            }
                        }
                    }

                    for reading in &decoded {
                        let idx = (reading.ts - base_ts) / ($interval as u64);

                        if let Some(temps) = intervals.get(&idx) {
                            let min_temp = *temps.iter().min().unwrap();
                            let max_temp = *temps.iter().max().unwrap();

                            prop_assert!(
                                reading.value >= min_temp && reading.value <= max_temp,
                                "Decoded temp {} not in range [{}, {}] for interval {}",
                                reading.value, min_temp, max_temp, idx
                            );
                        }
                    }
                }

                /// Property: with exactly one reading per interval, decoded equals input exactly
                #[test]
                fn prop_single_reading_identity(temps in prop::collection::vec(-100i32..140, 1..100)) {
                    let mut enc = Encoder::<i32, $interval>::new();

                    for (i, &temp) in temps.iter().enumerate() {
                        let ts = BASE_TS + (i as u64) * ($interval as u64);
                        enc.append(ts, temp).unwrap();
                    }

                    let decoded = enc.decode();
                    prop_assert_eq!(decoded.len(), temps.len(),
                        "Decoded count {} doesn't match input count {}", decoded.len(), temps.len());

                    for (i, reading) in decoded.iter().enumerate() {
                        prop_assert_eq!(reading.value, temps[i],
                            "Single reading at interval {} should be exact: expected {}, got {}",
                            i, temps[i], reading.value);
                    }
                }

                /// Property: multiple readings in same interval are averaged correctly
                #[test]
                fn prop_averaging_within_interval(
                    interval_temps in prop::collection::vec(
                        prop::collection::vec(-500i32..500, 1..50),
                        1..20
                    ),
                ) {
                    let mut enc = Encoder::<i32, $interval>::new();

                    for (interval_idx, temps) in interval_temps.iter().enumerate() {
                        let interval_start = BASE_TS + (interval_idx as u64) * ($interval as u64);
                        for (j, &temp) in temps.iter().enumerate() {
                            let offset = (j as u64) % ($interval as u64);
                            enc.append(interval_start + offset, temp).unwrap();
                        }
                    }

                    let decoded = enc.decode();
                    prop_assert_eq!(decoded.len(), interval_temps.len(),
                        "Expected {} intervals, got {}", interval_temps.len(), decoded.len());

                    for (i, (reading, temps)) in decoded.iter().zip(interval_temps.iter()).enumerate() {
                        let sum: i32 = temps.iter().sum();
                        let count = temps.len() as i32;
                        let expected_avg = if sum >= 0 {
                            (sum + count / 2) / count
                        } else {
                            (sum - count / 2) / count
                        };

                        prop_assert_eq!(reading.value, expected_avg,
                            "Interval {}: expected avg {} (sum={}, count={}), got {}",
                            i, expected_avg, sum, count, reading.value);
                    }
                }

                /// Property: timestamps are quantized to interval boundaries
                #[test]
                fn prop_timestamp_quantization(
                    readings_per_interval in prop::collection::vec(1u8..5, 1..50),
                ) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    let mut expected_intervals = Vec::new();

                    for (interval_idx, &count) in readings_per_interval.iter().enumerate() {
                        let interval_start = BASE_TS + (interval_idx as u64) * ($interval as u64);
                        expected_intervals.push(interval_start);

                        for j in 0..count {
                            let jitter = (j as u64 * 17) % ($interval as u64);
                            enc.append(interval_start + jitter, 22).unwrap();
                        }
                    }

                    let decoded = enc.decode();
                    prop_assert_eq!(decoded.len(), expected_intervals.len(),
                        "Expected {} readings, got {}", expected_intervals.len(), decoded.len());

                    for (reading, &expected_ts) in decoded.iter().zip(expected_intervals.iter()) {
                        prop_assert_eq!(reading.ts, expected_ts,
                            "Expected timestamp {}, got {}", expected_ts, reading.ts);
                    }
                }

                /// Property: gaps in input timestamps are preserved in output
                #[test]
                fn prop_gap_preservation(
                    interval_indices in prop::collection::btree_set(0u64..100, 1..30),
                ) {
                    let mut enc = Encoder::<i32, $interval>::new();
                    let indices: Vec<u64> = interval_indices.into_iter().collect();

                    for &idx in &indices {
                        let ts = BASE_TS + idx * ($interval as u64);
                        enc.append(ts, 22).unwrap();
                    }

                    let decoded = enc.decode();
                    prop_assert_eq!(decoded.len(), indices.len(),
                        "Expected {} readings, got {}", indices.len(), decoded.len());

                    for (reading, &expected_idx) in decoded.iter().zip(indices.iter()) {
                        let expected_ts = BASE_TS + expected_idx * ($interval as u64);
                        prop_assert_eq!(reading.ts, expected_ts,
                            "Expected timestamp {} (interval {}), got {} (interval {})",
                            expected_ts, expected_idx, reading.ts,
                            (reading.ts - BASE_TS) / ($interval as u64));
                    }

                    for (i, window) in decoded.windows(2).enumerate() {
                        let ts_diff = window[1].ts - window[0].ts;
                        let idx_diff = indices[i + 1] - indices[i];
                        let expected_diff = idx_diff * ($interval as u64);

                        prop_assert_eq!(ts_diff, expected_diff,
                            "Gap between readings {} and {}: expected {} seconds ({} intervals), got {} seconds",
                            i, i + 1, expected_diff, idx_diff, ts_diff);
                    }
                }

                /// Property: decoded count equals number of unique intervals in input
                #[test]
                fn prop_interval_deduplication(
                    timestamps in prop::collection::vec(0u64..10000, 1..100),
                ) {
                    if timestamps.is_empty() {
                        return Ok(());
                    }

                    let mut sorted_timestamps = timestamps.clone();
                    sorted_timestamps.sort();

                    let mut enc = Encoder::<i32, $interval>::new();

                    for &ts_offset in &sorted_timestamps {
                        enc.append(BASE_TS + ts_offset, 22).unwrap();
                    }

                    let decoded = enc.decode();

                    let base_offset = sorted_timestamps[0];
                    let mut unique_intervals = std::collections::BTreeSet::new();
                    let mut prev_idx: Option<u64> = None;

                    for &ts_offset in &sorted_timestamps {
                        let idx = (ts_offset - base_offset) / ($interval as u64);
                        if prev_idx.is_none() || idx > prev_idx.unwrap() {
                            unique_intervals.insert(idx);
                            prev_idx = Some(idx);
                        }
                    }

                    prop_assert_eq!(decoded.len(), unique_intervals.len(),
                        "Expected {} unique intervals, got {} decoded readings.",
                        unique_intervals.len(), decoded.len());
                }

                /// Property: with gaps and one reading per interval, values are preserved exactly
                #[test]
                fn prop_lossless_with_gaps(
                    interval_indices in prop::collection::btree_set(0u64..100, 1..30),
                    temps in prop::collection::vec(-100i32..140, 30),
                ) {
                    let indices: Vec<u64> = interval_indices.into_iter().collect();
                    if indices.is_empty() {
                        return Ok(());
                    }

                    let mut enc = Encoder::<i32, $interval>::new();
                    let mut expected: Vec<(u64, i32)> = Vec::new();

                    for (i, &idx) in indices.iter().enumerate() {
                        let ts = BASE_TS + idx * ($interval as u64);
                        let temp = temps[i % temps.len()];
                        if enc.append(ts, temp).is_ok() {
                            expected.push((ts, temp));
                        }
                    }

                    let decoded = enc.decode();
                    prop_assert_eq!(decoded.len(), expected.len(),
                        "Expected {} readings, got {}", expected.len(), decoded.len());

                    for (reading, &(expected_ts, expected_temp)) in decoded.iter().zip(expected.iter()) {
                        prop_assert_eq!(reading.ts, expected_ts,
                            "Timestamp mismatch: expected {}, got {}", expected_ts, reading.ts);
                        prop_assert_eq!(reading.value, expected_temp,
                            "Value mismatch at ts {}: expected {}, got {}",
                            expected_ts, expected_temp, reading.value);
                    }
                }
            }
        }
    };
}

// Generate property tests for common intervals
proptest_interval!(60, interval_60);
proptest_interval!(300, interval_300);
proptest_interval!(600, interval_600);
proptest_interval!(3600, interval_3600);
