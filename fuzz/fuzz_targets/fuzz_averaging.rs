#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;
use std::collections::BTreeMap;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes for interval + some data
    if data.len() < 5 {
        return;
    }

    // First 2 bytes determine interval (use larger interval so multiple readings fit)
    let interval = (u16::from_le_bytes([data[0], data[1]]).max(100) as u64).max(100);
    let mut enc: Encoder<i32> = Encoder::new(interval as u16);
    let base_ts = 1_760_000_000u64;

    // Track readings per interval for expected average calculation
    // Key: interval_idx calculated from the first reading's actual timestamp
    let mut interval_readings: BTreeMap<u64, Vec<i32>> = BTreeMap::new();
    let mut prev_avg: Option<i32> = None;
    let mut actual_base_ts: Option<u64> = None;

    // Generate multiple readings per interval
    // Format: each 2 bytes = (offset within interval, temp)
    for (i, chunk) in data[2..].chunks(2).enumerate() {
        if chunk.len() < 2 {
            break;
        }

        let interval_idx = i as u64 / 3; // ~3 readings per interval
        let offset = (chunk[0] as u64) % interval; // offset within interval
        let temp = chunk[1] as i8 as i32;
        let ts = base_ts + interval_idx * interval + offset;

        // Check if this would cause delta overflow when finalized
        if let Some(prev) = prev_avg {
            // Rough estimate - actual avg might differ
            let readings = interval_readings.get(&interval_idx);
            let current_count = readings.map(|r| r.len()).unwrap_or(0);
            if current_count == 0 {
                // New interval - check delta from previous interval's avg
                let delta = temp - prev;
                if delta < -1024 || delta > 1023 {
                    continue;
                }
            }
        }

        if enc.append(ts, temp).is_ok() {
            // First successful append sets the actual base_ts (encoder uses this)
            if actual_base_ts.is_none() {
                actual_base_ts = Some(ts);
            }

            // Calculate interval index relative to actual base_ts
            let actual_interval_idx = (ts - actual_base_ts.unwrap()) / interval;

            let readings = interval_readings.entry(actual_interval_idx).or_default();
            readings.push(temp);

            // Update prev_avg when we move to a new interval
            if readings.len() == 1 && actual_interval_idx > 0 {
                if let Some(prev_readings) = interval_readings.get(&(actual_interval_idx - 1)) {
                    if !prev_readings.is_empty() {
                        let sum: i32 = prev_readings.iter().sum();
                        prev_avg = Some(sum / prev_readings.len() as i32);
                    }
                }
            }
        }
    }

    let decoded = enc.decode();

    // Property: Each decoded reading should be the average of inputs for that interval
    if let Some(base) = actual_base_ts {
        for reading in &decoded {
            let interval_idx = (reading.ts - base) / interval;
            if let Some(readings) = interval_readings.get(&interval_idx) {
                if !readings.is_empty() {
                    let sum: i32 = readings.iter().sum();
                    let count = readings.len() as i32;
                    // Use same rounding as encoder: (sum + count/2) / count for positive, adjusted for negative
                    let expected_avg = if sum >= 0 {
                        (sum + count / 2) / count
                    } else {
                        (sum - count / 2) / count
                    };

                    assert_eq!(
                        reading.value, expected_avg,
                        "Average mismatch for interval {}: expected {}, got {} (inputs: {:?})",
                        interval_idx, expected_avg, reading.value, readings
                    );
                }
            }
        }
    }
});
