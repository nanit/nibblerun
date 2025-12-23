#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;
use std::collections::BTreeMap;

const INTERVAL: u16 = 300;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let mut enc: Encoder<i32, INTERVAL> = Encoder::new();
    let base_ts = 1_760_000_000u32;

    // Track readings per interval for expected keep-last value
    // Key: interval_idx calculated from the first reading's actual timestamp
    let mut interval_readings: BTreeMap<u32, Vec<i32>> = BTreeMap::new();
    let mut prev_last: Option<i32> = None;
    let mut actual_base_ts: Option<u32> = None;

    // Generate multiple readings per interval
    // Format: each 2 bytes = (offset within interval, temp)
    for (i, chunk) in data.chunks(2).enumerate() {
        if chunk.len() < 2 {
            break;
        }

        let interval_idx = i as u32 / 3; // ~3 readings per interval
        let offset = (chunk[0] as u32) % u32::from(INTERVAL); // offset within interval
        let temp = chunk[1] as i8 as i32;
        let ts = base_ts + interval_idx * u32::from(INTERVAL) + offset;

        // Check if this would cause delta overflow when finalized
        if let Some(prev) = prev_last {
            // Check delta from previous interval's last value
            let readings = interval_readings.get(&interval_idx);
            let current_count = readings.map(|r| r.len()).unwrap_or(0);
            if current_count == 0 {
                // New interval - check delta from previous interval's last value
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
            let actual_interval_idx = (ts - actual_base_ts.unwrap()) / u32::from(INTERVAL);

            let readings = interval_readings.entry(actual_interval_idx).or_default();
            readings.push(temp);

            // Update prev_last when we move to a new interval
            if readings.len() == 1 && actual_interval_idx > 0 {
                if let Some(prev_readings) = interval_readings.get(&(actual_interval_idx - 1)) {
                    if !prev_readings.is_empty() {
                        // Keep-last: use the last value from previous interval
                        prev_last = prev_readings.last().copied();
                    }
                }
            }
        }
    }

    let decoded = enc.decode();

    // Property: Each decoded reading should be the last value for that interval (keep-last)
    if let Some(base) = actual_base_ts {
        for reading in &decoded {
            let interval_idx = (reading.ts - base) / u32::from(INTERVAL);
            if let Some(readings) = interval_readings.get(&interval_idx) {
                if !readings.is_empty() {
                    // Keep-last semantics: expect the last value in the interval
                    let expected_last = *readings.last().unwrap();

                    assert_eq!(
                        reading.value, expected_last,
                        "Keep-last mismatch for interval {}: expected {}, got {} (inputs: {:?})",
                        interval_idx, expected_last, reading.value, readings
                    );
                }
            }
        }
    }
});
