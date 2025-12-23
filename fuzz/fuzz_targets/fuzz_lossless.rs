#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;

const INTERVAL: u16 = 300;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let mut enc: Encoder<i32, INTERVAL> = Encoder::new();
    let base_ts = 1_760_000_000u32;

    // Track input (interval_idx, temp) pairs
    let mut inputs: Vec<(u32, i32)> = Vec::new();
    let mut prev_temp: Option<i32> = None;
    let mut current_interval: u32 = 0;

    // Each 2 bytes: (gap, temp)
    // gap determines how many intervals to skip (0 = consecutive)
    for chunk in data.chunks(2) {
        if chunk.len() < 2 {
            break;
        }

        let gap = chunk[0] as u32;
        let temp = chunk[1] as i8 as i32;

        // Check delta constraint
        if let Some(prev) = prev_temp {
            let delta = temp - prev;
            if delta < -1024 || delta > 1023 {
                continue;
            }
        }

        // Advance by gap intervals
        current_interval = current_interval.saturating_add(gap);

        // Timestamp exactly at interval boundary
        let ts = base_ts + current_interval * u32::from(INTERVAL);

        if enc.append(ts, temp).is_ok() {
            inputs.push((current_interval, temp));
            prev_temp = Some(temp);
            // Move to next interval for next reading
            current_interval = current_interval.saturating_add(1);
        }
    }

    // Decode and verify lossless roundtrip
    let decoded = enc.decode().unwrap();

    assert_eq!(
        decoded.len(),
        inputs.len(),
        "Count mismatch: expected {}, got {}",
        inputs.len(),
        decoded.len()
    );

    for (reading, &(expected_interval, expected_temp)) in decoded.iter().zip(inputs.iter()) {
        assert_eq!(
            reading.value, expected_temp,
            "Value mismatch: expected {}, got {}",
            expected_temp, reading.value
        );

        // Verify timestamp is exactly at the expected interval boundary
        let expected_ts = base_ts + expected_interval * u32::from(INTERVAL);
        assert_eq!(
            reading.ts, expected_ts,
            "Timestamp mismatch: expected {} (interval {}), got {}",
            expected_ts, expected_interval, reading.ts
        );
    }
});
