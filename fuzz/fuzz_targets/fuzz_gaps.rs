#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;

const INTERVAL: u16 = 300;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let mut enc: Encoder<i32, INTERVAL> = Encoder::new();
    let base_ts = 1_760_000_000u32;

    // Track which intervals have readings
    let mut intervals_with_data: Vec<u32> = Vec::new();
    let mut prev_temp: Option<i32> = None;

    // Generate readings with gaps
    // Format: each 3 bytes = (interval_idx high, interval_idx low, temp)
    for chunk in data.chunks(3) {
        if chunk.len() < 3 {
            break;
        }

        // Use 2 bytes for interval index to allow gaps
        let interval_idx = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        let temp = chunk[2] as i8 as i32;
        let ts = base_ts + interval_idx * u32::from(INTERVAL);

        // Check delta constraint
        if let Some(prev) = prev_temp {
            let delta = temp - prev;
            if delta < -1024 || delta > 1023 {
                continue;
            }
        }

        if enc.append(ts, temp).is_ok() {
            // Only track if this is a new interval (encoder deduplicates)
            if intervals_with_data.is_empty() || *intervals_with_data.last().unwrap() < interval_idx
            {
                intervals_with_data.push(interval_idx);
                prev_temp = Some(temp);
            }
        }
    }

    let decoded = enc.decode().unwrap();

    // Property: Gaps between readings are preserved correctly
    // The timestamp difference should equal (index_diff * interval)
    if decoded.len() >= 2 && intervals_with_data.len() >= 2 {
        for i in 0..decoded.len() - 1 {
            let ts_diff = decoded[i + 1].ts - decoded[i].ts;
            let expected_idx_diff = intervals_with_data[i + 1] - intervals_with_data[i];
            let expected_ts_diff = expected_idx_diff * u32::from(INTERVAL);

            assert_eq!(
                ts_diff, expected_ts_diff,
                "Gap mismatch between readings {} and {}: expected {} ({} intervals), got {}",
                i,
                i + 1,
                expected_ts_diff,
                expected_idx_diff,
                ts_diff
            );
        }
    }

    // Also verify count matches
    assert_eq!(
        decoded.len(),
        intervals_with_data.len(),
        "Reading count mismatch: expected {}, got {}",
        intervals_with_data.len(),
        decoded.len()
    );
});
