#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;
use std::collections::HashMap;

const INTERVAL: u16 = 300;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let mut enc: Encoder<i32, INTERVAL> = Encoder::new();
    let mut ts = 1_760_000_000u64;
    let mut base_ts: Option<u64> = None;

    // Track min/max temps per interval
    let mut interval_bounds: HashMap<u64, (i32, i32)> = HashMap::new();

    // Bytes are interpreted as (ts_delta: u16, temp: i8) tuples
    for chunk in data.chunks(3) {
        if chunk.len() < 3 {
            break;
        }
        let delta = u16::from_le_bytes([chunk[0], chunk[1]]) as u64;
        let temp = chunk[2] as i8 as i32;
        ts = ts.saturating_add(delta);

        let _count_before = enc.count();
        if enc.append(ts, temp).is_ok() {
            // First successful append sets the base_ts
            if base_ts.is_none() {
                base_ts = Some(ts);
            }

            // Only track if append actually added a reading (not same interval)
            // The encoder uses base_ts from first reading
            let base = base_ts.unwrap();
            let interval_idx = (ts - base) / u64::from(INTERVAL);

            interval_bounds
                .entry(interval_idx)
                .and_modify(|(min, max)| {
                    *min = (*min).min(temp);
                    *max = (*max).max(temp);
                })
                .or_insert((temp, temp));
        }
    }

    // Property: Decoded temps are within [min, max] of inputs for that interval
    let decoded = enc.decode();

    if let Some(base) = base_ts {
        for reading in &decoded {
            let interval_idx = (reading.ts - base) / u64::from(INTERVAL);
            if let Some(&(min, max)) = interval_bounds.get(&interval_idx) {
                assert!(
                    reading.value >= min && reading.value <= max,
                    "Decoded value {} outside bounds [{}, {}] for interval {}",
                    reading.value, min, max, interval_idx
                );
            }
        }
    }
});
