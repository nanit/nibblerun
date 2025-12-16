#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes for interval + some data
    if data.len() < 5 {
        return;
    }

    // First 2 bytes determine interval (1-65535)
    let interval = u16::from_le_bytes([data[0], data[1]]).max(1);
    let mut enc: Encoder<i32> = Encoder::new(interval);
    let base_ts = 1_760_000_000u64;

    // Generate readings with exactly one per interval
    // Each byte after interval becomes a temp, spaced by interval
    let mut expected_temps = Vec::new();
    let mut prev_temp: Option<i32> = None;

    for (i, &byte) in data[2..].iter().enumerate() {
        let temp = byte as i8 as i32;
        let ts = base_ts + (i as u64) * (interval as u64);

        // Check delta constraint
        if let Some(prev) = prev_temp {
            let delta = temp - prev;
            if delta < -1024 || delta > 1023 {
                continue; // Skip this reading, delta too large
            }
        }

        if enc.append(ts, temp).is_ok() {
            expected_temps.push(temp);
            prev_temp = Some(temp);
        }
    }

    // Property: Single reading per interval should decode to exact input value
    let decoded = enc.decode();

    assert_eq!(
        decoded.len(),
        expected_temps.len(),
        "Count mismatch: expected {}, got {}",
        expected_temps.len(),
        decoded.len()
    );

    for (i, (reading, &expected)) in decoded.iter().zip(expected_temps.iter()).enumerate() {
        assert_eq!(
            reading.value, expected,
            "Single reading at interval {} should be exact: expected {}, got {}",
            i, expected, reading.value
        );
    }
});
