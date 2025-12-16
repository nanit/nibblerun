#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::{decode, Encoder};

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes for interval + some data
    if data.len() < 3 {
        return;
    }

    // First 2 bytes determine interval (1-65535)
    let interval = u16::from_le_bytes([data[0], data[1]]).max(1);
    let mut enc: Encoder<i32> = Encoder::new(interval);
    let mut ts = 1_760_000_000u64;

    // Remaining bytes are interpreted as (ts_delta: u16, temp: i8) tuples
    for chunk in data[2..].chunks(3) {
        if chunk.len() < 3 {
            break;
        }
        let delta = u16::from_le_bytes([chunk[0], chunk[1]]) as u64;
        let temp = chunk[2] as i8 as i32;
        ts = ts.saturating_add(delta);
        // Ignore errors - fuzzer may generate invalid data (out of order, delta overflow, etc)
        let _ = enc.append(ts, temp);
    }

    // Property 1: size() == to_bytes().len()
    let bytes = enc.to_bytes();
    assert_eq!(enc.size(), bytes.len(), "size mismatch");

    // Property 2: count() == decode().len()
    let decoded = decode::<i32>(&bytes, u64::from(interval));
    assert_eq!(enc.count(), decoded.len(), "count mismatch");

    // Property 3: direct decode equals decode via bytes
    let direct = enc.decode();
    assert_eq!(direct.len(), decoded.len(), "decode length mismatch");
    for (d, b) in direct.iter().zip(decoded.iter()) {
        assert_eq!(d.ts, b.ts, "timestamp mismatch");
        assert_eq!(d.value, b.value, "value mismatch");
    }

    // Property 4: timestamps are monotonic
    for window in decoded.windows(2) {
        assert!(window[0].ts < window[1].ts, "timestamps not monotonic");
    }

    // Property 5: timestamps are aligned to interval
    if let Some(first) = decoded.first() {
        let base = first.ts;
        for reading in &decoded {
            let offset = reading.ts - base;
            assert_eq!(offset % (interval as u64), 0, "timestamp not aligned");
        }
    }
});
