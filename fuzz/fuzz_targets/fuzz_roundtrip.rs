#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::{decode, Encoder};

const INTERVAL: u16 = 300;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut enc: Encoder<i32, INTERVAL> = Encoder::new();
    let mut ts = 1_760_000_000u64;

    // Bytes are interpreted as (ts_delta: u16, temp: i8) tuples
    for chunk in data.chunks(3) {
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
    let decoded = decode::<i32, INTERVAL>(&bytes);
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
            assert_eq!(offset % u64::from(INTERVAL), 0, "timestamp not aligned");
        }
    }
});
