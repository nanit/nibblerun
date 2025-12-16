#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;

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
        let _ = enc.append(ts, temp);
    }

    // Property: Multiple to_bytes() calls return identical results
    let bytes1 = enc.to_bytes();
    let bytes2 = enc.to_bytes();
    let bytes3 = enc.to_bytes();

    assert_eq!(bytes1, bytes2, "to_bytes() not idempotent (1st vs 2nd call)");
    assert_eq!(bytes2, bytes3, "to_bytes() not idempotent (2nd vs 3rd call)");
});
