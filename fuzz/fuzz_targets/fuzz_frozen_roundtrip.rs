#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::{decode, decode_appendable, Encoder};

const INTERVAL: u16 = 300;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    // Build encoder from fuzz data
    let base_ts = 1_760_000_000u32;
    let first_val = data[0] as i8;
    let mut enc: Encoder<i8, INTERVAL> = Encoder::new();
    let _ = enc.append(base_ts, first_val);

    let mut prev_val = first_val as i32;
    let mut interval_idx = 1u32;

    for chunk in data[1..].chunks(2) {
        if chunk.len() < 2 {
            break;
        }

        let gap = (chunk[0] & 0x0F) as u32; // 0-15 interval gap
        let val = chunk[1] as i8;

        // Check delta constraint
        let delta = val as i32 - prev_val;
        if delta < -1024 || delta > 1023 {
            continue;
        }

        interval_idx += gap;
        let ts = base_ts + interval_idx * u32::from(INTERVAL);

        if enc.append(ts, val).is_ok() {
            prev_val = val as i32;
            interval_idx += 1;
        }
    }

    // Property: decode(freeze()) == decode_appendable(to_bytes())
    let frozen = enc.freeze();
    let from_frozen = decode::<i8, INTERVAL>(&frozen);
    let from_appendable = decode_appendable::<i8, INTERVAL>(&enc.to_bytes());

    assert_eq!(from_frozen.len(), from_appendable.len(), "count mismatch");

    for (f, a) in from_frozen.iter().zip(from_appendable.iter()) {
        assert_eq!(f.ts, a.ts, "timestamp mismatch");
        assert_eq!(f.value, a.value, "value mismatch");
    }
});
