#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::appendable::{append, create, decode, decode_frozen, freeze};

const INTERVAL: u16 = 300;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    // Build appendable buffer from fuzz data
    let base_ts = 1_760_000_000u64;
    let first_val = data[0] as i8;
    let mut buf = create::<i8, INTERVAL>(base_ts, first_val);

    let mut prev_val = first_val as i32;
    let mut interval_idx = 1u64;

    for chunk in data[1..].chunks(2) {
        if chunk.len() < 2 {
            break;
        }

        let gap = (chunk[0] & 0x0F) as u64; // 0-15 interval gap
        let val = chunk[1] as i8;

        // Check delta constraint
        let delta = val as i32 - prev_val;
        if delta < -1024 || delta > 1023 {
            continue;
        }

        interval_idx += gap;
        let ts = base_ts + interval_idx * u64::from(INTERVAL);

        if append::<i8, INTERVAL>(&mut buf, ts, val).is_ok() {
            prev_val = val as i32;
            interval_idx += 1;
        }
    }

    // Property: decode_frozen(freeze(buf)) == decode(buf)
    let frozen = freeze::<i8, INTERVAL>(&buf);
    let from_frozen = decode_frozen::<i8, INTERVAL>(&frozen);
    let from_appendable = decode::<i8, INTERVAL>(&buf);

    assert_eq!(from_frozen.len(), from_appendable.len(), "count mismatch");

    for (f, a) in from_frozen.iter().zip(from_appendable.iter()) {
        assert_eq!(f.ts, a.ts, "timestamp mismatch");
        assert_eq!(f.value, a.value, "value mismatch");
    }
});
