#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::decode_appendable;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to decode_appendable() - should never panic
    // This tests decoding the appendable format (from Encoder::to_bytes())
    // May return empty vec for malformed input, but should not crash
    let _ = decode_appendable::<i8, 300>(data);
    let _ = decode_appendable::<i16, 300>(data);
    let _ = decode_appendable::<i32, 300>(data);
});
