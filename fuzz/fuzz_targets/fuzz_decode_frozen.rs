#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::decode;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to decode() (frozen format) - should never panic
    // May return empty vec for malformed input, but should not crash
    let _ = decode::<i8, 300>(data);
    let _ = decode::<i32, 300>(data);
});
