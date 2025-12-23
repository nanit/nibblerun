#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::decode_frozen;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to decode_frozen() - should never panic
    // May return Err or empty vec for malformed input, but should not crash
    let _ = decode_frozen::<i8, 300>(data);
    let _ = decode_frozen::<i32, 300>(data);
});
