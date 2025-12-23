#![no_main]

use libfuzzer_sys::fuzz_target;
use nibblerun::Encoder;

fuzz_target!(|data: &[u8]| {
    // Feed arbitrary bytes to Encoder::from_bytes() - should never panic
    // This tests decoding the appendable format (from Encoder::to_bytes())
    // May return Err for malformed input, but should not crash
    if let Ok(enc) = Encoder::<i8, 300>::from_bytes(data) {
        let _ = enc.decode(); // May return Err, that's fine
    }
    if let Ok(enc) = Encoder::<i16, 300>::from_bytes(data) {
        let _ = enc.decode(); // May return Err, that's fine
    }
    if let Ok(enc) = Encoder::<i32, 300>::from_bytes(data) {
        let _ = enc.decode(); // May return Err, that's fine
    }
});
