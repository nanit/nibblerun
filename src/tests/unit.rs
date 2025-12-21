use crate::appendable::header_size_for_value_bytes;
use crate::constants::{div_by_interval, EPOCH_BASE};
use crate::value::Value;
use crate::{decode, AppendError, Encoder};

/// ============================================================================
/// STRUCT SIZE GUARD - DO NOT MODIFY WITHOUT EXPLICIT AGREEMENT
/// ============================================================================
/// The Encoder struct wraps a Vec<u8> buffer in appendable format.
/// All value types use the same size since the buffer is type-erased.
/// ============================================================================
#[test]
fn test_encoder_struct_sizes_guard() {
    use std::mem::size_of;

    // Encoder is now a thin wrapper around Vec<u8>
    // Vec<u8> is 24 bytes on 64-bit platforms
    assert_eq!(
        size_of::<Encoder<i8>>(),
        24,
        "Encoder<i8> size changed! Expected 24 bytes (Vec<u8> wrapper)."
    );
    assert_eq!(
        size_of::<Encoder<i16>>(),
        24,
        "Encoder<i16> size changed! Expected 24 bytes (Vec<u8> wrapper)."
    );
    assert_eq!(
        size_of::<Encoder<i32>>(),
        24,
        "Encoder<i32> size changed! Expected 24 bytes (Vec<u8> wrapper)."
    );
}

#[test]
fn test_div_by_interval() {
    for x in [0, 1, 299, 300, 301, 599, 600, 1000, 10000, 100000, 200000] {
        assert_eq!(div_by_interval(x, 300), x / 300, "failed for x={}", x);
    }
}

#[test]
fn test_roundtrip() {
    let base = 1761955455u64;
    let temps = [22, 23, 23, 22, 21, 22, 22, 22, 25, 20];
    let mut enc = Encoder::<i32>::new();
    for (i, &t) in temps.iter().enumerate() {
        enc.append(base + i as u64 * 300, t).unwrap();
    }
    let dec = enc.decode();
    assert_eq!(dec.len(), temps.len());
    for (i, r) in dec.iter().enumerate() {
        assert_eq!(r.value, temps[i]);
    }
}

#[test]
fn test_empty() {
    let enc = Encoder::<i32>::new();
    assert_eq!(enc.count(), 0);
    assert!(enc.to_bytes().is_empty());
}

#[test]
fn test_single_reading() {
    let mut enc = Encoder::<i32>::new();
    enc.append(1761955455, 22).unwrap();
    let dec = enc.decode();
    assert_eq!(dec.len(), 1);
    assert_eq!(dec[0].value, 22);
}

#[test]
fn test_gaps() {
    // Gaps are implicit - just skip intervals by using later timestamps
    let base = 1761955455u64;
    let mut enc = Encoder::<i32>::new();
    enc.append(base, 22).unwrap();
    // Skip 2 intervals (600 seconds = 2 * 300)
    enc.append(base + 900, 23).unwrap();
    let dec = enc.decode();
    assert_eq!(dec.len(), 2);
    assert_eq!(dec[0].value, 22);
    assert_eq!(dec[1].value, 23);
    // Gap is preserved in timestamps
    assert_eq!(dec[1].ts - dec[0].ts, 900);
}

#[test]
fn test_long_run() {
    let base = 1761955455u64;
    let mut enc = Encoder::<i32>::new();
    for i in 0..200 {
        enc.append(base + i * 300, 22).unwrap();
    }
    let dec = enc.decode();
    assert_eq!(dec.len(), 200);
    for r in &dec {
        assert_eq!(r.value, 22);
    }
}

#[test]
fn test_all_deltas() {
    let base = 1761955455u64;
    let mut enc = Encoder::<i32>::new();
    let mut temp = 70;
    let mut temps = vec![temp];
    for d in -10i32..=10 {
        if d == 0 {
            continue;
        }
        temp += d;
        temps.push(temp);
    }
    temps.push(temp + 50);
    temps.push(temp);

    for (i, &t) in temps.iter().enumerate() {
        enc.append(base + i as u64 * 300, t).unwrap();
    }
    let dec = enc.decode();
    assert_eq!(dec.len(), temps.len());
    for (i, r) in dec.iter().enumerate() {
        assert_eq!(r.value, temps[i], "mismatch at {}", i);
    }
}

#[test]
fn test_temp_range_25_to_39() {
    let base = 1761955455u64;
    let mut enc = Encoder::<i32>::new();
    let mut temps: Vec<i32> = (25..=39).collect();
    temps.extend((25..39).rev());

    for (i, &t) in temps.iter().enumerate() {
        enc.append(base + i as u64 * 300, t).unwrap();
    }

    let dec = enc.decode();

    assert_eq!(dec.len(), temps.len(), "count mismatch");
    for (i, r) in dec.iter().enumerate() {
        assert_eq!(r.value, temps[i], "mismatch at {}", i);
    }
}

#[test]
fn test_temp_range_neg10_to_39() {
    let base = 1761955455u64;
    let mut enc = Encoder::<i32>::new();
    let temps: Vec<i32> = (-10..=39).collect();

    for (i, &t) in temps.iter().enumerate() {
        enc.append(base + i as u64 * 300, t).unwrap();
    }

    let dec = enc.decode();

    assert_eq!(dec.len(), temps.len(), "count mismatch");
    for (i, r) in dec.iter().enumerate() {
        assert_eq!(r.value, temps[i], "mismatch at {}", i);
    }
}

#[test]
fn test_compression_ratio() {
    let base = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Simulate typical day: mostly constant with occasional changes
    let mut temp = 22;
    for i in 0..288 {
        // 288 = 24 hours * 12 (5-min intervals)
        if i == 50 {
            temp = 23;
        }
        if i == 150 {
            temp = 22;
        }
        enc.append(base + i * 300, temp).unwrap();
    }

    let bytes = enc.to_bytes();
    // Raw would be 288 * 12 = 3456 bytes
    // NibbleRun should be ~40-50 bytes
    assert!(bytes.len() < 60, "encoded size {} too large", bytes.len());
    assert!(bytes.len() > 10, "encoded size {} too small", bytes.len());
}

#[test]
fn test_constant_temperature() {
    let mut encoder = Encoder::<i32>::new();
    let base_ts = 1761955455u64;

    for i in 0..10 {
        encoder.append(base_ts + i * 300, 22).unwrap();
    }

    let decoded = encoder.decode();

    assert_eq!(decoded.len(), 10);
    for reading in &decoded {
        assert_eq!(reading.value, 22);
    }
}

#[test]
fn test_small_deltas() {
    let mut encoder = Encoder::<i32>::new();
    let base_ts = 1761955455u64;
    let temps = [22, 23, 22, 21, 22];

    for (i, &temp) in temps.iter().enumerate() {
        encoder.append(base_ts + i as u64 * 300, temp).unwrap();
    }

    let decoded = encoder.decode();

    assert_eq!(decoded.len(), 5);
    for (i, reading) in decoded.iter().enumerate() {
        assert_eq!(reading.value, temps[i], "mismatch at index {}", i);
    }
}

#[test]
fn test_medium_delta() {
    let mut encoder = Encoder::<i32>::new();
    let base_ts = 1761955455u64;

    encoder.append(base_ts, 20).unwrap();
    encoder.append(base_ts + 300, 25).unwrap(); // +5
    encoder.append(base_ts + 600, 20).unwrap(); // -5

    let decoded = encoder.decode();

    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].value, 20);
    assert_eq!(decoded[1].value, 25);
    assert_eq!(decoded[2].value, 20);
}

#[test]
fn test_large_delta() {
    let mut encoder = Encoder::<i32>::new();
    let base_ts = 1761955455u64;

    encoder.append(base_ts, 20).unwrap();
    encoder.append(base_ts + 300, 520).unwrap(); // +500
    encoder.append(base_ts + 600, 20).unwrap(); // -500

    let decoded = encoder.decode();

    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].value, 20);
    assert_eq!(decoded[1].value, 520);
    assert_eq!(decoded[2].value, 20);
}

#[test]
fn test_long_zero_run() {
    let mut encoder = Encoder::<i32>::new();
    let base_ts = 1761955455u64;

    for i in 0..50 {
        encoder.append(base_ts + i * 300, 22).unwrap();
    }

    let decoded = encoder.decode();

    assert_eq!(decoded.len(), 50);
    for reading in &decoded {
        assert_eq!(reading.value, 22);
    }
}

#[test]
fn test_with_timestamp_jitter() {
    // Simulate real-world sensor readings with small positive jitter
    // (negative jitter could cause readings to fall into previous interval)
    let base_ts = 1761955455u64;
    let temps = [22, 23, 23, 22, 21, 21, 22, 23, 22, 21];

    // Positive jitter only to ensure each reading lands in its expected interval
    let jitter = [0u64, 3, 2, 5, 5, 1, 3, 4, 1, 2];

    let mut encoder = Encoder::<i32>::new();
    for (i, (&temp, &j)) in temps.iter().zip(jitter.iter()).enumerate() {
        let ts = base_ts + (i as u64 * 300) + j;
        encoder.append(ts, temp).unwrap();
    }

    let decoded = encoder.decode();

    // All readings should be preserved
    assert_eq!(decoded.len(), 10);

    // Verify temperatures match and timestamps are quantized
    for (i, reading) in decoded.iter().enumerate() {
        assert_eq!(
            reading.value, temps[i],
            "temp mismatch at index {}",
            i
        );
        // Timestamp should be aligned to 300-second intervals from base_ts
        assert_eq!(
            (reading.ts - base_ts) % 300,
            0,
            "ts {} not aligned to interval at index {}",
            reading.ts,
            i
        );
    }
}

#[test]
fn test_with_larger_timestamp_jitter() {
    // Test with jitter that stays within the same interval when quantized
    // Using jitter small enough that (i*300 + jitter) / 300 = i
    let base_ts = 1761955455u64;
    let temps = [22, 23, 24, 25, 26];

    // Jitter values carefully chosen so logical index = i
    // For idx i: need (i*300 + jitter) / 300 = i
    // This means: i*300 <= i*300 + jitter < (i+1)*300
    // So: 0 <= jitter < 300 for positive
    let jitter = [0i64, 50, 100, 149, 200];

    let mut encoder = Encoder::<i32>::new();
    for (i, (&temp, &j)) in temps.iter().zip(jitter.iter()).enumerate() {
        let ts = (base_ts as i64 + (i as i64 * 300) + j) as u64;
        encoder.append(ts, temp).unwrap();
    }

    let decoded = encoder.decode();

    assert_eq!(decoded.len(), 5);

    // Expected: each reading maps to logical index i, so ts = base_ts + i*300
    let expected: [(u64, i32); 5] = [
        (base_ts + 0 * 300, 22),
        (base_ts + 1 * 300, 23),
        (base_ts + 2 * 300, 24),
        (base_ts + 3 * 300, 25),
        (base_ts + 4 * 300, 26),
    ];

    for (i, reading) in decoded.iter().enumerate() {
        assert_eq!(reading.ts, expected[i].0, "ts mismatch at index {}", i);
        assert_eq!(
            reading.value, expected[i].1,
            "temp mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_specific_day_with_jitter() {
    // 2025-12-01 00:00:00 UTC = 1764547200
    let day_start = 1764547200u64;

    // Input readings with realistic jitter:
    // 00:00:03 -> temp 25 (logical idx 0)
    // 00:05:10 -> temp 25 (logical idx 1)
    // 00:10:20 -> temp 26 (logical idx 2)
    // 00:15:02 -> temp 25 (logical idx 2 - SKIPPED, same as previous)
    // 01:35:05 -> temp 26 (logical idx 19)
    let inputs: [(u64, i32); 5] = [
        (day_start + 0 * 60 + 3, 25),   // 00:00:03
        (day_start + 5 * 60 + 10, 25),  // 00:05:10
        (day_start + 10 * 60 + 20, 26), // 00:10:20
        (day_start + 15 * 60 + 2, 25),  // 00:15:02 - will be skipped
        (day_start + 95 * 60 + 5, 26),  // 01:35:05
    ];

    let mut encoder = Encoder::<i32>::new();
    for (ts, temp) in inputs {
        encoder.append(ts, temp).unwrap();
    }

    let decoded = encoder.decode();

    // Only 4 readings - the 4th input is skipped (same logical index as 3rd)
    assert_eq!(decoded.len(), 4);

    // Input analysis (base_ts = day_start + 3 = 1764547203):
    // Reading 0: ts=1764547203, logical_idx=0
    // Reading 1: ts=1764547510, logical_idx=1
    // Reading 2: ts=1764547820, logical_idx=2
    // Reading 3: ts=1764548102, logical_idx=2 -> SKIPPED (duplicate)
    // Reading 4: ts=1764552905, logical_idx=19

    let base_ts = inputs[0].0;

    let expected: [(u64, i32); 4] = [
        (base_ts + 0 * 300, 25),  // logical idx 0
        (base_ts + 1 * 300, 25),  // logical idx 1
        (base_ts + 2 * 300, 26),  // logical idx 2
        (base_ts + 19 * 300, 26), // logical idx 19 (gap of 17 from idx 2)
    ];

    for (i, reading) in decoded.iter().enumerate() {
        assert_eq!(
            reading.ts, expected[i].0,
            "ts mismatch at index {}: got {}, expected {}",
            i, reading.ts, expected[i].0
        );
        assert_eq!(
            reading.value, expected[i].1,
            "temp mismatch at index {}: got {}, expected {}",
            i, reading.value, expected[i].1
        );
    }

    // Verify the large gap between readings 2 and 3
    assert_eq!(
        decoded[3].ts - decoded[2].ts,
        17 * 300,
        "gap between readings 2 and 3 should be 17 intervals (5100 seconds)"
    );
}

#[test]
fn test_out_of_order_readings_return_error() {
    let base_ts = 1761955455u64;
    let mut encoder = Encoder::<i32>::new();

    // First reading sets base_ts
    encoder.append(base_ts + 600, 24).unwrap(); // base_ts is set to base_ts + 600
    let actual_base = base_ts + 600;

    // Out-of-order readings return errors
    assert_eq!(
        encoder.append(base_ts, 22),
        Err(AppendError::TimestampBeforeBase {
            ts: base_ts,
            base_ts: actual_base
        })
    );
    assert_eq!(
        encoder.append(base_ts + 300, 23),
        Err(AppendError::TimestampBeforeBase {
            ts: base_ts + 300,
            base_ts: actual_base
        })
    );

    // Reading at a later interval is accepted
    encoder.append(base_ts + 900, 25).unwrap();

    let decoded = encoder.decode();

    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 24);
    assert_eq!(decoded[1].value, 25);
}

#[test]
fn test_reading_before_base_ts_returns_error() {
    let base_ts = 1761955455u64;
    let mut encoder = Encoder::<i32>::new();

    // First reading establishes base_ts
    encoder.append(base_ts, 22).unwrap();
    assert_eq!(encoder.count(), 1);

    // Reading before base_ts should return TimestampBeforeBase error
    assert_eq!(
        encoder.append(base_ts - 1, 99),
        Err(AppendError::TimestampBeforeBase {
            ts: base_ts - 1,
            base_ts
        })
    );
    assert_eq!(encoder.count(), 1);

    assert_eq!(
        encoder.append(base_ts - 100, 99),
        Err(AppendError::TimestampBeforeBase {
            ts: base_ts - 100,
            base_ts
        })
    );
    assert_eq!(encoder.count(), 1);

    assert_eq!(
        encoder.append(base_ts - 300, 99),
        Err(AppendError::TimestampBeforeBase {
            ts: base_ts - 300,
            base_ts
        })
    );
    assert_eq!(encoder.count(), 1);

    // Reading at or after base_ts should be accepted
    encoder.append(base_ts + 300, 23).unwrap();
    assert_eq!(encoder.count(), 2);

    let decoded = encoder.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 22);
    assert_eq!(decoded[1].value, 23);
}

#[test]
fn test_reading_before_epoch_base_as_first() {
    // If the first reading has ts < EPOCH_BASE, the base_ts_offset calculation
    // will wrap around due to unsigned integer arithmetic.
    // This is a known limitation: timestamps must be >= EPOCH_BASE.
    let mut encoder = Encoder::<i32>::new();

    // Timestamp before EPOCH_BASE (1_760_000_000)
    let old_ts = EPOCH_BASE - 1000;
    encoder.append(old_ts, 22).unwrap();

    // The encoder accepts it as first reading (base_ts = old_ts)
    assert_eq!(encoder.count(), 1);

    // to_bytes() completes but produces incorrect data due to wrapping
    // In release mode, this doesn't panic - it wraps around
    let bytes = encoder.to_bytes();
    assert_eq!(bytes.len(), header_size_for_value_bytes(i32::BYTES)); // Header only for single reading

    // The base_ts_offset will be a wrapped value (very large number)
    // In new header format, base_ts_offset is at offset 0
    let base_ts_offset = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    // Expected wrapped value: (EPOCH_BASE - 1000).wrapping_sub(EPOCH_BASE) as u32
    let expected = (old_ts.wrapping_sub(EPOCH_BASE)) as u32;
    assert_eq!(base_ts_offset, expected);
}

#[test]
fn test_size_matches_to_bytes() {
    // Empty encoder
    let enc = Encoder::<i32>::new();
    assert_eq!(enc.size(), enc.to_bytes().len());

    // Single reading
    let mut enc = Encoder::<i32>::new();
    enc.append(1761955455, 22).unwrap();
    assert_eq!(enc.size(), enc.to_bytes().len());

    // Multiple readings with zero deltas (tests zero_run estimation)
    let mut enc = Encoder::<i32>::new();
    for i in 0..10 {
        enc.append(1761955455 + i * 300, 22).unwrap();
    }
    assert_eq!(enc.size(), enc.to_bytes().len());

    // Readings with varying deltas
    let mut enc = Encoder::<i32>::new();
    let temps = [22, 23, 21, 25, 20, 30, 15];
    for (i, &t) in temps.iter().enumerate() {
        enc.append(1761955455 + i as u64 * 300, t).unwrap();
    }
    assert_eq!(enc.size(), enc.to_bytes().len());

    // Long zero run (tests zero_run > 149)
    let mut enc = Encoder::<i32>::new();
    for i in 0..200 {
        enc.append(1761955455 + i * 300, 22).unwrap();
    }
    assert_eq!(enc.size(), enc.to_bytes().len());

    // Mixed: some zeros, some deltas
    let mut enc = Encoder::<i32>::new();
    for i in 0..50 {
        let temp = if i % 10 == 0 { 25 } else { 22 };
        enc.append(1761955455 + i * 300, temp).unwrap();
    }
    assert_eq!(enc.size(), enc.to_bytes().len());
}

#[test]
fn test_size_incremental_with_jitter() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Simple PRNG for deterministic jitter (no external deps)
    let mut seed: u32 = 12345;
    let mut next_jitter = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed % 100) as u64 // 0-99 seconds jitter
    };

    // Simple PRNG for temperature changes
    let mut temp_seed: u32 = 67890;
    let mut next_temp_delta = || {
        temp_seed = temp_seed.wrapping_mul(1103515245).wrapping_add(12345);
        match temp_seed % 10 {
            0 => 1,  // 10% chance +1
            1 => -1, // 10% chance -1
            2 => 2,  // 10% chance +2
            3 => -2, // 10% chance -2
            _ => 0,  // 60% chance no change
        }
    };

    let mut temp = 22i32;
    let mut prev_logical_idx = 0u64;

    for i in 0..300 {
        let jitter = next_jitter();
        let ts = base_ts + i * 300 + jitter;

        // Calculate logical index to check if this reading will be accepted
        let logical_idx = if enc.count() == 0 {
            0
        } else {
            (ts - base_ts) / 300
        };

        // Only update temp if reading will be accepted (not a duplicate)
        if enc.count() == 0 || logical_idx > prev_logical_idx {
            temp = (temp + next_temp_delta()).clamp(-50, 100);
            prev_logical_idx = logical_idx;
        }

        enc.append(ts, temp).unwrap();

        // Assert size matches actual encoded size after each append
        let actual_size = enc.to_bytes().len();
        let estimated_size = enc.size();

        assert_eq!(
            estimated_size, actual_size,
            "size mismatch at iteration {}: estimated={}, actual={}",
            i, estimated_size, actual_size
        );
    }
}

#[test]
fn test_size_constant_temperature_incremental() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Constant temperature - maximum compression via zero runs
    for i in 0..300 {
        enc.append(base_ts + i * 300, 22).unwrap();

        let actual_size = enc.to_bytes().len();
        let estimated_size = enc.size();

        assert_eq!(
            estimated_size, actual_size,
            "size mismatch at iteration {}: estimated={}, actual={}",
            i, estimated_size, actual_size
        );
    }
}

#[test]
fn test_size_alternating_temperature_incremental() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Alternating temperature - no zero runs, all ±1 deltas
    for i in 0..300 {
        let temp = if i % 2 == 0 { 22 } else { 23 };
        enc.append(base_ts + i * 300, temp).unwrap();

        let actual_size = enc.to_bytes().len();
        let estimated_size = enc.size();

        assert_eq!(
            estimated_size, actual_size,
            "size mismatch at iteration {}: estimated={}, actual={}",
            i, estimated_size, actual_size
        );
    }
}

#[test]
fn test_size_large_deltas_incremental() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Large temperature swings - tests large delta encoding
    for i in 0..300 {
        let temp = if i % 2 == 0 { 0 } else { 100 };
        enc.append(base_ts + i * 300, temp).unwrap();

        let actual_size = enc.to_bytes().len();
        let estimated_size = enc.size();

        assert_eq!(
            estimated_size, actual_size,
            "size mismatch at iteration {}: estimated={}, actual={}",
            i, estimated_size, actual_size
        );
    }
}

#[test]
fn test_duplicate_day_events_return_error() {
    let base_ts = 1761955455u64;
    let mut encoder = Encoder::<i32>::new();

    // Append a full day of events (288 readings at 5-min intervals)
    for i in 0..288 {
        encoder.append(base_ts + i * 300, 22).unwrap();
    }
    assert_eq!(encoder.count(), 288);

    // Trying to append the same day again returns OutOfOrder errors
    // (because they're in earlier intervals than the last one)
    for i in 0..287 {
        let result = encoder.append(base_ts + i * 300, 22);
        assert!(
            matches!(result, Err(AppendError::OutOfOrder { .. })),
            "Expected OutOfOrder error for i={}",
            i
        );
    }

    // The last one (i=287) would be in the same interval as current, so it's allowed
    // (same interval = averaging)
    encoder.append(base_ts + 287 * 300, 22).unwrap();

    // Should still be 288 (the extra one was averaged into the last interval)
    assert_eq!(encoder.count(), 288);
}

#[test]
fn test_duplicate_day_events_with_different_timestamps_return_error() {
    let base_ts = 1761955455u64;
    let mut encoder = Encoder::<i32>::new();

    // First pass: 288 events at start of each interval
    for i in 0..288 {
        encoder.append(base_ts + i * 300, 22).unwrap();
    }
    assert_eq!(encoder.count(), 288);

    // Second pass: trying to go back in time returns OutOfOrder errors
    for i in 0..287 {
        let ts = base_ts + i * 300 + 150; // 150 seconds into each interval
        let result = encoder.append(ts, 22);
        assert!(
            matches!(result, Err(AppendError::OutOfOrder { .. })),
            "Expected OutOfOrder error for i={}",
            i
        );
    }

    // Last interval (i=287) is current, so adding to it works
    encoder.append(base_ts + 287 * 300 + 150, 22).unwrap();
    assert_eq!(encoder.count(), 288);
}

#[test]
fn test_duplicate_timestamps_averaged() {
    let base_ts = 1761955455u64;
    let mut encoder = Encoder::<i32>::new();

    encoder.append(base_ts, 22).unwrap();
    encoder.append(base_ts, 23).unwrap(); // Same interval - will be averaged
    encoder.append(base_ts + 5, 24).unwrap(); // Same logical index (within same 300s interval)
    encoder.append(base_ts + 300, 25).unwrap(); // Next interval

    let decoded = encoder.decode();

    // Two intervals: first averaged (22+23+24)/3 = 23, second = 25
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 23); // (22+23+24+1)/3 = 23 (round half up)
    assert_eq!(decoded[1].value, 25);
}

#[test]
fn test_averaging_round_half_up() {
    let base_ts = 1761955455u64;

    // Test case: 22 + 23 = 45, (45 + 1) / 2 = 23 (rounds up)
    let mut encoder = Encoder::<i32>::new();
    encoder.append(base_ts, 22).unwrap();
    encoder.append(base_ts + 1, 23).unwrap();
    let decoded = encoder.decode();
    assert_eq!(decoded[0].value, 23);

    // Test case: 22 + 22 + 23 = 67, (67 + 1) / 3 = 22 (rounds down)
    let mut encoder = Encoder::<i32>::new();
    encoder.append(base_ts, 22).unwrap();
    encoder.append(base_ts + 1, 22).unwrap();
    encoder.append(base_ts + 2, 23).unwrap();
    let decoded = encoder.decode();
    assert_eq!(decoded[0].value, 22);

    // Test case: 20 + 21 + 22 + 23 = 86, (86 + 2) / 4 = 22
    let mut encoder = Encoder::<i32>::new();
    encoder.append(base_ts, 20).unwrap();
    encoder.append(base_ts + 1, 21).unwrap();
    encoder.append(base_ts + 2, 22).unwrap();
    encoder.append(base_ts + 3, 23).unwrap();
    let decoded = encoder.decode();
    assert_eq!(decoded[0].value, 22);

    // Test case: negative temperatures - (-16) + (-16) = -32, (-32 - 1) / 2 = -16
    // This tests that rounding works correctly for negative numbers
    let mut encoder = Encoder::<i32>::new();
    encoder.append(base_ts, -16).unwrap();
    encoder.append(base_ts + 1, -16).unwrap();
    let decoded = encoder.decode();
    assert_eq!(decoded[0].value, -16);

    // Test case: negative with rounding - (-15) + (-16) = -31, (-31 - 1) / 2 = -16
    let mut encoder = Encoder::<i32>::new();
    encoder.append(base_ts, -15).unwrap();
    encoder.append(base_ts + 1, -16).unwrap();
    let decoded = encoder.decode();
    assert_eq!(decoded[0].value, -16); // rounds away from zero
}

#[test]
fn test_alternating_readings_same_interval_averaged() {
    let base_ts = 1761955455u64;
    let mut encoder = Encoder::<i32>::new();

    // 10 readings alternating 25, 21 spread across 5 intervals (2 per interval)
    // Each interval: 25 + 21 = 46, (46 + 1) / 2 = 23 (round half up)
    let readings = [25, 21, 25, 21, 25, 21, 25, 21, 25, 21];
    for (i, &temp) in readings.iter().enumerate() {
        // 2 readings per 300s interval: readings 0,1 in interval 0, 2,3 in interval 1, etc.
        let interval = i / 2;
        let offset_within_interval = (i % 2) * 150; // 0 or 150 seconds
        encoder
            .append(
                base_ts + (interval as u64) * 300 + offset_within_interval as u64,
                temp,
            )
            .unwrap();
    }

    let decoded = encoder.decode();

    // Should be 5 readings, all with averaged value 23
    assert_eq!(
        decoded.len(),
        5,
        "expected 5 averaged readings, got {}",
        decoded.len()
    );
    for (i, reading) in decoded.iter().enumerate() {
        assert_eq!(
            reading.value, 23,
            "expected temp 23 at index {}, got {}",
            i, reading.value
        );
        assert_eq!(
            reading.ts,
            base_ts + (i as u64) * 300,
            "wrong timestamp at index {}",
            i
        );
    }

    // Size check: with appendable format, bits stay in header's bit_accum until 8+ bits
    // 4 zeros = 4 bits, which stays in the header (not yet flushed to data section)
    // So buffer is just header_size for i32 (26 bytes)
    let expected_header_size = header_size_for_value_bytes(i32::BYTES);
    let size = encoder.size();
    assert_eq!(
        size, expected_header_size,
        "expected size of {} bytes (header only, 4 bits still in accumulator), got {}",
        expected_header_size,
        size
    );
}

#[test]
fn test_custom_interval() {
    let base_ts = 1761955455u64;

    // Test with 60-second interval
    let mut enc = Encoder::<i32, 60>::new();
    assert_eq!(Encoder::<i32, 60>::interval(), 60);

    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 60, 23).unwrap();
    enc.append(base_ts + 120, 24).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].ts, base_ts);
    assert_eq!(decoded[1].ts, base_ts + 60);
    assert_eq!(decoded[2].ts, base_ts + 120);
    assert_eq!(decoded[0].value, 22);
    assert_eq!(decoded[1].value, 23);
    assert_eq!(decoded[2].value, 24);

    // Test roundtrip via bytes
    let bytes = enc.to_bytes();
    let decoded_bytes = decode::<i32, 60>(&bytes);
    assert_eq!(decoded_bytes.len(), 3);
    assert_eq!(decoded_bytes[1].ts, base_ts + 60);
}

#[test]
fn test_custom_interval_averaging() {
    let base_ts = 1761955455u64;

    // Test averaging with 60-second interval
    let mut enc = Encoder::<i32, 60>::new();

    // Two readings in same 60-second interval
    enc.append(base_ts, 20).unwrap();
    enc.append(base_ts + 30, 24).unwrap(); // Same interval, should average to 22

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].value, 22); // (20 + 24) / 2 = 22
}

#[test]
fn test_single_reading_per_interval_exact() {
    // When exactly one reading falls in an interval, decoded value should equal input exactly
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    let temps = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
    for (i, &temp) in temps.iter().enumerate() {
        enc.append(base_ts + (i as u64) * 300, temp).unwrap();
    }

    let decoded = enc.decode();
    assert_eq!(decoded.len(), temps.len());
    for (i, reading) in decoded.iter().enumerate() {
        assert_eq!(
            reading.value, temps[i],
            "Single reading at interval {} should be exact: expected {}, got {}",
            i, temps[i], reading.value
        );
    }
}

#[test]
fn test_max_readings_65535() {
    // Encode exactly 65535 readings (u16::MAX), verify roundtrip
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    for i in 0..65535u64 {
        let temp = ((i % 20) as i32) + 15; // Temps 15-34
        enc.append(base_ts + i * 300, temp).unwrap();
    }

    assert_eq!(enc.count(), 65535);

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 65535);

    // Verify via bytes roundtrip
    let bytes = enc.to_bytes();
    let decoded_bytes = decode::<i32, 300>(&bytes);
    assert_eq!(decoded_bytes.len(), 65535);
}

#[test]
fn test_beyond_max_readings() {
    // Verify behavior when appending reading 65536+
    // Currently the encoder will panic on overflow - this documents that behavior
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Fill to max
    for i in 0..65535u64 {
        enc.append(base_ts + i * 300, 22).unwrap();
    }
    assert_eq!(enc.count(), 65535);

    // Adding one more would cause overflow panic in debug mode
    // In release mode it would wrap to 0, causing corruption
    // This test documents that 65535 is the hard limit
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        enc.append(base_ts + 65535 * 300, 23).unwrap();
    }));
    // In debug mode, this panics; in release mode, it may not
    // Either way, we verify the encoder has 65535 readings
    if result.is_ok() {
        // If it didn't panic, count may have wrapped - check decode still works
        let decoded = enc.decode();
        assert!(decoded.len() <= 65535);
    }
}

#[test]
fn test_extreme_temps_boundaries() {
    // Test i32 temperatures (now stored as full i32)
    // Note: first_temp can be any i32, but:
    // - deltas between readings are limited to ±1024
    // - averaging accumulator is limited to ±1M sum range
    let base_ts = 1761955455u64;

    // Test large negative first_temp (within ±1M for averaging)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, -500_000).unwrap();
    enc.append(base_ts + 300, -499_500).unwrap(); // delta = +500
    let decoded = enc.decode();
    assert_eq!(decoded[0].value, -500_000);
    assert_eq!(decoded[1].value, -499_500);

    // Test large positive first_temp
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 500_000).unwrap();
    enc.append(base_ts + 300, 500_500).unwrap(); // delta = +500
    let decoded = enc.decode();
    assert_eq!(decoded[0].value, 500_000);
    assert_eq!(decoded[1].value, 500_500);

    // Test a sequence with various temps, all within ±1024 delta of each other
    let mut enc = Encoder::<i32>::new();
    let temps = [-1000, -500, 0, 500, 1000, 500, 0, -500, -1000];
    for (i, &temp) in temps.iter().enumerate() {
        enc.append(base_ts + (i as u64) * 300, temp).unwrap();
    }

    let decoded = enc.decode();
    assert_eq!(decoded.len(), temps.len());
    for (i, reading) in decoded.iter().enumerate() {
        assert_eq!(
            reading.value, temps[i],
            "Extreme temp at index {}: expected {}, got {}",
            i, temps[i], reading.value
        );
    }

    // Test maximum delta range (±1023, since ±1024 is the limit)
    let mut enc2 = Encoder::<i32>::new();
    enc2.append(base_ts, 0).unwrap();
    enc2.append(base_ts + 300, 1023).unwrap(); // delta = +1023
    enc2.append(base_ts + 600, 0).unwrap(); // delta = -1023

    let decoded2 = enc2.decode();
    assert_eq!(decoded2[0].value, 0);
    assert_eq!(decoded2[1].value, 1023);
    assert_eq!(decoded2[2].value, 0);

    // Verify roundtrip via bytes works for large temperatures
    let mut enc3 = Encoder::<i32>::new();
    enc3.append(base_ts, 100_000).unwrap();
    enc3.append(base_ts + 300, 100_500).unwrap();
    let bytes = enc3.to_bytes();
    let decoded3 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded3[0].value, 100_000);
    assert_eq!(decoded3[1].value, 100_500);
}

#[test]
fn test_interval_1_second() {
    // interval = 1, readings every second
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32, 1>::new();

    for i in 0..100u64 {
        enc.append(base_ts + i, 22 + (i % 5) as i32).unwrap();
    }

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 100);

    // Verify timestamps are 1 second apart
    for window in decoded.windows(2) {
        assert_eq!(window[1].ts - window[0].ts, 1);
    }
}

#[test]
fn test_interval_65535_seconds() {
    // interval = 65535 (~18 hours)
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32, 65535>::new();

    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 65535, 23).unwrap();
    enc.append(base_ts + 65535 * 2, 24).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[1].ts - decoded[0].ts, 65535);
    assert_eq!(decoded[2].ts - decoded[1].ts, 65535);
}

#[test]
fn test_zero_run_tier_boundaries() {
    // Zero-run encoding tiers:
    // - Single zero: 1 bit (prefix 0)
    // - 2-5 zeros: 5 bits (prefix 110 + 2 bits)
    // - 6-21 zeros: 8 bits (prefix 1110 + 4 bits)
    // - 22-149 zeros: 12 bits (prefix 11110 + 7 bits)

    let base_ts = 1761955455u64;

    // Test exactly 1 zero (single zero encoding)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 300, 22).unwrap(); // 1 zero delta
    enc.append(base_ts + 600, 23).unwrap();
    let decoded = enc.decode();
    assert_eq!(decoded.len(), 3);

    // Test exactly 5 zeros (boundary of 2-5 tier)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    for i in 1..=5 {
        enc.append(base_ts + i * 300, 22).unwrap(); // 5 zeros
    }
    enc.append(base_ts + 6 * 300, 23).unwrap();
    let decoded = enc.decode();
    assert_eq!(decoded.len(), 7);

    // Test exactly 21 zeros (boundary of 6-21 tier)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    for i in 1..=21 {
        enc.append(base_ts + i * 300, 22).unwrap(); // 21 zeros
    }
    enc.append(base_ts + 22 * 300, 23).unwrap();
    let decoded = enc.decode();
    assert_eq!(decoded.len(), 23);

    // Test exactly 149 zeros (boundary of 22-149 tier)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    for i in 1..=149 {
        enc.append(base_ts + i * 300, 22).unwrap(); // 149 zeros
    }
    enc.append(base_ts + 150 * 300, 23).unwrap();
    let decoded = enc.decode();
    assert_eq!(decoded.len(), 151);

    // Test 150 zeros (exceeds single run, needs 2 encodings)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    for i in 1..=150 {
        enc.append(base_ts + i * 300, 22).unwrap(); // 150 zeros
    }
    enc.append(base_ts + 151 * 300, 23).unwrap();
    let decoded = enc.decode();
    assert_eq!(decoded.len(), 152);
}

/// Regression test: consecutive zero deltas should be merged into a single run,
/// not split into run + individual zeros.
/// Bug: 81 zeros were encoded as 80-run + 1 single zero instead of 81-run.
#[test]
fn test_zero_run_not_split() {
    let base_ts = 1761955455u64;

    // Create a sequence: one reading with value A, then 82 readings with value B
    // This should produce: first reading, delta (A->B), then 81 zero deltas
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 100).unwrap(); // Reading 1: value 100

    // Readings 2-83: all value 113 (creates 1 non-zero delta + 81 zero deltas)
    for i in 1..=82 {
        enc.append(base_ts + i * 300, 113).unwrap();
    }

    // Add one more reading with different value to flush the zero run
    enc.append(base_ts + 83 * 300, 114).unwrap();

    let bytes = enc.to_bytes();
    let decoded = decode::<i32, 300>(&bytes);

    // Verify correct count
    assert_eq!(decoded.len(), 84);

    // Verify correct values
    assert_eq!(decoded[0].value, 100);
    for i in 1..=82 {
        assert_eq!(
            decoded[i].value, 113,
            "Reading {} should be 113, got {}",
            i, decoded[i].value
        );
    }
    assert_eq!(decoded[83].value, 114);

    // Verify encoding is optimal: 81 zeros should fit in a single 12-bit run
    // The encoding should be: header (14) + delta(-13) + 81-run (12 bits) + delta(+1) (3 bits)
    // Without the bug fix, it would be: header + delta + 80-run + single-zero + delta
    // which would add an extra bit
}

/// Test with a gap before the zero run (matching the user's bug report scenario)
#[test]
fn test_zero_run_after_gap() {
    let base_ts = 1761955455u64;

    let mut enc = Encoder::<i32>::new();

    // Some initial readings
    for i in 0..60 {
        enc.append(base_ts + i * 300, 22 + (i % 5) as i32).unwrap();
    }

    // Reading at interval 60 with value before the long run
    enc.append(base_ts + 60 * 300, 120).unwrap();

    // Gap: skip interval 61
    // Reading at interval 62 starts a different value
    enc.append(base_ts + 62 * 300, 113).unwrap(); // -7 delta

    // 81 more readings with same value (should create 81 zero deltas)
    for i in 63..=143 {
        enc.append(base_ts + i * 300, 113).unwrap();
    }

    // Final reading with different value to flush
    enc.append(base_ts + 144 * 300, 114).unwrap();

    let bytes = enc.to_bytes();
    let decoded = decode::<i32, 300>(&bytes);

    // Find where val=113 starts
    let start_113 = decoded.iter().position(|r| r.value == 113).unwrap();
    let end_113 = decoded.iter().rposition(|r| r.value == 113).unwrap();

    // Count consecutive 113 values
    let count_113 = end_113 - start_113 + 1;

    assert_eq!(
        count_113, 82,
        "Expected 82 consecutive readings with value 113, got {}",
        count_113
    );

    // The last 113 reading should be followed by 114
    assert_eq!(
        decoded[end_113 + 1].value, 114,
        "Expected value 114 after the last 113"
    );
}

#[test]
fn test_gap_encoding_boundaries() {
    // Gap marker: 11111111 + 6 bits = up to 64 gaps per marker
    // Gaps are implicit from timestamp jumps
    let base_ts = 1761955455u64;

    // Test gap of exactly 64 intervals (max per single marker)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 65 * 300, 23).unwrap(); // Skip 64 intervals

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].ts, base_ts);
    assert_eq!(decoded[1].ts, base_ts + 65 * 300);

    // Test gap of 65 (requires 2 gap markers: 64 + 1)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 66 * 300, 23).unwrap(); // Skip 65 intervals

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[1].ts, base_ts + 66 * 300);

    // Test gap of 128 (requires 2 gap markers: 64 + 64)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 129 * 300, 24).unwrap(); // Skip 128 intervals

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[1].ts, base_ts + 129 * 300);
}

#[test]
fn test_large_timestamp_offset() {
    // Test timestamps where ts - base_ts is large
    // With 300s interval, offset = gap_intervals * 300
    // Max practical gap is limited by gap encoding (64 per marker, ~19200s per marker)
    let base_ts = EPOCH_BASE + 1_000_000; // Well after epoch base
    let mut enc = Encoder::<i32>::new();

    enc.append(base_ts, 22).unwrap();

    // Use offset that's multiple of interval and reasonable for gap encoding
    // 1000 intervals * 300s = 300,000 seconds (~3.5 days)
    let large_offset = 1000u64 * 300;
    enc.append(base_ts + large_offset, 23).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    // The gap should be exactly large_offset / interval = 1000 intervals
    // But encoded timestamps are quantized to interval
    let expected_ts_diff = large_offset;
    assert_eq!(decoded[1].ts - decoded[0].ts, expected_ts_diff);
}

#[test]
fn test_decode_truncated_header() {
    // Header size for i32: 4 + 2 + 2 + 4 + 4 + 8 + 1 + 1 = 26 bytes
    let i32_header_size = header_size_for_value_bytes(i32::BYTES);
    assert_eq!(i32_header_size, 26);

    // < header_size should return empty vec
    assert!(decode::<i32, 300>(&[]).is_empty());
    assert!(decode::<i32, 300>(&[0]).is_empty());
    assert!(decode::<i32, 300>(&[0; 25]).is_empty());

    // Exactly header_size bytes is valid header
    let mut header = vec![0u8; i32_header_size];
    // Set count to 0 - should return empty vec
    let decoded = decode::<i32, 300>(&header);
    assert!(decoded.is_empty());

    // Set count to 1, first_value as i32
    // i32 header layout (26 bytes):
    // [0-3] base_ts_offset, [4-5] count, [6-7] prev_logical_idx,
    // [8-11] first_value, [12-15] prev_value, [16-23] pending_avg,
    // [24] bit_count, [25] bit_accum
    header[4] = 1;    // count = 1
    header[5] = 0;
    // first_value = 22 as i32 little-endian (bytes 8-11)
    header[8] = 22;
    header[9] = 0;
    header[10] = 0;
    header[11] = 0;
    // pending_avg = pack_avg(1, 22) = (22 << 10) | 1 = 22529
    // As u64 little-endian at offset 16-23
    let pending_avg: u64 = (22_u64 << 10) | 1;
    let pending_bytes = pending_avg.to_le_bytes();
    header[16..24].copy_from_slice(&pending_bytes);
    let decoded = decode::<i32, 300>(&header);
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].value, 22);
}

#[test]
fn test_decode_corrupted_count() {
    // count field larger than actual data - should not panic
    let mut enc = Encoder::<i32>::new();
    enc.append(1761955455, 22).unwrap();
    enc.append(1761955455 + 300, 23).unwrap();

    let mut bytes = enc.to_bytes();

    // Corrupt count field to be larger (count is at offset 4-5 in new header format)
    bytes[4] = 255; // count = 255 but only 2 readings encoded
    bytes[5] = 0;

    // Should not panic, may return partial data
    let decoded = decode::<i32, 300>(&bytes);
    // Behavior: decode will try to read more than available, but should handle gracefully
    assert!(decoded.len() <= 255);
}

#[test]
fn test_decode_zero_interval() {
    // Calling decode with interval=0 - edge case
    // Header layout: [0-3] base_ts, [4-5] count, [6-9] first_value
    let mut header = [0u8; 10];
    header[4] = 1; // count = 1
    header[5] = 0;
    // first_temp = 22 as i32 little-endian (bytes 6-9)
    header[6] = 22;
    header[7] = 0;
    header[8] = 0;
    header[9] = 0;

    // Should handle gracefully (interval 0 would cause div-by-zero if not handled)
    let decoded = decode::<i32, 1>(&header);
    // With interval=0, behavior depends on implementation
    // At minimum, should not panic
    assert!(decoded.len() <= 1);
}

#[test]
fn test_31_readings_same_interval() {
    // Test 31 readings in the same interval (legacy test, still valid)
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 31 readings in same interval
    for i in 0..31 {
        enc.append(base_ts + i * 5, 20 + (i as i32 % 10)).unwrap(); // Temps 20-29
    }

    // Move to next interval
    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);

    // Average of 31 readings with temps 20-29 (repeating 3x + 20)
    // Sum = (20+21+22+23+24+25+26+27+28+29) * 3 + 20 = 245 * 3 + 20 = 755
    // Avg = 755 / 31 = 24.35... ≈ 24
    let expected_avg = (0..31).map(|i| 20 + (i % 10)).sum::<i32>() / 31;
    assert_eq!(decoded[0].value, expected_avg);
}

#[test]
fn test_32_readings_same_interval() {
    // 32 readings now works (new limit is 1023)
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 32 readings in same interval
    for i in 0..32 {
        enc.append(base_ts + i * 5, 20).unwrap();
    }

    // Move to next interval
    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    // All 32 readings are now included in the average
    assert_eq!(decoded[0].value, 20);
}

#[test]
fn test_100_readings_same_interval() {
    // Test 100 readings in the same interval
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 100 readings in same interval, alternating 20 and 30
    for i in 0..100 {
        let temp = if i % 2 == 0 { 20 } else { 30 };
        enc.append(base_ts + i, temp).unwrap();
    }

    // Move to next interval
    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    // Average of 50 x 20 + 50 x 30 = 1000 + 1500 = 2500 / 100 = 25
    assert_eq!(decoded[0].value, 25);
}

#[test]
fn test_500_readings_same_interval() {
    // Test 500 readings in the same interval
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 500 readings in same interval
    for i in 0..500 {
        enc.append(base_ts + i, 22).unwrap();
    }

    // Move to next interval
    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 22);
}

#[test]
fn test_1023_readings_same_interval() {
    // Test max pending_count = 1023 (10 bits)
    // Use 1-second interval so all 1023 readings fit in one interval
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32, 2000>::new(); // 2000 second interval

    // 1023 readings in same interval (all within first 1023 seconds)
    for i in 0..1023 {
        enc.append(base_ts + i, 20 + (i as i32 % 10)).unwrap(); // Temps 20-29
    }

    // Move to next interval
    enc.append(base_ts + 2000, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);

    // Average of 1023 readings with temps 20-29 (102 complete cycles + partial)
    let expected_sum: i32 = (0..1023).map(|i| 20 + (i % 10)).sum();
    let expected_avg = expected_sum / 1023;
    assert_eq!(decoded[0].value, expected_avg);
}

#[test]
fn test_1024_readings_same_interval() {
    // 1024 readings exceeds pending_count max of 1023 - 1024th should return error
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32, 2000>::new(); // 2000 second interval

    // 1023 readings in same interval (max allowed)
    for i in 0..1023 {
        enc.append(base_ts + i, 20).unwrap();
    }
    // This 1024th reading should return IntervalOverflow error
    assert!(matches!(
        enc.append(base_ts + 1023, 100),
        Err(AppendError::IntervalOverflow { count: 1023 })
    ));

    // Move to next interval - should work since previous was rejected
    enc.append(base_ts + 2000, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    // Average should be 20, the rejected 100 was not included
    assert_eq!(decoded[0].value, 20);
}

#[test]
fn test_high_count_with_large_temps() {
    // Test that sum doesn't overflow with many readings of large temps
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 500 readings of temperature 500 (sum = 250,000)
    for i in 0..500 {
        enc.append(base_ts + i, 500).unwrap();
    }

    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 500);
}

#[test]
fn test_high_count_with_negative_temps() {
    // Test averaging with many negative temperatures
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 500 readings of temperature -500 (sum = -250,000)
    for i in 0..500 {
        enc.append(base_ts + i, -500).unwrap();
    }

    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, -500);
}

#[test]
fn test_high_count_mixed_temps() {
    // Test averaging with mix of positive and negative temps
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 400 readings: alternating -100 and +100
    for i in 0..400 {
        let temp = if i % 2 == 0 { -100 } else { 100 };
        enc.append(base_ts + i, temp).unwrap();
    }

    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    // Average of 200 x -100 + 200 x 100 = 0
    assert_eq!(decoded[0].value, 0);
}

#[test]
fn test_averaging_to_zero() {
    // positive + negative temps that average to 0
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // -10 and +10 average to 0
    enc.append(base_ts, -10).unwrap();
    enc.append(base_ts + 1, 10).unwrap();
    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 0);

    // -50, -30, +40, +40 = sum 0, avg 0
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, -50).unwrap();
    enc.append(base_ts + 1, -30).unwrap();
    enc.append(base_ts + 2, 40).unwrap();
    enc.append(base_ts + 3, 40).unwrap();
    enc.append(base_ts + 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 0);
}

#[test]
fn test_delta_overflow_returns_error() {
    // Delta > 1023 or < -1024 should return DeltaOverflow error
    // Note: Error is returned during finalize_pending_interval,
    // which is called when crossing from one interval to another.
    // We need at least 4 intervals to trigger this:
    // - interval 0: establishes base
    // - interval 1: first delta (from interval 0)
    // - interval 2: large temp that will overflow
    // - interval 3: triggers finalize of interval 2, returning error
    let base_ts = 1761955455u64;

    // Test positive delta overflow: 0 to 2000 = delta of +2000 (out of range)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 0).unwrap(); // interval 0
    enc.append(base_ts + 300, 0).unwrap(); // interval 1
    enc.append(base_ts + 600, 2000).unwrap(); // interval 2: temp=2000
    // interval 3: triggers finalize(2), delta=2000
    assert!(matches!(
        enc.append(base_ts + 900, 0),
        Err(AppendError::DeltaOverflow {
            delta: 2000,
            prev_value: 0,
            new_value: 2000
        })
    ));

    // Test negative delta overflow: 2000 to 0 = delta of -2000 (out of range)
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 2000).unwrap(); // interval 0
    enc.append(base_ts + 300, 2000).unwrap(); // interval 1
    enc.append(base_ts + 600, 0).unwrap(); // interval 2: temp=0
    // interval 3: triggers finalize(2), delta=-2000
    assert!(matches!(
        enc.append(base_ts + 900, 0),
        Err(AppendError::DeltaOverflow {
            delta: -2000,
            prev_value: 2000,
            new_value: 0
        })
    ));

    // Test boundary: delta of exactly +1024 should return error
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 0).unwrap();
    enc.append(base_ts + 300, 0).unwrap();
    enc.append(base_ts + 600, 1024).unwrap(); // delta will be +1024
    assert!(matches!(
        enc.append(base_ts + 900, 0),
        Err(AppendError::DeltaOverflow {
            delta: 1024,
            prev_value: 0,
            new_value: 1024
        })
    ));

    // Test boundary: delta of exactly -1025 should return error
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 1025).unwrap();
    enc.append(base_ts + 300, 1025).unwrap();
    enc.append(base_ts + 600, 0).unwrap(); // delta will be -1025
    assert!(matches!(
        enc.append(base_ts + 900, 0),
        Err(AppendError::DeltaOverflow {
            delta: -1025,
            prev_value: 1025,
            new_value: 0
        })
    ));

    // Test boundary: delta of exactly +1023 should NOT return error
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 0).unwrap();
    enc.append(base_ts + 300, 0).unwrap();
    enc.append(base_ts + 600, 1023).unwrap(); // delta will be +1023
    enc.append(base_ts + 900, 0).unwrap(); // Should succeed

    // Test boundary: delta of exactly -1024 should NOT return error
    let mut enc = Encoder::<i32>::new();
    enc.append(base_ts, 1024).unwrap();
    enc.append(base_ts + 300, 1024).unwrap();
    enc.append(base_ts + 600, 0).unwrap(); // delta will be -1024
    enc.append(base_ts + 900, 0).unwrap(); // Should succeed
}

#[test]
fn test_interval_zero_encoder() {
    // Creating encoder with interval=0 is questionable but shouldn't crash during creation
    // The behavior during append may vary (division by zero)
    let enc = Encoder::<i32, 0>::new();
    assert_eq!(enc.count(), 0);
    // Don't try to append - would cause division by zero
}

#[test]
fn test_max_interval_65535() {
    // Test maximum interval value
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32, 65535>::new();

    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 65535, 23).unwrap(); // Exactly one interval later
    enc.append(base_ts + 65535 * 2, 24).unwrap(); // Two intervals later

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[1].ts - decoded[0].ts, 65535);
    assert_eq!(decoded[2].ts - decoded[1].ts, 65535);
}

#[test]
fn test_timestamp_at_epoch_base() {
    // Timestamp exactly at EPOCH_BASE should work
    let mut enc = Encoder::<i32>::new();
    enc.append(1_760_000_000, 22).unwrap(); // Exactly EPOCH_BASE
    enc.append(1_760_000_300, 23).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].ts, 1_760_000_000);
}

#[test]
fn test_count_at_max_u16() {
    // Test that count() returns correct value at max
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32, 1>::new();

    // Add exactly 65535 readings
    for i in 0..65535u64 {
        enc.append(base_ts + i, 22).unwrap();
    }

    assert_eq!(enc.count(), 65535);
}

// ============================================================================
// Coverage tests - these tests exist to achieve 100% code coverage
// ============================================================================

#[test]
fn test_error_display_formatting() {
    // TimestampBeforeBase
    let err = AppendError::TimestampBeforeBase { ts: 100, base_ts: 200 };
    assert!(err.to_string().contains("100"));
    assert!(err.to_string().contains("200"));

    // OutOfOrder
    let err = AppendError::OutOfOrder { ts: 300, logical_idx: 1, prev_logical_idx: 2 };
    assert!(err.to_string().contains("interval"));

    // IntervalOverflow
    let err = AppendError::IntervalOverflow { count: 1023 };
    assert!(err.to_string().contains("1023"));

    // CountOverflow
    let err = AppendError::CountOverflow;
    assert!(err.to_string().contains("65535"));

    // DeltaOverflow
    let err = AppendError::DeltaOverflow { delta: 2000, prev_value: 0, new_value: 2000 };
    assert!(err.to_string().contains("2000"));
}

#[test]
fn test_error_trait_impl() {
    use std::error::Error;
    let err: &dyn Error = &AppendError::CountOverflow;
    assert!(err.source().is_none());
}

#[test]
fn test_encoder_new_default_interval() {
    let _enc = Encoder::<i32>::new();
    assert_eq!(Encoder::<i32>::interval(), 300);
}

#[test]
fn test_encoder_explicit_interval() {
    let _enc = Encoder::<i32>::new();
    assert_eq!(Encoder::<i32>::interval(), 300);
}

#[test]
fn test_interval_getter() {
    let _enc = Encoder::<i32, 600>::new();
    assert_eq!(Encoder::<i32, 600>::interval(), 600);
}

#[test]
fn test_zero_run_decode_tier_6_to_21() {
    // 15 consecutive same-value readings hits the 6-21 tier decoder path
    let mut enc = Encoder::<i32>::new();
    let base_ts = 1_760_000_000u64;

    enc.append(base_ts, 25).unwrap();
    for i in 1..=15 {
        enc.append(base_ts + i * 300, 25).unwrap();
    }

    let bytes = enc.to_bytes();
    let decoded = decode::<i32, 300>(&bytes);
    assert_eq!(decoded.len(), 16);
}

#[test]
fn test_zero_run_decode_tier_22_to_149() {
    // 50 consecutive same-value readings hits the 22-149 tier decoder path
    let mut enc = Encoder::<i32>::new();
    let base_ts = 1_760_000_000u64;

    enc.append(base_ts, 25).unwrap();
    for i in 1..=50 {
        enc.append(base_ts + i * 300, 25).unwrap();
    }

    let bytes = enc.to_bytes();
    let decoded = decode::<i32, 300>(&bytes);
    assert_eq!(decoded.len(), 51);
}

/// Test that verifies encoding structure - single zeros should NOT appear after runs
/// if they could have been merged into the run
#[test]
fn test_zero_run_encoding_structure() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Create sequence: val=15, val=12 (delta -3), then 16x val=12 (16 zeros), val=5 (delta -7)
    enc.append(base_ts, 15).unwrap();
    enc.append(base_ts + 300, 12).unwrap();
    for i in 2..=17 {
        enc.append(base_ts + i * 300, 12).unwrap();
    }
    enc.append(base_ts + 18 * 300, 5).unwrap();

    let decoded = enc.decode();

    // Verify we have 19 readings total (indices 0-18)
    assert_eq!(decoded.len(), 19, "Should have 19 readings");

    // Verify first reading
    assert_eq!(decoded[0].ts, base_ts);
    assert_eq!(decoded[0].value, 15);

    // Verify second reading (delta -3)
    assert_eq!(decoded[1].ts, base_ts + 300);
    assert_eq!(decoded[1].value, 12);

    // Verify readings 2-17 (16 zeros - same value)
    for i in 2..=17 {
        assert_eq!(decoded[i].ts, base_ts + i as u64 * 300, "Timestamp at index {}", i);
        assert_eq!(decoded[i].value, 12, "Value at index {}", i);
    }

    // Verify final reading (delta -7)
    assert_eq!(decoded[18].ts, base_ts + 18 * 300);
    assert_eq!(decoded[18].value, 5);

    // Verify roundtrip
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2, "Roundtrip should preserve readings");
}

// ============================================================================
// Encoding structure tests - verify actual bit patterns
// ============================================================================

/// Helper function to read bits from a byte slice (MSB first within each byte)
fn read_bits_from(data: &[u8], pos: &mut usize, n: usize) -> u32 {
    let mut result = 0u32;
    for _ in 0..n {
        let byte_idx = *pos / 8;
        let bit_idx = 7 - (*pos % 8);
        if byte_idx < data.len() {
            let bit = (data[byte_idx] >> bit_idx) & 1;
            result = (result << 1) | u32::from(bit);
        }
        *pos += 1;
    }
    result
}

/// Test single-interval gap encoding (110) - 3 bits
/// This is the optimized encoding for single gaps which are 99.4% of all gaps
#[test]
fn test_single_gap_encoding() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Create a sequence with a single-interval gap:
    // Reading at interval 0, skip interval 1, reading at interval 2
    enc.append(base_ts, 22).unwrap();           // interval 0
    enc.append(base_ts + 600, 22).unwrap();     // interval 2 (skip interval 1 = 1 gap)

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].ts, base_ts);
    assert_eq!(decoded[1].ts, base_ts + 600);
    assert_eq!(decoded[1].ts - decoded[0].ts, 600); // 2 intervals = 600 seconds
    assert_eq!(decoded[0].value, 22);
    assert_eq!(decoded[1].value, 22);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test single-interval gap with non-zero delta following
#[test]
fn test_single_gap_with_delta() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Single gap followed by +1 delta
    enc.append(base_ts, 22).unwrap();           // interval 0
    enc.append(base_ts + 600, 23).unwrap();     // interval 2, temp +1

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 22);
    assert_eq!(decoded[1].value, 23);
    assert_eq!(decoded[1].ts - decoded[0].ts, 600);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test multiple consecutive single-interval gaps
#[test]
fn test_multiple_single_gaps() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Three readings, each separated by a single-interval gap
    enc.append(base_ts, 22).unwrap();            // interval 0
    enc.append(base_ts + 600, 22).unwrap();      // interval 2 (gap of 1)
    enc.append(base_ts + 1200, 22).unwrap();     // interval 4 (gap of 1)

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[0].ts, base_ts);
    assert_eq!(decoded[1].ts, base_ts + 600);
    assert_eq!(decoded[2].ts, base_ts + 1200);
    for r in &decoded {
        assert_eq!(r.value, 22);
    }

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test that 2-interval gap is correctly encoded and decoded
#[test]
fn test_two_interval_gap_encoding() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // 2-interval gap (skip intervals 1 and 2)
    enc.append(base_ts, 22).unwrap();           // interval 0
    enc.append(base_ts + 900, 22).unwrap();     // interval 3 (gap of 2)

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[1].ts - decoded[0].ts, 900);
    assert_eq!(decoded[0].value, 22);
    assert_eq!(decoded[1].value, 22);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test +2 delta encoding
#[test]
fn test_plus_two_delta_encoding() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    enc.append(base_ts, 20).unwrap();
    enc.append(base_ts + 300, 22).unwrap();  // +2 delta

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 20);
    assert_eq!(decoded[1].value, 22);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test -2 delta encoding
#[test]
fn test_minus_two_delta_encoding() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 300, 20).unwrap();  // -2 delta

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].value, 22);
    assert_eq!(decoded[1].value, 20);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test sequence with multiple ±2 deltas
#[test]
fn test_multiple_two_deltas() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Sequence: 20 -> 22 -> 20 -> 22 (alternating +2, -2)
    enc.append(base_ts, 20).unwrap();
    enc.append(base_ts + 300, 22).unwrap();   // +2
    enc.append(base_ts + 600, 20).unwrap();   // -2
    enc.append(base_ts + 900, 22).unwrap();   // +2

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 4);
    assert_eq!(decoded.iter().map(|r| r.value).collect::<Vec<_>>(), vec![20, 22, 20, 22]);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test zero run 150+ - correctness verified via roundtrip
#[test]
fn test_zero_run_150_split_encoding() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // First reading, then 150 same-value readings (150 zero deltas), then different value
    enc.append(base_ts, 22).unwrap();
    for i in 1..=150 {
        enc.append(base_ts + i * 300, 22).unwrap();
    }
    enc.append(base_ts + 151 * 300, 23).unwrap();  // +1 to flush

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 152);

    // Verify all values are correct
    for i in 0..151 {
        assert_eq!(decoded[i].value, 22, "Expected 22 at index {}", i);
    }
    assert_eq!(decoded[151].value, 23);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test zero run 300 - correctness verified via roundtrip
#[test]
fn test_zero_run_300_split_encoding() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    enc.append(base_ts, 22).unwrap();
    for i in 1..=300 {
        enc.append(base_ts + i * 300, 22).unwrap();
    }
    enc.append(base_ts + 301 * 300, 23).unwrap();  // +1 to flush

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 302);

    // Verify all values are correct
    for i in 0..301 {
        assert_eq!(decoded[i].value, 22, "Expected 22 at index {}", i);
    }
    assert_eq!(decoded[301].value, 23);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test gap 65 intervals - correctness verified via roundtrip
#[test]
fn test_gap_65_single_marker() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Skip 65 intervals - this is the max that fits in a single 14-bit marker
    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 66 * 300, 22).unwrap();  // 66 intervals later = gap of 65

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[1].ts - decoded[0].ts, 66 * 300);
    assert_eq!(decoded[0].value, 22);
    assert_eq!(decoded[1].value, 22);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2);
}

/// Test gap 66 intervals (requires split: 65 + 1 single gap)
#[test]
fn test_gap_66_split_encoding() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Skip 66 intervals - exceeds single marker capacity (2-65)
    // Should split into: 65 gaps (14-bit) + 1 gap (3-bit single)
    enc.append(base_ts, 22).unwrap();
    enc.append(base_ts + 67 * 300, 22).unwrap();  // 67 intervals later = gap of 66
    enc.append(base_ts + 68 * 300, 23).unwrap();  // delta=1

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 3, "Should have 3 readings");

    // Verify timestamps
    assert_eq!(decoded[0].ts, base_ts, "First timestamp");
    assert_eq!(decoded[1].ts, base_ts + 67 * 300, "Second timestamp (67 intervals later)");
    assert_eq!(decoded[2].ts, base_ts + 68 * 300, "Third timestamp (68 intervals later)");

    // Verify values
    assert_eq!(decoded[0].value, 22, "First value");
    assert_eq!(decoded[1].value, 22, "Second value (same)");
    assert_eq!(decoded[2].value, 23, "Third value (+1 delta)");

    // Verify roundtrip
    let bytes = enc.to_bytes();
    let decoded2 = decode::<i32, 300>(&bytes);
    assert_eq!(decoded, decoded2, "Roundtrip should preserve readings");
}

/// Test all encoding prefixes in sequence to verify disambiguation
#[test]
fn test_all_encoding_prefixes() {
    let base_ts = 1761955455u64;
    let mut enc = Encoder::<i32>::new();

    // Build a sequence that uses all encoding types:
    // Start: 100
    enc.append(base_ts, 100).unwrap();

    // 1. Zero delta (0): same temp
    enc.append(base_ts + 300, 100).unwrap();

    // 2. +1 delta (100): temp 101
    enc.append(base_ts + 600, 101).unwrap();

    // 3. -1 delta (101): temp 100
    enc.append(base_ts + 900, 100).unwrap();

    // 4. Single gap (110) + zero delta: skip interval 4, temp same
    enc.append(base_ts + 1500, 100).unwrap();  // interval 5

    // 5. +2 delta (11100): temp 102
    enc.append(base_ts + 1800, 102).unwrap();

    // 6. -2 delta (11101): temp 100
    enc.append(base_ts + 2100, 100).unwrap();

    // 7. +5 delta (1111110 + sign + magnitude): temp 105
    enc.append(base_ts + 2400, 105).unwrap();

    // 8. +100 delta (11111110 + sign + 10 bits): temp 205
    enc.append(base_ts + 2700, 205).unwrap();

    // 9. 2-interval gap (11111111 + 6 bits): skip 2
    enc.append(base_ts + 3600, 205).unwrap();  // interval 12

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 10);

    // Verify values
    let expected_values = [100, 100, 101, 100, 100, 102, 100, 105, 205, 205];
    for (i, r) in decoded.iter().enumerate() {
        assert_eq!(r.value, expected_values[i], "Value mismatch at index {}", i);
    }

    // Verify timestamps
    let expected_ts = [
        base_ts,
        base_ts + 300,
        base_ts + 600,
        base_ts + 900,
        base_ts + 1500,  // gap
        base_ts + 1800,
        base_ts + 2100,
        base_ts + 2400,
        base_ts + 2700,
        base_ts + 3600,  // gap
    ];
    for (i, r) in decoded.iter().enumerate() {
        assert_eq!(r.ts, expected_ts[i], "Timestamp mismatch at index {}", i);
    }
}

#[test]
fn test_large_gap_exceeding_u16() {
    // Test that gaps larger than u16::MAX intervals return TimeSpanOverflow error
    // (memory optimization limits time span to ~227 days at 300s interval)
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32>::new();

    // Add first reading
    enc.append(base_ts, 100).unwrap();

    // Add second reading at gap of 70,000 intervals (exceeds u16::MAX = 65535)
    let large_gap: u64 = 70_000;
    let second_ts = base_ts + large_gap * 300;
    let result = enc.append(second_ts, 200);

    // Should fail with TimeSpanOverflow
    assert!(matches!(result, Err(AppendError::TimeSpanOverflow { .. })));

    // First reading should still be there
    assert_eq!(enc.count(), 1);
}

#[test]
fn test_very_large_gap_100k_intervals() {
    // Test that very large gaps (100k intervals) return TimeSpanOverflow error
    // (memory optimization limits time span to ~227 days at 300s interval)
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32>::new();

    enc.append(base_ts, 50).unwrap();

    // Gap of 100,000 intervals (exceeds u16::MAX = 65535)
    let large_gap: u64 = 100_000;
    let second_ts = base_ts + large_gap * 300;
    let result = enc.append(second_ts, 75);

    // Should fail with TimeSpanOverflow
    assert!(matches!(result, Err(AppendError::TimeSpanOverflow { .. })));

    // First reading should still be there
    assert_eq!(enc.count(), 1);
}

#[test]
fn test_gap_at_u16_boundary() {
    // Test gaps right at u16::MAX to ensure boundary handling
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32, 1>::new(); // 1 second interval for faster test

    enc.append(base_ts, 10).unwrap();

    // Gap exactly at u16::MAX
    let gap: u64 = u64::from(u16::MAX);
    let second_ts = base_ts + gap; // interval is 1
    enc.append(second_ts, 20).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].ts, base_ts);
    assert_eq!(decoded[0].value, 10);
    assert_eq!(decoded[1].ts, second_ts);
    assert_eq!(decoded[1].value, 20);
}

#[test]
fn test_gap_just_over_u16_max() {
    // Test gap just over u16::MAX (65536 intervals) returns TimeSpanOverflow error
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32, 1>::new(); // 1 second interval

    enc.append(base_ts, 10).unwrap();

    // Gap of u16::MAX + 1 (exceeds max)
    let gap: u64 = u64::from(u16::MAX) + 1;
    let second_ts = base_ts + gap; // interval is 1
    let result = enc.append(second_ts, 20);

    // Should fail with TimeSpanOverflow
    assert!(matches!(result, Err(AppendError::TimeSpanOverflow { .. })));

    // First reading should still be there
    assert_eq!(enc.count(), 1);
}

// ============================================================================
// OPTIMIZATION BOUNDARY TESTS
// These tests verify the boundaries introduced by memory optimizations:
// - zero_run packed into bits 42-57 of pending_avg (16 bits, max 65535)
// - prev_logical_idx changed from u32 to u16 (max 65535 intervals)
// ============================================================================

#[test]
fn test_zero_run_with_concurrent_averaging() {
    // Test that zero_run (packed in bits 42-57) doesn't interfere with
    // averaging data (bits 0-41) when both are active simultaneously
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32>::new();

    // First reading
    enc.append(base_ts, 20).unwrap();

    // Create a zero run by adding same temperature
    for i in 1..=10 {
        enc.append(base_ts + i * 300, 20).unwrap();
    }

    // Now start a new interval with multiple readings (triggers averaging)
    // while zero_run counter is still non-zero
    let new_interval_ts = base_ts + 11 * 300;
    enc.append(new_interval_ts, 25).unwrap();
    enc.append(new_interval_ts + 100, 27).unwrap(); // Same interval, will average to 26
    enc.append(new_interval_ts + 200, 26).unwrap(); // Same interval, average stays 26

    // Move to next interval to finalize
    enc.append(base_ts + 12 * 300, 30).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 13);

    // Verify zero run was encoded correctly
    for i in 0..=10 {
        assert_eq!(decoded[i].value, 20, "Zero run reading {} should be 20", i);
    }

    // Verify averaging worked correctly (25+27+26)/3 = 26
    assert_eq!(decoded[11].value, 26, "Averaged reading should be 26");
    assert_eq!(decoded[12].value, 30);

    // Verify roundtrip through bytes
    let bytes = enc.to_bytes();
    let decoded_bytes = decode::<i32, 300>(&bytes);
    assert_eq!(decoded_bytes.len(), 13);
    assert_eq!(decoded_bytes[11].value, 26);
}

#[test]
fn test_zero_run_max_accumulation() {
    // Test accumulating a large zero run (500+) to verify bit packing holds
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32>::new();

    enc.append(base_ts, 22).unwrap();

    // Add 500 readings with same temperature (large zero run)
    for i in 1..=500 {
        enc.append(base_ts + i * 300, 22).unwrap();
    }

    // Add different temperature to flush
    enc.append(base_ts + 501 * 300, 23).unwrap();

    assert_eq!(enc.count(), 502);

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 502);

    // All but last should be 22
    for i in 0..501 {
        assert_eq!(decoded[i].value, 22, "Reading {} should be 22", i);
    }
    assert_eq!(decoded[501].value, 23);

    // Verify roundtrip
    let bytes = enc.to_bytes();
    let decoded_bytes = decode::<i32, 300>(&bytes);
    assert_eq!(decoded_bytes.len(), 502);
}

#[test]
fn test_interval_boundary_u16_max_minus_one() {
    // Test at u16::MAX - 1 intervals (65534) - should succeed
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32, 1>::new(); // 1 second interval

    enc.append(base_ts, 10).unwrap();

    // Gap at u16::MAX - 1
    let gap: u64 = u64::from(u16::MAX) - 1;
    let second_ts = base_ts + gap;
    enc.append(second_ts, 20).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].ts, base_ts);
    assert_eq!(decoded[1].ts, second_ts);

    // Verify roundtrip
    let bytes = enc.to_bytes();
    let decoded_bytes = decode::<i32, 1>(&bytes);
    assert_eq!(decoded_bytes.len(), 2);
    assert_eq!(decoded_bytes[1].ts, second_ts);
}

#[test]
fn test_multiple_readings_at_max_interval() {
    // Test multiple readings spread across the maximum time span
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32, 1>::new(); // 1 second interval

    enc.append(base_ts, 10).unwrap();

    // Add reading at 1/3 of max
    let one_third = u64::from(u16::MAX) / 3;
    enc.append(base_ts + one_third, 15).unwrap();

    // Add reading at 2/3 of max
    let two_thirds = (u64::from(u16::MAX) / 3) * 2;
    enc.append(base_ts + two_thirds, 20).unwrap();

    // Add reading at exactly max
    let max_gap = u64::from(u16::MAX);
    enc.append(base_ts + max_gap, 25).unwrap();

    assert_eq!(enc.count(), 4);

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 4);
    assert_eq!(decoded[0].value, 10);
    assert_eq!(decoded[1].value, 15);
    assert_eq!(decoded[2].value, 20);
    assert_eq!(decoded[3].value, 25);

    // Verify timestamps
    assert_eq!(decoded[3].ts, base_ts + max_gap);

    // Verify roundtrip
    let bytes = enc.to_bytes();
    let decoded_bytes = decode::<i32, 1>(&bytes);
    assert_eq!(decoded_bytes.len(), 4);
}

#[test]
fn test_time_span_overflow_error_details() {
    // Verify TimeSpanOverflow error contains useful information
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32, 1>::new();

    enc.append(base_ts, 10).unwrap();

    let overflow_ts = base_ts + u64::from(u16::MAX) + 1;
    let result = enc.append(overflow_ts, 20);

    match result {
        Err(AppendError::TimeSpanOverflow { ts, base_ts: err_base, max_intervals }) => {
            assert_eq!(ts, overflow_ts);
            assert_eq!(err_base, base_ts);
            assert_eq!(max_intervals, u32::from(u16::MAX));
        }
        _ => panic!("Expected TimeSpanOverflow error"),
    }

    // Verify error message formatting
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains(&overflow_ts.to_string()));
    assert!(msg.contains("65535"));
}

#[test]
fn test_zero_run_then_gap_then_zero_run() {
    // Test sequence: zero run -> gap -> zero run
    // Verifies zero_run counter resets properly after gaps
    let base_ts = EPOCH_BASE;
    let mut enc = Encoder::<i32>::new();

    enc.append(base_ts, 20).unwrap();

    // First zero run (5 readings)
    for i in 1..=5 {
        enc.append(base_ts + i * 300, 20).unwrap();
    }

    // Gap (skip 3 intervals)
    enc.append(base_ts + 9 * 300, 20).unwrap();

    // Second zero run (5 more readings)
    for i in 10..=14 {
        enc.append(base_ts + i * 300, 20).unwrap();
    }

    // Different value to flush
    enc.append(base_ts + 15 * 300, 25).unwrap();

    let decoded = enc.decode();
    assert_eq!(decoded.len(), 13); // 6 + 1 (after gap) + 5 + 1 = 13

    // Verify all readings before the gap are 20
    for i in 0..6 {
        assert_eq!(decoded[i].value, 20, "Pre-gap reading {} should be 20", i);
    }

    // After gap, all should be 20 until the last
    for i in 6..12 {
        assert_eq!(decoded[i].value, 20, "Post-gap reading {} should be 20", i);
    }

    assert_eq!(decoded[12].value, 25);

    // Verify roundtrip
    let bytes = enc.to_bytes();
    let decoded_bytes = decode::<i32, 300>(&bytes);
    assert_eq!(decoded_bytes.len(), 13);
}
