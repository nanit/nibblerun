use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use nibblerun::{decode, Encoder};

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}_readings"), |b| {
            b.iter(|| {
                let mut enc: Encoder<i32> = Encoder::new();
                let base_ts = 1_760_000_000u64;
                for i in 0..count {
                    enc.append(base_ts + i * 300, black_box(22 + (i % 5) as i32)).unwrap();
                }
                black_box(enc.to_bytes())
            })
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    // Pre-encode data
    let mut enc: Encoder<i32> = Encoder::new();
    let base_ts = 1_760_000_000u64;
    for i in 0..10000u64 {
        enc.append(base_ts + i * 300, 22 + (i % 5) as i32).unwrap();
    }
    let bytes = enc.to_bytes();

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(10000));
    group.bench_function("10000_readings", |b| {
        b.iter(|| black_box(decode::<i32, 300>(black_box(&bytes))))
    });
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    group.throughput(Throughput::Elements(1000));
    group.bench_function("1000_readings", |b| {
        b.iter(|| {
            let mut enc: Encoder<i32> = Encoder::new();
            let base_ts = 1_760_000_000u64;
            for i in 0..1000u64 {
                enc.append(base_ts + i * 300, black_box(22)).unwrap();
            }
            let bytes = enc.to_bytes();
            black_box(decode::<i32, 300>(&bytes))
        })
    });
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
criterion_main!(benches);
