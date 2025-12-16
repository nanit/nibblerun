.PHONY: build test clippy fmt bench coverage coverage-html fuzz fuzz-coverage clean

build:
	cargo build

test:
	cargo test

clippy:
	cargo clippy

fmt:
	cargo fmt

# Run benchmarks
bench:
	cargo bench

# Run tests with coverage summary
coverage:
	cargo llvm-cov --summary-only

# Generate HTML coverage report
coverage-html:
	cargo llvm-cov --html
	@echo "Report: target/llvm-cov/html/index.html"

# Run all fuzz targets for 30 seconds each
fuzz:
	@for target in fuzz_decode fuzz_roundtrip fuzz_gaps fuzz_idempotent fuzz_lossy_bounds fuzz_single_reading fuzz_averaging fuzz_lossless; do \
		echo "Running $$target..."; \
		cargo +nightly fuzz run $$target -- -max_total_time=30; \
	done

# Generate combined fuzz coverage report
fuzz-coverage:
	@for target in fuzz_decode fuzz_roundtrip fuzz_gaps fuzz_idempotent fuzz_lossy_bounds fuzz_single_reading fuzz_averaging fuzz_lossless; do \
		echo "Generating coverage for $$target..."; \
		cargo +nightly fuzz coverage $$target; \
	done
	@echo "Merging coverage data..."
	llvm-profdata merge -sparse fuzz/coverage/*/coverage.profdata -o fuzz/coverage/merged.profdata
	@echo "Generating report..."
	llvm-cov report \
		target/$$(rustc -vV | grep host | cut -d' ' -f2)/coverage/$$(rustc -vV | grep host | cut -d' ' -f2)/release/fuzz_roundtrip \
		-instr-profile=fuzz/coverage/merged.profdata

clean:
	cargo clean
	rm -rf fuzz/coverage fuzz/corpus
