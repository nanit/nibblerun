.PHONY: test fuzz fuzz-parallel clean

FUZZ_TIME ?= 30

test:
	cargo test

fuzz:
	@for target in fuzz_roundtrip fuzz_decode fuzz_idempotent fuzz_lossy_bounds fuzz_single_reading fuzz_averaging fuzz_gaps fuzz_lossless; do \
		echo "=== Running $$target ===" && \
		cargo fuzz run $$target -- -max_total_time=$(FUZZ_TIME) || exit 1; \
	done

fuzz-parallel:
	cargo fuzz list | xargs -P7 -I{} cargo fuzz run {} -- -max_total_time=$(FUZZ_TIME)

clean:
	cargo clean
	rm -rf fuzz/target fuzz/corpus fuzz/artifacts
