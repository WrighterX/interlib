# Build Commands

## Python Development Build
```bash
maturin develop
```

## Python Release Build
```bash
maturin build --release
```

## Rust Build Only
```bash
cargo build --release
```

## MATLAB FFI Build
```bash
CARGO_TARGET_DIR=target/matlab cargo build --lib --no-default-features --features matlab
```

## Full Build with All Features
```bash
cargo build --all-features
```

---

# Test Commands

## Rust Tests
```bash
cargo test                          # All tests
cargo test <test_name>              # Single test
cargo test --package interlib --lib lagrange::tests  # Specific module
```

## Python Tests
```bash
python python/tests/test_api.py     # API contract test
```

## MATLAB Tests (Docker)
```bash
make matlab-test                    # Linear test
make matlab-test-newton             # Newton test
make matlab-test-hermite            # Hermite test
make matlab-test-lagrange           # Lagrange test
make matlab-test-least-squares      # Least squares test
make matlab-test-rbf                # RBF test
make matlab-test-chebyshev          # Chebyshev test
make matlab-test-quadratic          # Quadratic test
make matlab-test-cubic-spline       # Cubic spline test
make matlab-test-installation       # Installation test
```

---

# Benchmark Commands

## Important Note for Benchmarking
**Always use release builds for performance testing.** Debug builds can be 10x slower:
```bash
maturin build --release
pip install --force-reinstall target/wheels/interlib-*.whl
```

## Python Benchmarks
```bash
python python/benches/bench.py                    # Full benchmark suite
python python/benches/core_focus_bench.py         # Core algorithm focus
python python/benches/real_data_bench.py          # Real data benchmarks
python python/benches/limited_comparison.py       # Limited comparison
```

---

# MATLAB Commands

## Build MATLAB FFI
```bash
CARGO_TARGET_DIR=target/matlab cargo build --lib --no-default-features --features matlab
```

## MATLAB Toolbox
```bash
make matlab-toolbox-stage          # Stage toolbox files
make matlab-toolbox-build          # Build toolbox
make matlab-toolbox-package        # Package toolbox
make matlab-toolbox-package-batch  # Batch packaging
```

## MATLAB Testing
```bash
make matlab-session                # Start MATLAB container
make matlab-test                   # Run linear test
make matlab-test-<method>          # Run specific test
make matlab-test-<method>-batch    # Run in batch mode
```

---

# Lint Commands

## Rust Linting
```bash
cargo clippy -- -D warnings        # Run Rust linter
cargo clippy -- -D warnings -D missing_docs  # With documentation check
```

## Formatting
```bash
cargo fmt                          # Format Rust code
cargo fmt --check                  # Check formatting without applying
```

---

# Quick Reference

## Development Cycle
1. `maturin develop` - Build Python module
2. `cargo test` - Run Rust tests
3. `python python/tests/test_api.py` - Run Python tests
4. `cargo clippy -- -D warnings` - Check for warnings
5. `cargo fmt --check` - Check formatting

## Release Cycle
1. `cargo test` - All tests pass
2. `cargo clippy -- -D warnings` - No warnings
3. `cargo fmt` - Format code
4. `maturin build --release` - Build release wheel

## MATLAB Development
1. `CARGO_TARGET_DIR=target/matlab cargo build --lib --no-default-features --features matlab`
2. `make matlab-test` - Run MATLAB tests
