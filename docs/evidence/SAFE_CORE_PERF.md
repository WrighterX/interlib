# Safe Core Performance Investigation (`improve-safe-rust-core`)

## Scope

This report captures the baseline audit, wrapper dependency paths, and before/after benchmark evidence for safe core-performance work.

## 1) Core API Audit (Task 1.1)

Audited modules:

- `src/linear_core.rs`
- `src/quadratic_core.rs`
- `src/cubic_spline_core.rs`
- `src/hermite_core.rs`
- `src/lagrange_core.rs`
- `src/newton_core.rs`
- `src/least_squares_core.rs`
- `src/rbf_core.rs`
- `src/chebyshev_core.rs`

### API surface summary

| Core | `evaluate_single` | `evaluate_many` | `fill_many` | Notes |
|---|---|---|---|---|
| linear | ✅ | ✅ | ✅ | `evaluate_many` currently has its own loop (not delegated) |
| quadratic | ✅ | ✅ | ✅ | `evaluate_many` currently has its own loop (not delegated) |
| cubic_spline | ✅ | ✅ | ✅ | `evaluate_many` currently has its own loop (not delegated) |
| hermite | ✅ | ✅ | ✅ | `evaluate_many` already delegates through `fill_many` |
| lagrange | ✅ | ✅ | ✅ | `evaluate_many` currently has its own loop (not delegated) |
| newton | ✅ | ✅ | ✅ | `evaluate_many` currently has its own loop (not delegated) |
| least_squares | ✅ | ✅ | ✅ | `evaluate_many` already delegates through `fill_many` |
| rbf | ✅ | ✅ | ✅ | `evaluate_many` already delegates through `fill_many` |
| chebyshev | ✅ | ❌ | ✅ | no `evaluate_many` convenience method yet |

Allocation behavior baseline:

- Every `evaluate_many` returns `Vec<f64>` (allocates output).
- `fill_many` exists in all cores and can avoid output reallocation when wrappers use it.
- Several `evaluate_many` methods duplicate loop logic instead of centralizing through `fill_many`.

## 2) Wrapper Dependency Paths (Task 1.2)

### Python wrappers (`src/*.rs`)

- Linear wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `update_y`, `add_point`, `repr`
- Quadratic wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `repr`
- Cubic spline wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `num_segments`, `repr`
- Hermite wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `get_coefficients`, `repr`
- Lagrange wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `update_y`, `add_point`, `repr`
- Newton wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `get_coefficients`, `repr`
- Least squares wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `get_coefficients`, `degree`, `r_squared`, `repr`
- RBF wrapper uses: `fit`, `evaluate_single`, `evaluate_many`, `weights`, `repr`
- Chebyshev wrapper uses: `fit`, `evaluate_single`, `fill_many`, `nodes`, `coefficients`, `set_method`, `repr`

### MATLAB/FFI wrappers (`src/ffi/*.rs`)

- Linear FFI uses: `fit`, `evaluate_single`, `fill_many`
- Quadratic FFI uses: `fit`, `evaluate_single`, `fill_many`
- Cubic spline FFI uses: `fit`, `evaluate_single`, `fill_many`
- Hermite FFI uses: `fit`, `evaluate_single`, `fill_many`
- Lagrange FFI uses: `fit`, `update_y`, `add_point`, `evaluate_single`, `fill_many`
- Newton FFI uses: `fit`, `evaluate_single`, `fill_many`
- Least squares FFI uses: `fit`, `evaluate_single`, `fill_many`
- RBF FFI uses: `fit`, `evaluate_single`, `fill_many`
- Chebyshev FFI uses: `fit`, `evaluate_single`, `fill_many`

Observation: MATLAB path already strongly prefers `fill_many`, so centralizing core bulk logic around `fill_many` benefits both MATLAB and Python.

## 3) Focused Baseline Benchmark (Task 1.3)

Command:

```bash
cd interlib
.venv/bin/python python/benches/core_focus_bench.py --runs 3 --warmup 1
```

Log:

- `docs/evidence/logs/safe_core_focus_before.log`

### Baseline highlights

- Linear large-vector eval: interlib ~25–29 ms vs SciPy ~1.45 ms (interlib slower)
- Cubic spline large-vector eval: interlib ~24–28 ms vs SciPy ~1.8–1.9 ms (interlib slower)
- Hermite large-vector eval: interlib ~98–351 ms vs SciPy ~1.8–2.0 ms (interlib much slower)
- RBF large-vector eval (n=32): interlib ~54.7 ms vs SciPy ~17.3 ms (interlib slower)
- RBF n=64/128 failed in interlib with singular matrix error.

## 4) Baseline command/evidence index (Task 1.4)

- Benchmark harness: `python/benches/core_focus_bench.py`
- Baseline benchmark log: `docs/evidence/logs/safe_core_focus_before.log`
- API grep snapshots used in audit (from session commands)

## 5) Core API Consistency Changes (Tasks 2.1–2.2)

Implemented:

- Standardized `evaluate_many -> fill_many` delegation in:
  - `linear_core`
  - `quadratic_core`
  - `cubic_spline_core`
  - `lagrange_core`
  - `newton_core`
  - `hermite_core`
- Added `ChebyshevCore::evaluate_many` convenience path delegating to `fill_many`.
- Updated Chebyshev Python list-input path to call `core.evaluate_many`.

Result:

- Bulk evaluation logic is more centralized.
- MATLAB/FFI and Python now share the same core batch path more consistently.

## 6) Piecewise Safe Fast Paths (Tasks 3.1–3.2)

Implemented monotonic-query fast paths in safe Rust:

- `LinearCore::fill_many`: detects non-decreasing query vectors and advances a single segment cursor.
- `QuadraticCore::fill_many`: non-decreasing fast path using moving partition position.
- `CubicSplineCore::fill_many`: non-decreasing fast path using moving segment index.

Fallback for unordered input remains unchanged and correctness-compatible.

## 7) Correctness Tests (Tasks 2.3, 3.3, 4.3, 5.4)

Added or extended Rust tests to compare:

- `evaluate_many`
- `fill_many`
- repeated `evaluate_single`

for:

- linear
- quadratic
- cubic spline
- newton
- hermite

Added RBF tests for:

- successful small fit/evaluate
- duplicate-x rejection
- singular-system diagnostic error

## 8) Hermite Scaling Investigation (Tasks 4.1–4.4)

Inspection outcome:

- Hermite core already uses Horner-style evaluation on the Newton/Hermite basis.
- Dominant cost is algorithmic (`O(num_coefficients * num_queries)`) for large vector workloads.
- No safe low-risk optimization was applied that changes asymptotic behavior without changing semantics.

Compatibility checks:

- Rust tests include Hermite scalar/vector consistency.
- Python API contract smoke still passes.

## 9) RBF Robustness and Diagnostics (Tasks 5.1–5.5)

Implemented:

- Duplicate x-value detection in `RBFCore::fit` (sorted-copy check).
- Improved singular-system error text with pivot/context diagnostics.
- Added tests covering duplicate-node rejection and singular solve diagnostics.

Decision on regularization (Task 5.3):

- Keep regularization/smoothing as a **separate future change** to avoid implicit semantic changes in this optimization pass.

Observed benchmark status (after changes):

- RBF remains slower than SciPy in the focused large-vector benchmark.
- RBF still fails for some larger train sizes due to singularity.
- Diagnostics are clearer, but solver robustness still needs a separate numerical-design change.

## 10) Minimal Typed Error Pattern (Tasks 6.1–6.4)

Implemented:

- Added `src/core_error.rs` with `CoreError` enum + `Display` conversion to `String`.
- Applied pattern to representative cores:
  - `LinearCore`
  - `RBFCore`

Verification:

- Python maps these core errors to `ValueError` via existing wrapper conversions.
- FFI still routes errors through `fail(...)` and `*_last_error` / `interlib_last_error` reporting.

Future migration guidance:

- Continue typed-error rollout incrementally by core in a follow-up change, preserving user-facing error compatibility.

## 11) Before/After Focused Benchmark Snapshot (Task 3.4, 7.5)

Harness:

- `python/benches/core_focus_bench.py`

Logs:

- Before: `docs/evidence/logs/safe_core_focus_before.log`
- After: `docs/evidence/logs/safe_core_focus_after_final.log`

Highlights:

- **Linear eval (100k points)** improved from ~25–29 ms to ~17–18 ms.
- **Cubic spline eval (100k points)** improved from ~24–28 ms to ~18–19 ms.
- **Hermite** remained dominated by high-order basis evaluation cost.
- **RBF** remained limited by singularity and slower eval for tested settings.

Interpretation:

- Safe monotonic fast paths improved piecewise evaluation throughput materially.
- Further gains likely require deeper algorithmic changes (especially Hermite/RBF), not `unsafe` micro-optimizations.

## 12) Command and Validation Log Index

- Baseline benchmark: `docs/evidence/logs/safe_core_focus_before.log`
- After benchmark: `docs/evidence/logs/safe_core_focus_after_final.log`
- Rust tests: `docs/evidence/logs/safe_core_final_cargo_test.log`
- Rust check: `docs/evidence/logs/safe_core_final_cargo_check.log`
- Python rebuild: `docs/evidence/logs/safe_core_final_maturin.log`
- Python API contract: `docs/evidence/logs/safe_core_final_api_contract.log`
- Python error mapping check: `docs/evidence/logs/safe_core_python_error_mapping.log`
- FFI check: `docs/evidence/logs/safe_core_ffi_check_after_impl.log`
- MATLAB build: `docs/evidence/logs/safe_core_final_matlab_build.log`

