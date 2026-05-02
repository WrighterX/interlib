# Interlib Validation & Delivery Evidence

This report records the execution evidence for the OpenSpec change
`complete-validation-docs-case-studies`.

## Environment

- Repository: `interlib/` (Rust + PyO3 + MATLAB FFI)
- Python runtime: `interlib/.venv/bin/python`
- Build/install workflow: `python -m maturin develop` (local), `maturin build` (wheel)
- MATLAB runtime tests: license/container dependent (recorded explicitly when skipped)

## 1) Audit Against Project Requirement List

### 1.1 Implementation inventory

- Rust/Python interpolators exported in `src/lib.rs`:
  - `LinearInterpolator`
  - `LagrangeInterpolator`
  - `NewtonInterpolator`
  - `QuadraticInterpolator`
  - `CubicSplineInterpolator`
  - `HermiteInterpolator`
  - `LeastSquaresInterpolator`
  - `RBFInterpolator`
  - `ChebyshevInterpolator`
- Shared algorithm cores present under `src/*_core.rs`.
- MATLAB FFI wrappers and tests present under `matlab/+interlib/`, `matlab/tests/`, and `matlab/examples/`.

### 1.2 Validation/comparison coverage audit

Scripts with SciPy reference comparisons are present in:

- `python/benches/bench.py`
- `python/benches/altbench.py`
- `python/benches/limited_comparison.py`
- `python/benches/real_data_bench.py`

Coverage notes:

- Direct SciPy/NumPy-style references exist for linear, quadratic (`interp1d`), cubic spline, Lagrange/Newton-style polynomial baselines, Hermite baseline (`CubicHermiteSpline` in real-data bench), and RBF.
- Methods without strict one-to-one SciPy equivalent in all scripts are validated via analytical/case-study behavior and method-specific checks (notably Chebyshev and least-squares approximation behavior).

### 1.3 Visualization assets audit

Visualization tooling exists in:

- `python/benches/bench.py`
- `python/benches/altbench.py`
- `python/benches/self_analysis_visual.py`
- MATLAB plot demos under `matlab/examples/plot_*_demo.m`

Chosen output directory for generated Python evidence artifacts:

- `interlib/python/benches/benchmark_plots/`

Chosen output directory for case-study run logs:

- `interlib/python/case_studies/output/`

### 1.4 Documentation/tutorial gap audit

Current strengths:

- `README.md` already demonstrates basic import and fit/evaluate usage.
- `GUIDE.md` contains method summaries and selection guidance.
- `matlab/README.md` documents the separate FFI/native-library model.

Gaps to close:

- README should include an explicit local dev workflow (`.venv` + `maturin develop`) alongside wheel installation.
- README should provide a concise API contract reminder (scalar input -> `float`, sequence input -> ordered `list[float]`).
- README/GUIDE should better surface common error handling examples and method-family tutorial snippets.
- README should clearly cross-link Python wheel distribution versus MATLAB FFI/toolbox distribution in one place.

## 2) Command Execution Log

### 2.1 Python API smoke + rebuild workflow

Executed:

```bash
cd interlib
.venv/bin/python -m maturin develop
.venv/bin/python python/tests/api_contract_test.py
.venv/bin/python python/tests/_libtest.py
```

Evidence files:

- `docs/evidence/logs/maturin_develop.log`
- `docs/evidence/logs/api_contract_test.log`
- `docs/evidence/logs/libtest.log`

Result summary:

- `maturin develop`: success.
- `api_contract_test.py`: `API contract smoke test passed`.
- `_libtest.py`: launched all script tests successfully.

### 2.2 SciPy/reference comparison scripts

Executed:

```bash
cd interlib
.venv/bin/python python/benches/limited_comparison.py
```

Evidence file:

- `docs/evidence/logs/limited_comparison.log`

Result summary:

- Script completed and printed per-method timing comparisons against SciPy references.
- Observed runtime warning from SciPy barycentric baseline and RBF singular-matrix failures at larger synthetic sizes; run completed with those method-level failures reported by the script.

### 2.3 Real-data validation with cached datasets

Executed:

```bash
cd interlib
.venv/bin/python python/benches/real_data_bench.py --dataset noaa --station KSFO --field temperature --methods linear,cubicspline,rbf --sizes 16,32 --n-runs 2 --n-warmup 1
.venv/bin/python python/benches/real_data_bench.py --dataset nasa --command 499 --axis x --methods linear,cubicspline,rbf --sizes 16,32 --n-runs 2 --n-warmup 1
```

Evidence files:

- `docs/evidence/logs/real_data_noaa.log`
- `docs/evidence/logs/real_data_nasa.log`

Result summary:

- NOAA and NASA cached-data runs completed successfully.
- Comparative interlib vs SciPy fit/eval/error metrics were produced for selected methods and train sizes.

### 2.4 Script/API drift check

- No Python API-contract drift requiring code changes was observed in this validation pass.
- No numerical-algorithm modifications were made during tasks 2.1–2.4.

### 2.5 Reviewer-facing validation record

- This report (`docs/evidence/VALIDATION_EVIDENCE.md`) plus the log files under `docs/evidence/logs/` form the reviewer-facing validation record.

## 3) Visualization Outputs

### 3.1 Scripts executed

Executed:

```bash
cd interlib
.venv/bin/python python/benches/self_analysis_visual.py
.venv/bin/python python/benches/bench.py --runs 2 --sizes 10 20 50 --output-dir python/benches/benchmark_plots
```

Evidence logs:

- `docs/evidence/logs/self_analysis_visual.log`
- `docs/evidence/logs/bench_visual.log`

### 3.2 Generated output directory

All Python visualization artifacts were saved under:

- `python/benches/benchmark_plots/`

Generated files include:

- `performance_curves.png`
- `method_comparison.png`
- `scaling_comparison.png`
- `summary_comparison.png`
- per-method performance charts (e.g., `performance_linear.png`, `performance_cubicspline.png`)
- per-method accuracy charts (from both visual scripts)

### 3.3 What the generated plots demonstrate

- `performance_*` and `summary_comparison.png`: timing comparisons for fit/eval workloads.
- `scaling_comparison.png` and `performance_curves.png`: scaling trends versus dataset size.
- `accuracy_*` and `method_comparison.png`: interpolation fit quality and residual behavior on analytical functions.

### 3.4 Visualization drift handling

- Fixed script-level drift in `python/benches/bench.py`:
  - removed a broken SciPy kwargs pass-through for RBF benchmark config,
  - fixed `plot_accuracy_chart` to avoid relying on an undefined global `args` symbol.
- Remaining RBF SciPy issues in `bench.py` (2D input contract and singular matrix at some settings) are documented as known script limitations in section 5.

## 4) Case Study Results

### 4.1 Function approximation (`cos(x)`, Runge)

Executed:

```bash
cd interlib
.venv/bin/python python/case_studies/function_approx.py
```

Evidence:

- `docs/evidence/logs/case_function_approx.log`
- `python/case_studies/output/case_function_approx.log`

Highlights:

- `cos(x)` test completed with low mean/max errors across methods.
- Runge-function test clearly demonstrated large error for uniform-node high-degree Lagrange and improved behavior for local/specialized methods.
- Reported insight text in script output confirms expected Runge phenomenon behavior.

### 4.2 Signal reconstruction from sampled data

Executed:

```bash
cd interlib
.venv/bin/python python/case_studies/signal_rec.py
```

Evidence:

- `docs/evidence/logs/case_signal_rec.log`
- `python/case_studies/output/case_signal_rec.log`

Highlights:

- Clean, noisy, and undersampled scenarios all executed.
- Output includes comparative mean-error tables and interpretation notes.
- Script conclusions document denoising trade-offs and undersampling/Nyquist limitations.

### 4.3 Engineering datasets (including temperature)

Executed:

```bash
cd interlib
.venv/bin/python python/case_studies/engineering.py
```

Evidence:

- `docs/evidence/logs/case_engineering.log`
- `python/case_studies/output/case_engineering.log`

Highlights:

- Temperature profile case completed with per-method error/estimate table.
- Pressure-volume and stress-strain engineering cases also completed.
- Script conclusions provide method recommendations for engineering usage patterns.

## 5) Limitations and Skipped Checks

Current limitations noted in this phase:

- Some synthetic benchmark scenarios in `limited_comparison.py` report method-level numerical issues (e.g., RBF singular matrix at larger sizes).
- `python/benches/bench.py` still has partial RBF SciPy-baseline limitations for certain paths:
  - SciPy RBF evaluation expects 2D input shape,
  - singular-matrix failures may occur for selected synthetic settings.
- Benchmark timing numbers are machine-dependent and reported as local evidence only.
- MATLAB runtime tests require licensed MATLAB/container access and are handled separately from Python validation.
- In this run, MATLAB runtime smoke tests were not executed; only the MATLAB/FFI build path (`make matlab-build`) was validated.

## 6) Requirement-to-Evidence Mapping

### Requirement 2: Implement interpolation methods

Status: **Complete**

Evidence:

- Rust/PyO3 module exports in `src/lib.rs` for 9 public interpolators.
- Core modules under `src/*_core.rs`.
- Python smoke coverage under `python/tests/` including `api_contract_test.py`.
- MATLAB wrappers and smoke tests under `matlab/+interlib/` and `matlab/tests/`.

### Requirement 3: Validate against existing numerical libraries

Status: **Complete (local evidence collected)**

Evidence:

- SciPy comparison scripts executed/logged:
  - `python/benches/limited_comparison.py`
  - `python/benches/real_data_bench.py` (NOAA + NASA cached datasets)
- Logs:
  - `docs/evidence/logs/limited_comparison.log`
  - `docs/evidence/logs/real_data_noaa.log`
  - `docs/evidence/logs/real_data_nasa.log`

Notes:

- Some method/settings combinations report numerical limitations (documented in section 5).

### Requirement 4: Develop visualization tools for comparative analysis

Status: **Complete**

Evidence:

- Visualization scripts executed:
  - `python/benches/self_analysis_visual.py`
  - `python/benches/bench.py`
- Generated artifacts in `python/benches/benchmark_plots/`.
- Representative outputs:
  - `performance_curves.png`
  - `method_comparison.png`
  - `scaling_comparison.png`
  - `summary_comparison.png`

### Requirement 5: Provide user documentation and tutorials

Status: **Complete (updated)**

Evidence:

- Updated `README.md`:
  - local dev workflow (`.venv` + `maturin develop`)
  - API quickstart and return-shape behavior
  - method-family tutorial guidance
  - explicit Python wheel vs MATLAB FFI distinction
- Updated `GUIDE.md` with mini tutorial examples for method families.
- Updated `matlab/README.md` with explicit distribution-path distinction.

### Requirement 6: Perform case studies

Status: **Complete**

Evidence (executed and logged):

- Function approximation (`cos(x)`, Runge):
  - `python/case_studies/function_approx.py`
  - `docs/evidence/logs/case_function_approx.log`
- Signal reconstruction:
  - `python/case_studies/signal_rec.py`
  - `docs/evidence/logs/case_signal_rec.log`
- Engineering datasets (temperature, pressure-volume, stress-strain):
  - `python/case_studies/engineering.py`
  - `docs/evidence/logs/case_engineering.log`

Copied output logs for convenience:

- `python/case_studies/output/case_function_approx.log`
- `python/case_studies/output/case_signal_rec.log`
- `python/case_studies/output/case_engineering.log`

## 7) Final Verification Commands

Executed from `interlib/`:

```bash
cargo check
cargo test --lib
.venv/bin/python python/tests/api_contract_test.py
make matlab-build
```

Evidence:

- `docs/evidence/logs/cargo_check.log`
- `docs/evidence/logs/cargo_test_lib.log`
- `docs/evidence/logs/api_contract_test_final.log`
- `docs/evidence/logs/matlab_build.log`

OpenSpec validation:

```bash
openspec validate complete-validation-docs-case-studies
```

Result: valid.
