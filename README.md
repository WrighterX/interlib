# interlib
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit)

interlib is a Python library for interpolation methods designed as an alternative to scipy.interpolate. Being built on Rust, it provides reliable and in some cases faster solutions to unknown data point problems. It includes polynomial, piecewise, approximation-based and advanced interpolators for all of your needs.

## Installation

### End users (wheel)

Install from a built wheel:

```bash
pip install interlib-<version>-<platform>.whl
```

### Local development (recommended)

From `interlib/`:

```bash
.venv/bin/python -m maturin develop
```

This rebuilds and installs the extension in editable mode inside the project venv.

## Python API Quickstart

```python
from interlib import LinearInterpolator

interp = LinearInterpolator()
interp.fit([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])

# scalar input -> float
y1 = interp(1.5)

# sequence input -> ordered list[float]
y_many = interp([0.5, 1.5])
```

Common user errors are reported as `ValueError` (for example: unfitted usage,
length mismatch, invalid constructor parameters, out-of-range Chebyshev input).

## Tutorial by Method Family

- **Polynomial exact-fit** (`LagrangeInterpolator`, `NewtonInterpolator`):
  good for small clean datasets; can oscillate at high degree/uniform nodes.
- **Piecewise local** (`LinearInterpolator`, `QuadraticInterpolator`, `CubicSplineInterpolator`):
  robust defaults for many engineering signals; cubic spline is the smoothest common default.
- **Derivative-constrained** (`HermiteInterpolator`): use when you know `dy` at sample points.
- **Approximation/noisy data** (`LeastSquaresInterpolator`): fit trend instead of exact point pass-through.
- **Kernel/global** (`RBFInterpolator`): flexible smooth interpolation; kernel/epsilon tuning matters.
- **Chebyshev function approximation** (`ChebyshevInterpolator`): stable high-accuracy approximation over fixed interval.

See `GUIDE.md` for a fuller selection guide and pitfalls.

## Real-Data Benchmarks

The repository includes cached real-data benchmark entrypoints:

```bash
python python/benches/real_data_bench.py --dataset noaa --station KSFO --field temperature
python python/benches/real_data_bench.py --dataset nasa --command 499 --axis x
```

These benchmark against held-out real observations (NOAA/NASA) instead of only synthetic functions.

## Case Studies

Runnable case studies:

- `python/case_studies/function_approx.py` (includes `cos(x)` and Runge function)
- `python/case_studies/signal_rec.py` (sampled signal reconstruction)
- `python/case_studies/engineering.py` (engineering-style datasets, including temperature profiles)

## MATLAB Integration (important distinction)

MATLAB integration is **not** the Python wheel path.

- Python uses **PyO3 + maturin wheels**.
- MATLAB uses **standalone Rust FFI shared library + MATLAB `.m` wrappers**.

MATLAB build path:

```bash
make matlab-build
```

MATLAB docs:

- `matlab/README.md`
- `matlab/MATLAB_DOCKER.md`

## MATLAB Release Notes

GitHub Actions currently targets standalone Rust MATLAB/FFI binaries.
The `.mltbx` toolbox package is built locally from a licensed MATLAB runtime:

```bash
MATLAB_IMAGE=my-matlab-image:auth make matlab-toolbox-package-batch
```

This writes `dist/interlib.mltbx`.
