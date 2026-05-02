# MATLAB Wrapper

This folder contains the MATLAB-side wrapper for the Rust C ABI
interpolators.

> MATLAB integration is separate from Python wheel distribution.
> Python users install a PyO3/maturin wheel, while MATLAB users load the
> standalone FFI native library (`libinterlib.so`/`interlib.dll`/`libinterlib.dylib`) through these `.m` wrappers.

## Usage

Add this folder to the MATLAB path, then use:

### Linear Interpolation
```matlab
interp = interlib.LinearInterpolator();
interp.fit([0; 1; 2], [0; 1; 4]);
y = interp(0.5);
```

### Newton Interpolation
```matlab
interp = interlib.NewtonInterpolator();
interp.fit([0; 1; 2], [0; 1; 4]);
y = interp(0.5);
```

### Quadratic Interpolation
```matlab
interp = interlib.QuadraticInterpolator();
interp.fit([0; 1; 2; 3], [0; 1; 4; 9]);
y = interp(0.5);
```

### Cubic Spline Interpolation
```matlab
interp = interlib.CubicSplineInterpolator();
interp.fit([0; 1; 2; 3], [0; 1; 4; 9]);
y = interp(0.5);
```

## Library Loading

By default the wrappers look for the standalone MATLAB/FFI build in
`target/matlab/debug` or `target/matlab/release`. They fall back to
`target/debug` or `target/release` only if needed. You can override the path
with environment variables:
- `INTERLIB_LINEAR_LIBRARY`
- `INTERLIB_NEWTON_LIBRARY`
- `INTERLIB_QUADRATIC_LIBRARY`
- `INTERLIB_CUBIC_SPLINE_LIBRARY`
- `INTERLIB_HERMITE_LIBRARY`
- `INTERLIB_LAGRANGE_LIBRARY`
- `INTERLIB_LEAST_SQUARES_LIBRARY`
- `INTERLIB_NATIVE_LIBRARY`

All MATLAB wrappers now share one loaded native library alias
(`interlib_native`) in a session. This is intentional so the load order of
linear, Newton, quadratic, cubic spline, and future wrappers does not matter.

## Installation Model

This repository currently uses a source-tree development model. That means the
MATLAB wrapper folder is added to the path directly from the checkout, and the
Rust crate builds the native library separately.

That is fine for development, but it is not the easiest possible setup for a
scientist or regular MATLAB user. The long-term packaging target is a MATLAB
toolbox (`.mltbx`) with a small installer or verification function that checks
the native library and path setup automatically.

CI currently covers the Rust MATLAB/FFI binaries only. The `.mltbx` package is
created locally because it still depends on a working MATLAB runtime and
license.

For now, the simplest local workflow is still:

```matlab
addpath('/path/to/interlib/matlab')
```

followed by constructing the interpolator class you need.

If you are using a packaged release or future toolbox build, run:

```matlab
interlib.verify_installation()
```

or:

```matlab
interlib.install()
```

These entrypoints are meant to confirm that MATLAB can see the package and
that the native shared library loads before you start using the interpolators.

To create the MATLAB toolbox locally, run:

```bash
MATLAB_IMAGE=my-matlab-image:auth make matlab-toolbox-package-batch
```

That command stages the toolbox bundle, runs MATLAB packaging in the container,
and writes `dist/interlib.mltbx`.

## Tests and Demos

Smoke tests:
- `matlab/tests/test_linear.m`
- `matlab/tests/test_installation.m`
- `matlab/tests/test_newton.m`
- `matlab/tests/test_quadratic.m`
- `matlab/tests/test_cubic_spline.m`
- `matlab/tests/test_hermite.m`
- `matlab/tests/test_lagrange.m`
- `matlab/tests/test_least_squares.m`
- `matlab/tests/test_rbf.m`
- `matlab/tests/test_chebyshev.m`

Plot demos:
- `matlab/examples/plot_linear_demo.m`
- `matlab/examples/plot_newton_demo.m`
- `matlab/examples/plot_quadratic_demo.m`
- `matlab/examples/plot_cubic_spline_demo.m`
- `matlab/examples/plot_hermite_demo.m`
- `matlab/examples/plot_lagrange_demo.m`
- `matlab/examples/plot_least_squares_demo.m`
- `matlab/examples/plot_rbf_demo.m`
- `matlab/examples/plot_chebyshev_demo.m`

## Container Workflow

Container helper:
- `../scripts/start_matlab_container.sh`
- `../scripts/run_matlab_linear_test.sh`
- `../scripts/run_matlab_installation_test.sh`
- `../scripts/run_matlab_newton_test.sh`
- `../scripts/run_matlab_quadratic_test.sh`
- `../scripts/run_matlab_cubic_spline_test.sh`
- `../scripts/run_matlab_hermite_test.sh`
- `../scripts/run_matlab_lagrange_test.sh`
- `../scripts/run_matlab_least_squares_test.sh`
- `../scripts/run_matlab_rbf_test.sh`
- `../scripts/run_matlab_chebyshev_test.sh`

### Docker Requirements

- Docker installed
- A valid MathWorks MATLAB license
- The standalone FFI Rust library built for Linux, typically
  `target/matlab/debug/libinterlib.so`

For a Login Named User license, use an interactive container as the primary
workflow. A fresh `matlab -batch` process may ask for licensing again even when
an interactive session is already authenticated.

### Interactive Workflow

Primary workflow:

```bash
make matlab-session
```

Then, in a second terminal:

```bash
make matlab-test
```

`make matlab-test` builds the standalone FFI Rust library, checks that the `matlab-login`
container is running, and prints the MATLAB commands to execute inside the
already open MATLAB prompt:

```bash
clear classes
addpath('/work/matlab')
addpath('/work/matlab/tests')
test_linear
```

If you use a different container name:

```bash
MATLAB_CONTAINER=my-container make matlab-session
MATLAB_CONTAINER=my-container make matlab-test
```

If you use a different image:

```bash
MATLAB_IMAGE=my-matlab-image:auth make matlab-session
```

### Batch Mode

```bash
MATLAB_IMAGE=mathworks/matlab:r2025b make matlab-test-batch
```

For Hermite run `make matlab-test-hermite` (interactive) or
`make matlab-test-hermite-batch` when your license allows unattended runs.

For Lagrange run `make matlab-test-lagrange` (interactive) or
`make matlab-test-lagrange-batch` when your license allows unattended runs.

For RBF run `make matlab-test-rbf` (interactive) or
`make matlab-test-rbf-batch` once the container workflow is ready for headless automation.

For Chebyshev run `make matlab-test-chebyshev` (interactive) or
`make matlab-test-chebyshev-batch` when your license supports automating the run.

Only use `matlab-test-batch` if your license setup supports non-interactive
batch launches without prompting again.

The MATLAB build uses:

```bash
CARGO_TARGET_DIR=target/matlab cargo build --lib --no-default-features --features ffi
```

This is intentional. The default Python build links against PyO3 and is not a
standalone MATLAB-loadable shared library.

### Committed-Image Flow

1. Start the official image and log in once.
2. Commit the running container with `docker commit <container_id> my-matlab-image`.
3. Reuse that image by setting `MATLAB_IMAGE=my-matlab-image` when running the helper.

For Login Named User licensing, the mounted interactive workflow above is still
the recommended path even after committing an authenticated image.

## Headless Versus GUI

The default container workflow here is headless. That is why MATLAB opens in a
terminal and why figures are not shown on screen by default.

If you want a GUI session, start MATLAB in the container's browser or VNC mode
instead of plain CLI mode. MathWorks documents those container entry modes for
interactive visual sessions. In that setup, you connect through the exposed
browser or VNC port rather than the terminal.

Example browser-mode launch:

```bash
docker run -it --rm -p 8888:8888 --shm-size=512M my-matlab-image:auth -browser
```

Then open the browser URL exposed by the container, or use the MATLAB image's
documented browser session flow.

Practical summary:

- CLI mode: good for smoke tests and automated runs
- browser/VNC mode: good for interactive figures and desktop-style usage

This repository currently uses CLI mode for the integration test workflow and
the plot demo saves an image file instead of opening a window.

### Recommended Workflow for Login Named User License

1. Run `make matlab-session`
2. In another terminal run `make matlab-test` (Linear), `make matlab-test-newton` (Newton), etc.
3. In the MATLAB prompt run:

```matlab
clear classes
addpath('/work/matlab')
addpath('/work/matlab/tests')
test_installation % or test_linear, test_newton, test_quadratic, test_cubic_spline, test_hermite, test_lagrange, test_least_squares, test_rbf, test_chebyshev
```

For a plot demo in the same session:

```matlab
clear classes
addpath('/work/matlab')
addpath('/work/matlab/examples')
plot_linear_demo % or plot_newton_demo, plot_quadratic_demo, plot_cubic_spline_demo, plot_hermite_demo, plot_lagrange_demo, plot_least_squares_demo
```

In a headless Docker container, the demo saves a PNG to
`$HOME/matlab_demo_output/linear_demo.png` inside the container instead of
opening a visible figure window. You can also pass a custom output path:

```matlab
plot_linear_demo('/tmp/linear_demo.png')
```
