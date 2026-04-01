# MATLAB Wrapper

This folder contains the MATLAB-side wrapper for the Rust C ABI
interpolators.

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

## Tests and Demos

Smoke tests:
- `matlab/tests/test_linear.m`
- `matlab/tests/test_newton.m`
- `matlab/tests/test_quadratic.m`
- `matlab/tests/test_cubic_spline.m`
- `matlab/tests/test_hermite.m`
- `matlab/tests/test_lagrange.m`
- `matlab/tests/test_least_squares.m`

Plot demos:
- `matlab/examples/plot_linear_demo.m`
- `matlab/examples/plot_newton_demo.m`
- `matlab/examples/plot_quadratic_demo.m`
- `matlab/examples/plot_cubic_spline_demo.m`
- `matlab/examples/plot_hermite_demo.m`
- `matlab/examples/plot_lagrange_demo.m`
- `matlab/examples/plot_least_squares_demo.m`

## Container Workflow

Container helper:
- `../scripts/start_matlab_container.sh`
- `../scripts/run_matlab_linear_test.sh`
- `../scripts/run_matlab_newton_test.sh`
- `../scripts/run_matlab_quadratic_test.sh`
- `../scripts/run_matlab_cubic_spline_test.sh`
- `../scripts/run_matlab_hermite_test.sh`
- `../scripts/run_matlab_lagrange_test.sh`
- `../scripts/run_matlab_least_squares_test.sh`

Recommended workflow for a Login Named User license:

1. Run `make matlab-session`
2. In another terminal run `make matlab-test` (Linear), `make matlab-test-newton` (Newton), etc.
3. In the MATLAB prompt run:

```matlab
clear classes
addpath('/work/matlab')
addpath('/work/matlab/tests')
test_linear % or test_newton, test_quadratic, test_cubic_spline, test_hermite, test_lagrange, test_least_squares
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

The container workflow in this repo is headless by default. If you want a GUI
MATLAB session, use the container's browser or VNC mode instead of the default
CLI launch.

Example browser-mode launch:

```bash
docker run -it --rm -p 8888:8888 --shm-size=512M my-matlab-image:auth -browser
```
