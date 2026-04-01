# MATLAB Wrapper

This folder contains the MATLAB-side wrapper for the linear Rust C ABI
prototype.

Add this folder to the MATLAB path, then use:

```matlab
interp = interlib.LinearInterpolator();
interp.fit([0; 1; 2], [0; 1; 4]);
y = interp(0.5);
```

By default the wrapper looks for the standalone MATLAB/FFI build in
`target/matlab/debug` or `target/matlab/release`. It falls back to
`target/debug` or `target/release` only if needed. You can override the path
with the environment variable `INTERLIB_LINEAR_LIBRARY`.

Smoke test:

- `matlab/tests/test_linear.m`

Plot demo:

- `matlab/examples/plot_linear_demo.m`

Container helper:

- `../scripts/start_matlab_container.sh`
- `../scripts/run_matlab_linear_test.sh`

Recommended workflow for a Login Named User license:

1. Run `make matlab-session`
2. In another terminal run `make matlab-test`
3. In the MATLAB prompt run:

```matlab
clear classes
addpath('/work/matlab')
addpath('/work/matlab/tests')
test_linear
```

For a plot demo in the same session:

```matlab
clear classes
addpath('/work/matlab')
addpath('/work/matlab/examples')
plot_linear_demo
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
