# MATLAB Docker Test

This repo includes a MATLAB smoke test for the linear prototype.

Requirements:

- Docker installed
- A valid MathWorks MATLAB license
- The standalone FFI Rust library built for Linux, typically
  `target/matlab/debug/libinterlib.so`

For a Login Named User license, use an interactive container as the primary
workflow. A fresh `matlab -batch` process may ask for licensing again even when
an interactive session is already authenticated.

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

Batch mode:

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

The MATLAB wrappers now share one native alias, `interlib_native`, so the
order in which you load linear, Newton, quadratic, cubic spline, or later
wrappers does not matter in a fresh MATLAB session.

Committed-image flow:

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
