#!/usr/bin/env bash
set -euo pipefail
# Run a MATLAB test script inside a Docker container.
# Usage: run_matlab_test.sh <test_script_name>
#   test_script_name: e.g., "test_linear", "test_newton", etc. (without .m)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEST_SCRIPT="${1:-test_linear}"
MATLAB_CONTAINER="${MATLAB_CONTAINER:-matlab-login}"
MATLAB_BATCH="${MATLAB_BATCH:-0}"

# Ensure the FFI library is built
CARGO_TARGET_DIR="${REPO_ROOT}/target/matlab" cargo build --lib --no-default-features --features matlab 2>&1

# Set environment variable for the library path
export INTERLIB_LIB_PATH="${REPO_ROOT}/target/matlab/debug"

if [ "${MATLAB_BATCH}" = "1" ]; then
    # Batch mode: run without interactive MATLAB
    docker exec "${MATLAB_CONTAINER}" \
        matlab -batch "cd /work; addpath(genpath('/work/matlab')); ${TEST_SCRIPT}"
else
    # Interactive mode: run in the existing container
    docker exec -it "${MATLAB_CONTAINER}" \
        matlab -batch "cd /work; addpath(genpath('/work/matlab')); ${TEST_SCRIPT}"
fi