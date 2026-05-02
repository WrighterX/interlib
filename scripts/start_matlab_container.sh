#!/usr/bin/env bash
set -euo pipefail
# Start a MATLAB container for testing the interlib FFI bindings.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MATLAB_IMAGE="${MATLAB_IMAGE:-my-matlab-image:auth}"
MATLAB_CONTAINER="${MATLAB_CONTAINER:-matlab-login}"

# Build the FFI library first
CARGO_TARGET_DIR="${REPO_ROOT}/target/matlab" cargo build --lib --no-default-features --features matlab

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${MATLAB_CONTAINER}$"; then
    echo "Container ${MATLAB_CONTAINER} already exists. Starting it..."
    docker start -ai "${MATLAB_CONTAINER}"
else
    echo "Creating new container ${MATLAB_CONTAINER} from image ${MATLAB_IMAGE}..."
    docker run -it --rm \
        --name "${MATLAB_CONTAINER}" \
        -v "${REPO_ROOT}:/work" \
        -w /work \
        "${MATLAB_IMAGE}"
fi