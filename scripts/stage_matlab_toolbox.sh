#!/usr/bin/env bash
set -euo pipefail
# Stage MATLAB toolbox files into a temporary directory for packaging.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STAGING_DIR="${REPO_ROOT}/build/matlab-toolbox"
mkdir -p "${STAGING_DIR}"

# Copy MATLAB source files
cp -r "${REPO_ROOT}/matlab/+interlib" "${STAGING_DIR}/+interlib"

# Copy the compiled shared library
if [ "$(uname)" = "Linux" ]; then
    cp "${REPO_ROOT}/target/matlab/debug/libinterlib.so" "${STAGING_DIR}/"
elif [ "$(uname)" = "Darwin" ]; then
    cp "${REPO_ROOT}/target/matlab/debug/libinterlib.dylib" "${STAGING_DIR}/"
else
    cp "${REPO_ROOT}/target/matlab/debug/interlib.dll" "${STAGING_DIR}/"
fi

# Copy C headers
cp "${REPO_ROOT}/matlab/headers/interlib_native.h" "${STAGING_DIR}/"

echo "Toolbox staged at ${STAGING_DIR}"