#!/usr/bin/env bash
set -euo pipefail
# Run MATLAB test for chebyshev interpolator.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/run_matlab_test.sh" "test_${method}"
