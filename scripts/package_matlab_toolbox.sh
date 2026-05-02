#!/usr/bin/env bash
set -euo pipefail
# Package and optionally test the MATLAB toolbox.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

bash "${SCRIPT_DIR}/build_matlab_toolbox.sh"

echo "MATLAB toolbox packaging complete."