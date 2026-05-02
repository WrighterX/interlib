#!/usr/bin/env bash
set -euo pipefail
# Build the MATLAB toolbox (.mltbx) from staged files.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STAGING_DIR="${REPO_ROOT}/build/matlab-toolbox"
OUTPUT_DIR="${REPO_ROOT}/dist"
mkdir -p "${OUTPUT_DIR}"

# Build the toolbox using MATLAB (if available) or just report the staged files.
if command -v matlab &> /dev/null; then
    matlab -batch "matlab.addons.toolbox.packageToolbox('${STAGING_DIR}/toolbox.prj', '${OUTPUT_DIR}/interlib.mltbx')"
    echo "Toolbox packaged to ${OUTPUT_DIR}/interlib.mltbx"
else
    echo "MATLAB not found. Toolbox files are staged at ${STAGING_DIR}"
    echo "Run 'matlab -batch \"matlab.addons.toolbox.packageToolbox('${STAGING_DIR}/toolbox.prj', '${OUTPUT_DIR}/interlib.mltbx')\"' in MATLAB to package."
fi