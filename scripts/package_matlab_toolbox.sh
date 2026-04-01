#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${MATLAB_IMAGE:-mathworks/matlab:r2025b}"
batch_mode="${MATLAB_BATCH:-0}"
container_name="${MATLAB_CONTAINER:-matlab-login}"

echo "Packaging MATLAB toolbox from staged bundle."
echo "Expected stage directory: ${MATLAB_TOOLBOX_DIST_DIR:-$repo_root/dist/matlab-toolbox}"

if [[ "$batch_mode" == "1" ]]; then
  echo "Using MATLAB image: ${image}"
  container_id="$(docker create -it \
    -v "$repo_root:/work" \
    -w /work \
    -e MATLAB_TOOLBOX_DIST_DIR="${MATLAB_TOOLBOX_DIST_DIR:-/work/dist/matlab-toolbox}" \
    -e MATLAB_TOOLBOX_FILE="${MATLAB_TOOLBOX_FILE:-/tmp/interlib.mltbx}" \
    "$image" \
    /bin/sh -lc "matlab -licmode onlinelicensing -batch \"run('/work/scripts/package_matlab_toolbox.m')\"")"
  trap 'docker rm -f "$container_id" >/dev/null 2>&1 || true' EXIT
  docker start -ai "$container_id"
  mkdir -p "$repo_root/dist"
  docker cp "$container_id:/tmp/interlib.mltbx" "$repo_root/dist/interlib.mltbx"
  docker rm -f "$container_id" >/dev/null 2>&1 || true
  trap - EXIT
else
  matlab -licmode onlinelicensing -batch "run('$repo_root/scripts/package_matlab_toolbox.m')"
fi
