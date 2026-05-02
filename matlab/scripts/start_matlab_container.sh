#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${MATLAB_IMAGE:-mathworks/matlab:r2025b}"
container_name="${MATLAB_CONTAINER:-matlab-login}"

printf 'Starting MATLAB container: %s\n' "$container_name"
printf 'Using MATLAB image: %s\n' "$image"
printf 'Repository mount: %s -> /work\n' "$repo_root"

if docker container inspect "$container_name" >/dev/null 2>&1; then
  printf 'Reusing existing container: %s\n' "$container_name"
  docker start -ai "$container_name"
else
  docker run --rm -it \
    --name "$container_name" \
    -v "$repo_root:/work" \
    -w /work \
    -e INTERLIB_LINEAR_LIBRARY=/work/target/matlab/debug/libinterlib.so \
    "$image" \
    matlab -licmode onlinelicensing
fi
