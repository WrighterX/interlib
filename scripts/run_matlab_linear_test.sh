#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${MATLAB_IMAGE:-mathworks/matlab:r2025b}"
target_dir="${INTERLIB_MATLAB_TARGET_DIR:-$repo_root/target/matlab}"
library_path="${INTERLIB_LINEAR_LIBRARY:-$target_dir/debug/libinterlib.so}"
container_name="${MATLAB_CONTAINER:-}"
batch_mode="${MATLAB_BATCH:-0}"

CARGO_TARGET_DIR="$target_dir" cargo build --lib --no-default-features --features ffi
printf 'Using linear library: %s\n' "$library_path"

if [[ "$batch_mode" == "1" ]]; then
  printf 'Using MATLAB image: %s\n' "$image"
  docker run --rm -it \
    -v "$repo_root:/work" \
    -w /work \
    -e INTERLIB_LINEAR_LIBRARY="$library_path" \
    "$image" \
    matlab -licmode onlinelicensing -batch "addpath('matlab'); test_linear"
else
  container_name="${container_name:-matlab-login}"
  printf 'Interactive MATLAB workflow selected.\n'
  printf 'Expected running container: %s\n' "$container_name"
  docker exec "$container_name" pwd >/dev/null
  printf '\n'
  printf 'Run these commands inside the MATLAB prompt in %s:\n' "$container_name"
  printf "clear classes\n"
  printf "addpath('/work/matlab')\n"
  printf "addpath('/work/matlab/tests')\n"
  printf "test_linear\n"
fi
