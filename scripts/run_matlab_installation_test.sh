#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
library_path="${repo_root}/target/matlab/debug/libinterlib.so"

echo "Using MATLAB library: ${library_path}"

if [[ "${MATLAB_BATCH:-0}" == "1" ]]; then
  echo "Running MATLAB installation verification in batch mode."
  matlab -licmode onlinelicensing -batch "addpath('${repo_root}/matlab'); addpath('${repo_root}/matlab/tests'); test_installation"
else
  container_name="${MATLAB_CONTAINER:-matlab-login}"
  echo "Interactive MATLAB workflow selected."
  echo "Expected running container: ${container_name}"
  echo
  echo "Run these commands inside the MATLAB prompt in ${container_name}:"
  echo "clear classes"
  echo "addpath('/work/matlab')"
  echo "addpath('/work/matlab/tests')"
  echo "test_installation"
fi
