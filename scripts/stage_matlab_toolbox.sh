#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist_root="${MATLAB_TOOLBOX_DIST_DIR:-$repo_root/dist/matlab-toolbox}"
target_dir="${INTERLIB_MATLAB_TARGET_DIR:-$repo_root/target/matlab}"

if [[ -d "$dist_root" ]]; then
  rm -rf "$dist_root"
fi

mkdir -p "$dist_root/matlab/native" "$dist_root/matlab/tests" "$dist_root/matlab/examples" "$dist_root/matlab/headers"

cp -R "$repo_root/matlab/+interlib" "$dist_root/matlab/"
cp -R "$repo_root/matlab/tests/." "$dist_root/matlab/tests/"
cp -R "$repo_root/matlab/examples/." "$dist_root/matlab/examples/"
cp -R "$repo_root/matlab/headers/." "$dist_root/matlab/headers/"
cp "$repo_root/matlab/README.md" "$dist_root/matlab/README.md"
cp "$repo_root/matlab/MATLAB_DOCKER.md" "$dist_root/matlab/MATLAB_DOCKER.md"

if [[ -f "$target_dir/debug/libinterlib.so" ]]; then
  cp "$target_dir/debug/libinterlib.so" "$dist_root/matlab/native/"
fi
if [[ -f "$target_dir/release/libinterlib.so" ]]; then
  cp "$target_dir/release/libinterlib.so" "$dist_root/matlab/native/"
fi
if [[ -f "$target_dir/debug/interlib.dll" ]]; then
  cp "$target_dir/debug/interlib.dll" "$dist_root/matlab/native/"
fi
if [[ -f "$target_dir/debug/libinterlib.dylib" ]]; then
  cp "$target_dir/debug/libinterlib.dylib" "$dist_root/matlab/native/"
fi

cat > "$dist_root/TOOLBOX_LAYOUT.txt" <<'EOF'
This directory is a staging area for the future interlib MATLAB toolbox.

Expected contents:
- matlab/+interlib/ wrapper package
- matlab/tests/ smoke tests
- matlab/examples/ demos
- matlab/headers/ native headers
- matlab/native/ platform-specific shared libraries
EOF

echo "Staged MATLAB toolbox bundle at: $dist_root"
