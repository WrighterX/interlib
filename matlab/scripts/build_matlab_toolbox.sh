#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist_root="${MATLAB_TOOLBOX_DIST_DIR:-$repo_root/dist/matlab-toolbox}"
toolbox_root="${MATLAB_TOOLBOX_ROOT:-$dist_root}"
toolbox_file="${MATLAB_TOOLBOX_FILE:-$repo_root/dist/interlib.mltbx}"

bash "$repo_root/scripts/stage_matlab_toolbox.sh"

cat > "$toolbox_root/TOOLBOX_MANIFEST.txt" <<'EOF'
interlib MATLAB toolbox manifest placeholder

This repository does not yet generate a real .mltbx file.
The eventual packaging step should consume the staged toolbox root and emit:
- toolbox name and version
- included MATLAB files
- bundled native binaries
- install entrypoint metadata
EOF

cat > "$toolbox_root/matlab/INSTALL.m" <<'EOF'
function result = INSTALL()
%INSTALL Placeholder entrypoint for future MATLAB toolbox packaging.
%
% The packaged toolbox should expose a post-install verification hook. The
% current source-tree workflow already provides:
%   interlib.verify_installation()
%
% This file exists so the staged toolbox has an obvious install hook when a
% real packaging command is added.

result = interlib.verify_installation();
end
EOF

echo "Staged MATLAB toolbox root: $toolbox_root"
echo "Toolbox output placeholder: $toolbox_file"
echo "Next step: add the MATLAB packaging command that turns the staged root into a real .mltbx file."
