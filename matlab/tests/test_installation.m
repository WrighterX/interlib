function test_installation()
%TEST_INSTALLATION Smoke test for the MATLAB toolbox verification entrypoint.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_NATIVE_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_NATIVE_LIBRARY', release_lib);
end

result = interlib.verify_installation();
assert(result.ok, 'verify_installation returned a failure result');

disp('test_installation passed');
end
