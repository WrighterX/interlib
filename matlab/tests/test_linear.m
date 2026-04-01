function test_linear()
%TEST_LINEAR Smoke test for the linear MATLAB wrapper.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_LINEAR_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_LINEAR_LIBRARY', release_lib);
end

interp = interlib.LinearInterpolator();

x = [0; 1; 2];
y = [0; 1; 4];
interp.fit(x, y);

single_value = interp(0.5);
assert(abs(single_value - 0.5) < 1e-12);

many_values = interp.evaluateMany([0.0; 0.5; 1.5; 2.0]);
assert(all(abs(many_values - [0.0; 0.5; 2.5; 4.0]) < 1e-12));

disp('test_linear passed');
end
