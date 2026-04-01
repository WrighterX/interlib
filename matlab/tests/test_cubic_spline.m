function test_cubic_spline()
%TEST_CUBIC_SPLINE Smoke test for the cubic spline MATLAB wrapper.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_CUBIC_SPLINE_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_CUBIC_SPLINE_LIBRARY', release_lib);
end

interp = interlib.CubicSplineInterpolator();

% y = x^2 (perfectly fit by cubic spline)
x = [0; 1; 2; 3; 4];
y = [0; 1; 4; 9; 16];
interp.fit(x, y);

single_value = interp(0.5);
assert(abs(single_value - 0.25) < 1e-12);

many_values = interp.evaluateMany([0.5; 1.5; 2.5]);
assert(all(abs(many_values - [0.25; 2.25; 6.25]) < 1e-12));

disp('test_cubic_spline passed');
end
