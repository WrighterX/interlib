function test_chebyshev()
%TEST_CHEBYSHEV Smoke test for the MATLAB Chebyshev wrapper.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_CHEBYSHEV_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_CHEBYSHEV_LIBRARY', release_lib);
end

n_points = 4;
nodes = cos((2 * (0:n_points-1) + 1) * pi / (2 * n_points));
y = nodes .^ 2;

interp = interlib.ChebyshevInterpolator(uint64(n_points));
interp.fit(y);

single_value = interp(0.0);
assert(abs(single_value - 0.0) < 1e-12);

many_values = interp.evaluateMany([0.0; 0.5; -0.5]);
expected = [0.0; 0.25; 0.25];
assert(all(abs(many_values - expected) < 1e-12));

disp('test_chebyshev passed');
end
