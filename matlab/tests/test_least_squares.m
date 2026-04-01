function test_least_squares()
%TEST_LEAST_SQUARES Smoke test for MATLAB least squares wrapper.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_LEAST_SQUARES_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_LEAST_SQUARES_LIBRARY', release_lib);
end

interp = interlib.LeastSquaresInterpolator(2);

x = [0; 1; 2; 3];
y = [0; 1; 4; 9];
interp.fit(x, y);

single = interp(2.5);
assert(abs(single - 6.25) < 1e-12);

many = interp.evaluateMany([0.5; 1.5; 2.5]);
assert(all(abs(many - [0.25; 2.25; 6.25]) < 1e-12));

disp('test_least_squares passed');
end
