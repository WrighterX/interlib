function test_rbf()
%TEST_RBF Smoke test for the MATLAB RBF wrapper.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_RBF_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_RBF_LIBRARY', release_lib);
end

interp = interlib.RBFInterpolator('gaussian', 1.0);

x = [0; 1; 2; 3];
y = [0; 1; 4; 2];
interp.fit(x, y);

single_value = interp(1);
assert(abs(single_value - 1) < 1e-12);

many_values = interp.evaluateMany([0.0; 1.0; 2.0]);
assert(all(abs(many_values - [0.0; 1.0; 4.0]) < 1e-12));

weights = interp.get_weights();
assert(numel(weights) == numel(x));

disp('test_rbf passed');
end
