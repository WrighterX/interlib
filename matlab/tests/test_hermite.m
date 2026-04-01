function test_hermite()
%TEST_HERMITE Smoke test for the Hermite MATLAB wrapper.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_HERMITE_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_HERMITE_LIBRARY', release_lib);
end

interp = interlib.HermiteInterpolator();

x = [0; 1; 2];
y = [0; 1; 8];
dy = [0; 3; 12];
interp.fit(x, y, dy);

single_value = interp(0.5);
assert(abs(single_value - 0.125) < 1e-12);

many_values = interp.evaluateMany([0.5; 1.5; 2.5]);
assert(all(abs(many_values - [0.125; 3.375; 15.625]) < 1e-12));

disp('test_hermite passed');
end
