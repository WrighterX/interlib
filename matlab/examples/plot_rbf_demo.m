function plot_rbf_demo(output_path)
%PLOT_RBF_DEMO Plot an RBF interpolation example and save the result.

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

x = linspace(0, 4, 6)';
y = sin(x) + 0.1 * x;
interp.fit(x, y);

xq = linspace(min(x), max(x), 200)';
yq = arrayfun(@(value) interp.evaluate(value), xq);

if nargin < 1 || isempty(output_path)
    output_dir = fullfile(getenv('HOME'), 'matlab_demo_output');
    output_path = fullfile(output_dir, 'rbf_demo.png');
else
    output_dir = fileparts(output_path);
end

if ~isfolder(output_dir)
    mkdir(output_dir);
end

fig = figure('Name', 'interlib RBF MATLAB Demo', 'Color', 'w', 'Visible', 'off');
plot(xq, yq, 'LineWidth', 2.0, 'Color', [0.30, 0.60, 0.42]);
hold on;
scatter(x, y, 60, 'filled', 'MarkerFaceColor', [0.85, 0.33, 0.10]);
grid on;
xlabel('x');
ylabel('y');
title('RBFInterpolator MATLAB Demo');
legend({'Interpolated curve', 'Data points'}, 'Location', 'best');
hold off;

exportgraphics(fig, output_path);
disp(['Saved plot to ', output_path]);
end
