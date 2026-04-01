function plot_lagrange_demo(output_path)
%PLOT_LAGRANGE_DEMO Plot Lagrange interpolation demo.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_LAGRANGE_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_LAGRANGE_LIBRARY', release_lib);
end

interp = interlib.LagrangeInterpolator();

x = (0:5)';
y = sin(x);
interp.fit(x, y);

xq = linspace(min(x), max(x), 200)';
yq = interp.evaluateMany(xq);

if nargin < 1 || isempty(output_path)
    output_dir = fullfile(getenv('HOME'), 'matlab_demo_output');
    output_path = fullfile(output_dir, 'lagrange_demo.png');
else
    output_dir = fileparts(output_path);
end

if ~isfolder(output_dir)
    mkdir(output_dir);
end

fig = figure('Name', 'interlib Lagrange MATLAB Demo', 'Color', 'w', 'Visible', 'off');
plot(xq, yq, 'LineWidth', 2.0, 'Color', [0.00, 0.45, 0.74]);
hold on;
scatter(x, y, 60, 'filled', 'MarkerFaceColor', [0.85, 0.33, 0.10]);
grid on;
xlabel('x');
ylabel('y');
title('LagrangeInterpolator MATLAB Demo (sin(x))');
legend({'Interpolated curve', 'Data points'}, 'Location', 'best');
hold off;

exportgraphics(fig, output_path);
disp(['Saved plot to ', output_path]);
end
