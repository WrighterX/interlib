function plot_chebyshev_demo(output_path)
%PLOT_CHEBYSHEV_DEMO Plot Chebyshev interpolation demo.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(repo_root, 'matlab'));

debug_lib = fullfile(repo_root, 'target', 'matlab', 'debug', 'libinterlib.so');
release_lib = fullfile(repo_root, 'target', 'matlab', 'release', 'libinterlib.so');

if isfile(debug_lib)
    setenv('INTERLIB_CHEBYSHEV_LIBRARY', debug_lib);
elseif isfile(release_lib)
    setenv('INTERLIB_CHEBYSHEV_LIBRARY', release_lib);
end

N_POINTS = 10;
interp = interlib.ChebyshevInterpolator(uint64(N_POINTS), -2.0, 2.0, true);

nodes = cos((2 * (0:N_POINTS-1) + 1) * pi / (2 * N_POINTS));
y = nodes .^ 2;
interp.fit(y);

xq = linspace(-2.0, 2.0, 300)';
yq = arrayfun(@(xi) interp.evaluate(xi), xq);

if nargin < 1 || isempty(output_path)
    output_dir = fullfile(getenv('HOME'), 'matlab_demo_output');
    output_path = fullfile(output_dir, 'chebyshev_demo.png');
else
    output_dir = fileparts(output_path);
end

if ~isfolder(output_dir)
    mkdir(output_dir);
end

fig = figure('Name', 'interlib Chebyshev MATLAB Demo', 'Color', 'w', 'Visible', 'off');
plot(xq, yq, 'LineWidth', 2.0, 'Color', [0.29, 0.54, 0.78]);
hold on;
scatter(nodes, y, 60, 'filled', 'MarkerFaceColor', [0.85, 0.33, 0.10]);
grid on;
xlabel('x');
ylabel('y');
title('ChebyshevInterpolator MATLAB Demo');
legend({'Interpolated curve', 'Chebyshev nodes'}, 'Location', 'best');
hold off;

exportgraphics(fig, output_path);
disp(['Saved plot to ', output_path]);
end
