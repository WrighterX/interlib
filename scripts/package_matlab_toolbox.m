function outputFile = package_matlab_toolbox()
%PACKAGE_MATLAB_TOOLBOX Package the staged bundle into a .mltbx file.
%
% This script expects scripts/stage_matlab_toolbox.sh to have prepared the
% dist/matlab-toolbox directory first. It packages the staged folder with the
% current platform as the supported target.

repo_root = fileparts(fileparts(mfilename('fullpath')));
toolbox_root = fullfile(repo_root, 'dist', 'matlab-toolbox', 'matlab');
timestamp = char(datetime('now', 'Format', 'yyyyMMdd-HHmmss'));
output_env = getenv('MATLAB_TOOLBOX_FILE');
if ~isempty(output_env)
    output_file = char(output_env);
else
    output_file = fullfile(repo_root, 'dist', ['interlib-' timestamp '.mltbx']);
end

if ~isfolder(toolbox_root)
    error('interlib:ToolboxStageMissing', ...
        'Toolbox staging directory not found: %s', toolbox_root);
end

identifier = "8a6f8f77-1d5a-4f61-8b07-3f1b7f7b5d21";
opts = matlab.addons.toolbox.ToolboxOptions(toolbox_root, identifier);
opts.ToolboxName = "interlib";
opts.ToolboxVersion = "0.1.0";
opts.Summary = "Rust-backed interpolation toolbox for MATLAB";
opts.Description = "interlib packages MATLAB wrappers around Rust interpolation cores.";
opts.OutputFile = output_file;
opts.MinimumMatlabRelease = "R2023a";

if ispc
    opts.SupportedPlatforms.Win64 = true;
    opts.SupportedPlatforms.Glnxa64 = false;
    opts.SupportedPlatforms.Maci64 = false;
    opts.SupportedPlatforms.MatlabOnline = false;
elseif ismac
    opts.SupportedPlatforms.Win64 = false;
    opts.SupportedPlatforms.Glnxa64 = false;
    opts.SupportedPlatforms.Maci64 = true;
    opts.SupportedPlatforms.MatlabOnline = false;
else
    opts.SupportedPlatforms.Win64 = false;
    opts.SupportedPlatforms.Glnxa64 = true;
    opts.SupportedPlatforms.Maci64 = false;
    opts.SupportedPlatforms.MatlabOnline = false;
end

matlab.addons.toolbox.packageToolbox(opts);
outputFile = string(output_file);
fprintf('Packaged toolbox: %s\n', outputFile);
end
