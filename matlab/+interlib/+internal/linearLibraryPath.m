function path = linearLibraryPath()
env_path = getenv('INTERLIB_LINEAR_LIBRARY');
if ~isempty(env_path)
    path = string(env_path);
    return;
end

matlab_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
repo_root = fileparts(matlab_dir);

if ispc
    filenames = ["interlib.dll"];
elseif ismac
    filenames = ["libinterlib.dylib", "interlib.dylib"];
else
    filenames = ["libinterlib.so", "interlib.so"];
end

search_dirs = {
    fullfile(repo_root, 'target', 'matlab', 'debug')
    fullfile(repo_root, 'target', 'matlab', 'release')
    fullfile(repo_root, 'target', 'debug')
    fullfile(repo_root, 'target', 'release')
};

for dir_idx = 1:numel(search_dirs)
    for file_idx = 1:numel(filenames)
        candidate = fullfile(search_dirs{dir_idx}, char(filenames(file_idx)));
        if isfile(candidate)
            path = string(candidate);
            return;
        end
    end
end

error('interlib:LinearLibraryMissing', ...
    'Could not locate the interlib shared library. Set INTERLIB_LINEAR_LIBRARY or build the Rust library into target/matlab/debug, target/matlab/release, target/debug, or target/release.');
