function path = leastSquaresLibraryPath()
env_path = getenv('INTERLIB_LEAST_SQUARES_LIBRARY');
if ~isempty(env_path)
    path = string(env_path);
    return;
end

path = interlib.internal.nativeLibraryPath();
end
