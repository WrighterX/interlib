function path = cubicSplineLibraryPath()
env_path = getenv('INTERLIB_CUBIC_SPLINE_LIBRARY');
if ~isempty(env_path)
    path = string(env_path);
    return;
end
path = interlib.internal.nativeLibraryPath();
end
