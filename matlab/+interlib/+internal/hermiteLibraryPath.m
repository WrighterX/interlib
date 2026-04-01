function path = hermiteLibraryPath()
env_path = getenv('INTERLIB_HERMITE_LIBRARY');
if ~isempty(env_path)
    path = string(env_path);
    return;
end

path = interlib.internal.nativeLibraryPath();
end
