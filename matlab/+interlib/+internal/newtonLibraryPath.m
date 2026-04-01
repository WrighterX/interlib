function path = newtonLibraryPath()
env_path = getenv('INTERLIB_NEWTON_LIBRARY');
if ~isempty(env_path)
    path = string(env_path);
    return;
end

env_path = getenv('INTERLIB_NATIVE_LIBRARY');
if ~isempty(env_path)
    path = string(env_path);
    return;
end

path = interlib.internal.nativeLibraryPath();
end
