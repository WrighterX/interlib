function path = rbfLibraryPath()
env_path = getenv('INTERLIB_RBF_LIBRARY');
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
