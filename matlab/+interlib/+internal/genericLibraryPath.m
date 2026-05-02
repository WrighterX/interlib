function path = genericLibraryPath(method)
% GENERICLIBRARYPATH Get the library path for any method
%
% Parameters:
%   method - String name of the interpolation method (e.g., 'linear', 'newton')
%
% Returns:
%   path - Path to the library file

env_var = sprintf('INTERLIB_%s_LIBRARY', upper(method));
env_path = getenv(env_var);
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
