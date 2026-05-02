function genericDestroy(handle, method)
% GENERICDESTROY Destroy an interpolator handle for any method
%
% Parameters:
%   handle - Pointer to the interpolator handle
%   method - String name of the interpolation method (e.g., 'linear', 'newton')

if isempty(handle)
    return;
end

alias = interlib.internal.nativeAlias();
func_name = sprintf('interlib_%s_destroy', method);
calllib(alias, func_name, handle);
end
