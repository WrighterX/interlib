function handle = genericCreate(method)
% GENERICCREATE Create a new interpolator handle for any method
%
% Parameters:
%   method - String name of the interpolation method (e.g., 'linear', 'newton')
%
% Returns:
%   handle - Pointer to the interpolator handle

alias = interlib.internal.nativeAlias();
func_name = sprintf('interlib_%s_create', method);
handle = calllib(alias, func_name);
if isempty(handle)
    error('interlib:%sCreateFailed', 'Failed to create a %s interpolator handle.', ...
        method, method);
end
end
