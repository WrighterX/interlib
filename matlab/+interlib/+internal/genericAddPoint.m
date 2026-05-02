function genericAddPoint(handle, x_new, y_new, method)
% GENERICADDPOINT Add a new data point to the interpolator
%
% Parameters:
%   handle - Pointer to the interpolator handle
%   x_new - New x value
%   y_new - New y value
%   method - String name of the interpolation method

alias = interlib.internal.nativeAlias();
func_name = sprintf('interlib_%s_add_point', method);
status = calllib(alias, func_name, handle, double(x_new), double(y_new));
if status ~= 0
    error('interlib:%sAddPointFailed', '%s', ...
        method, interlib.internal.genericLastError(method));
end
end
