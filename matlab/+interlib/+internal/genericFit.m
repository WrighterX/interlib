function genericFit(handle, x, y, method)
% GENERICFIT Fit interpolator to data points
%
% Parameters:
%   handle - Pointer to the interpolator handle
%   x - Array of x values
%   y - Array of y values
%   method - String name of the interpolation method

[x_vec, y_vec] = interlib.internal.coerceMatchingVectors(x, y);
alias = interlib.internal.nativeAlias();
func_name = sprintf('interlib_%s_fit', method);
status = calllib(alias, func_name, handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:%sFitFailed', '%s', ...
        method, interlib.internal.genericLastError(method));
end
end
