function genericUpdateY(handle, y, method)
% GENERICUPDATEY Update y values for interpolator
%
% Parameters:
%   handle - Pointer to the interpolator handle
%   y - New array of y values
%   method - String name of the interpolation method

alias = interlib.internal.nativeAlias();
y_vec = interlib.internal.coerceNumericVector(y);
func_name = sprintf('interlib_%s_update_y', method);
status = calllib(alias, func_name, handle, y_vec, numel(y_vec));
if status ~= 0
    error('interlib:%sUpdateFailed', '%s', ...
        method, interlib.internal.genericLastError(method));
end
end
