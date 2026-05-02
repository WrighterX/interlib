function y = genericEvaluateMany(handle, x, method)
% GENERIC EVALUATE MANY Evaluate interpolator at multiple points
%
% Parameters:
%   handle - Pointer to the interpolator handle
%   x - Array of x values
%   method - String name of the interpolation method
%
% Returns:
%   y - Array of interpolated values

alias = interlib.internal.nativeAlias();
x_vec = interlib.internal.coerceNumericVector(x);

if isempty(x_vec)
    y = [];
    return;
end

n = numel(x_vec);
out = libpointer('doublePtr', zeros(size(x_vec)));
func_name = sprintf('interlib_%s_eval_many', method);
status = calllib(alias, func_name, handle, x_vec, n, out);
if status ~= 0
    error('interlib:%sEvalManyFailed', '%s', ...
        method, interlib.internal.genericLastError(method));
end

y = reshape(out.Value, size(x));
end
