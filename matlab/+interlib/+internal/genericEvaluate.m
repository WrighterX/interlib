function y = genericEvaluate(handle, x, method)
% GENERICEVALUATE Evaluate interpolator at a single point or array
%
% Parameters:
%   handle - Pointer to the interpolator handle
%   x - Scalar or array of x values
%   method - String name of the interpolation method
%
% Returns:
%   y - Interpolated value(s)

alias = interlib.internal.nativeAlias();

if isscalar(x)
    out = libpointer('doublePtr', 0);
    func_name = sprintf('interlib_%s_eval', method);
    status = calllib(alias, func_name, handle, double(x), out);
    if status ~= 0
        error('interlib:%sEvalFailed', '%s', ...
            method, interlib.internal.genericLastError(method));
    end
    y = out.Value;
    return;
end

y = interlib.internal.genericEvaluateMany(handle, x, method);
end
