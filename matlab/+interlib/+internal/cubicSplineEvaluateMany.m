function y = cubicSplineEvaluateMany(handle, x)
alias = interlib.internal.cubicSplineAlias();
x_vec = interlib.internal.coerceNumericVector(x);

if isempty(x_vec)
    y = [];
    return;
end

n = numel(x_vec);
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_cubic_spline_eval_many', handle, x_vec, n, out);
if status ~= 0
    error('interlib:CubicSplineEvalManyFailed', '%s', interlib.internal.cubicSplineLastError());
end

y = reshape(out.Value, size(x));
end
