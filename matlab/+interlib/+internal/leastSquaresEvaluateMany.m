function y = leastSquaresEvaluateMany(handle, x)
alias = interlib.internal.leastSquaresAlias();
x_vec = interlib.internal.coerceNumericVector(x);

if isempty(x_vec)
    y = [];
    return;
end

n = numel(x_vec);
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_least_squares_eval_many', handle, x_vec, n, out);
if status ~= 0
    error('interlib:LeastSquaresEvalManyFailed', '%s', interlib.internal.leastSquaresLastError());
end

y = reshape(out.Value, size(x));
end
