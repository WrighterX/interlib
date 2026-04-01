function y = lagrangeEvaluateMany(handle, x)
alias = interlib.internal.lagrangeAlias();
x_vec = interlib.internal.coerceNumericVector(x);

if isempty(x_vec)
    y = [];
    return;
end

n = numel(x_vec);
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_lagrange_eval_many', handle, x_vec, n, out);
if status ~= 0
    error('interlib:LagrangeEvalManyFailed', '%s', interlib.internal.lagrangeLastError());
end

y = reshape(out.Value, size(x));
end
