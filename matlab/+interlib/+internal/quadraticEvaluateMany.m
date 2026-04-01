function y = quadraticEvaluateMany(handle, x)
alias = interlib.internal.quadraticAlias();
x_vec = interlib.internal.coerceNumericVector(x);

if isempty(x_vec)
    y = [];
    return;
end

n = numel(x_vec);
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_quadratic_eval_many', handle, x_vec, n, out);
if status ~= 0
    error('interlib:QuadraticEvalManyFailed', '%s', interlib.internal.quadraticLastError());
end

y = reshape(out.Value, size(x));
end
