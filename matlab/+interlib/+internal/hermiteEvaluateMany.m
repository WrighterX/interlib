function y = hermiteEvaluateMany(handle, x)
alias = interlib.internal.hermiteAlias();
x_vec = interlib.internal.coerceNumericVector(x);

if isempty(x_vec)
    y = [];
    return;
end

n = numel(x_vec);
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_hermite_eval_many', handle, x_vec, n, out);
if status ~= 0
    error('interlib:HermiteEvalManyFailed', '%s', interlib.internal.hermiteLastError());
end

y = reshape(out.Value, size(x));
end
