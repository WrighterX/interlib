function y = linearEvaluateMany(handle, x)
alias = interlib.internal.linearAlias();
x_vec = interlib.internal.coerceNumericVector(x);

if isempty(x_vec)
    y = [];
    return;
end

n = numel(x_vec);
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_linear_eval_many', handle, x_vec, n, out);
if status ~= 0
    error('interlib:LinearEvalManyFailed', '%s', interlib.internal.linearLastError());
end

y = reshape(out.Value, size(x));
end
