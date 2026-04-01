function y = chebyshevEvaluateMany(handle, x)
x_vec = interlib.internal.coerceNumericVector(x);
alias = interlib.internal.chebyshevAlias();
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_chebyshev_eval_many', handle, x_vec, numel(x_vec), out);
if status ~= 0
    error('interlib:ChebyshevEvalManyFailed', '%s', interlib.internal.chebyshevLastError());
end
y = reshape(out.Value, size(x_vec));
end
