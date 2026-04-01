function y = linearEvaluateMany(handle, x)
x_vec = interlib.internal.coerceNumericVector(x);
alias = interlib.internal.linearAlias();
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_linear_eval_many', handle, x_vec, numel(x_vec), out);
if status ~= 0
    error('interlib:LinearEvalManyFailed', '%s', interlib.internal.linearLastError());
end

y = reshape(out.Value, size(x_vec));
