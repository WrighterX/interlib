function y = rbfEvaluateMany(handle, x)
x_vec = interlib.internal.coerceNumericVector(x);
alias = interlib.internal.rbfAlias();
out = libpointer('doublePtr', zeros(size(x_vec)));
status = calllib(alias, 'interlib_rbf_eval_many', handle, x_vec, numel(x_vec), out);
if status ~= 0
    error('interlib:RBFEvaluateManyFailed', '%s', interlib.internal.rbfLastError());
end
y = reshape(out.Value, size(x_vec));
end
