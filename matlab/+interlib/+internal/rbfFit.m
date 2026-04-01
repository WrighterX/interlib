function rbfFit(handle, x, y)
[x_vec, y_vec] = interlib.internal.coerceMatchingVectors(x, y);
alias = interlib.internal.rbfAlias();
status = calllib(alias, 'interlib_rbf_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:RBFFitFailed', '%s', interlib.internal.rbfLastError());
end
end
