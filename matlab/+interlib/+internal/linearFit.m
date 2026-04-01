function linearFit(handle, x, y)
[x_vec, y_vec] = interlib.internal.coerceMatchingVectors(x, y);
alias = interlib.internal.linearAlias();
status = calllib(alias, 'interlib_linear_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:LinearFitFailed', '%s', interlib.internal.linearLastError());
end
