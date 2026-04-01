function newtonFit(handle, x, y)
alias = interlib.internal.newtonAlias();
[x_vec, y_vec] = interlib.internal.coerceMatchingVectors(x, y);

status = calllib(alias, 'interlib_newton_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:NewtonFitFailed', '%s', interlib.internal.newtonLastError());
end
end
