function quadraticFit(handle, x, y)
alias = interlib.internal.quadraticAlias();
[x_vec, y_vec] = interlib.internal.coerceMatchingVectors(x, y);

status = calllib(alias, 'interlib_quadratic_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:QuadraticFitFailed', '%s', interlib.internal.quadraticLastError());
end
end
