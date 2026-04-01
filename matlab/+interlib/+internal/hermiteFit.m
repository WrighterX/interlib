function hermiteFit(handle, x, y, dy)
alias = interlib.internal.hermiteAlias();
x_vec = interlib.internal.coerceNumericVector(x);
y_vec = interlib.internal.coerceNumericVector(y);
dy_vec = interlib.internal.coerceNumericVector(dy);
status = calllib(alias, 'interlib_hermite_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec), dy_vec, numel(dy_vec));
if status ~= 0
    error('interlib:HermiteFitFailed', '%s', interlib.internal.hermiteLastError());
end
end
