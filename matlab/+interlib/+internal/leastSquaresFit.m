function leastSquaresFit(handle, x, y)
alias = interlib.internal.leastSquaresAlias();
x_vec = interlib.internal.coerceNumericVector(x);
y_vec = interlib.internal.coerceNumericVector(y);
status = calllib(alias, 'interlib_least_squares_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:LeastSquaresFitFailed', '%s', interlib.internal.leastSquaresLastError());
end
end
