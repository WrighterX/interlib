function lagrangeFit(handle, x, y)
alias = interlib.internal.lagrangeAlias();
x_vec = interlib.internal.coerceNumericVector(x);
y_vec = interlib.internal.coerceNumericVector(y);
status = calllib(alias, 'interlib_lagrange_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:LagrangeFitFailed', '%s', interlib.internal.lagrangeLastError());
end
end
