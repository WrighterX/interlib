function chebyshevFit(handle, y)
alias = interlib.internal.chebyshevAlias();
y_vec = interlib.internal.coerceNumericVector(y);
status = calllib(alias, 'interlib_chebyshev_fit', handle, y_vec, numel(y_vec));
if status ~= 0
    error('interlib:ChebyshevFitFailed', '%s', interlib.internal.chebyshevLastError());
end
end
