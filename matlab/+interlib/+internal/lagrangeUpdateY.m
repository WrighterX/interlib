function lagrangeUpdateY(handle, y)
alias = interlib.internal.lagrangeAlias();
y_vec = interlib.internal.coerceNumericVector(y);
status = calllib(alias, 'interlib_lagrange_update_y', handle, y_vec, numel(y_vec));
if status ~= 0
    error('interlib:LagrangeUpdateFailed', '%s', interlib.internal.lagrangeLastError());
end
end
