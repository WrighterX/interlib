function lagrangeAddPoint(handle, x_new, y_new)
alias = interlib.internal.lagrangeAlias();
status = calllib(alias, 'interlib_lagrange_add_point', handle, double(x_new), double(y_new));
if status ~= 0
    error('interlib:LagrangeAddPointFailed', '%s', interlib.internal.lagrangeLastError());
end
end
