function cubicSplineFit(handle, x, y)
alias = interlib.internal.cubicSplineAlias();
[x_vec, y_vec] = interlib.internal.coerceMatchingVectors(x, y);

status = calllib(alias, 'interlib_cubic_spline_fit', handle, x_vec, numel(x_vec), y_vec, numel(y_vec));
if status ~= 0
    error('interlib:CubicSplineFitFailed', '%s', interlib.internal.cubicSplineLastError());
end
end
