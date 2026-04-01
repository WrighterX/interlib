function cubicSplineDestroy(handle)
alias = interlib.internal.cubicSplineAlias();
calllib(alias, 'interlib_cubic_spline_destroy', handle);
end
