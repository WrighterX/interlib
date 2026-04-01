function handle = cubicSplineCreate()
alias = interlib.internal.cubicSplineAlias();
handle = calllib(alias, 'interlib_cubic_spline_create');
if isempty(handle)
    error('interlib:CubicSplineCreateFailed', 'Failed to create a cubic spline interpolator handle.');
end
end
