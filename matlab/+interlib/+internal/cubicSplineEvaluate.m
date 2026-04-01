function y = cubicSplineEvaluate(handle, x)
alias = interlib.internal.cubicSplineAlias();

if isscalar(x)
    out = libpointer('doublePtr', 0);
    status = calllib(alias, 'interlib_cubic_spline_eval', handle, double(x), out);
    if status ~= 0
        error('interlib:CubicSplineEvalFailed', '%s', interlib.internal.cubicSplineLastError());
    end
    y = out.Value;
    return;
end

y = interlib.internal.cubicSplineEvaluateMany(handle, x);
end
