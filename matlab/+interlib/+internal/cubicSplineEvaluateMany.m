function y = cubicSplineEvaluateMany(handle, x)
y = interlib.internal.genericEvaluateMany(handle, x, 'cubic_spline');
end
