function y = cubicSplineEvaluate(handle, x)
y = interlib.internal.genericEvaluate(handle, x, 'cubic_spline');
end
