function y = leastSquaresEvaluate(handle, x)
alias = interlib.internal.leastSquaresAlias();
out = libpointer('doublePtr', 0);
status = calllib(alias, 'interlib_least_squares_eval', handle, double(x), out);
if status ~= 0
    error('interlib:LeastSquaresEvalFailed', '%s', interlib.internal.leastSquaresLastError());
end
y = out.Value;
end
