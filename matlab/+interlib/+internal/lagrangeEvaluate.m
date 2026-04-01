function y = lagrangeEvaluate(handle, x)
alias = interlib.internal.lagrangeAlias();
out = libpointer('doublePtr', 0);
status = calllib(alias, 'interlib_lagrange_eval', handle, double(x), out);
if status ~= 0
    error('interlib:LagrangeEvalFailed', '%s', interlib.internal.lagrangeLastError());
end
y = out.Value;
end
