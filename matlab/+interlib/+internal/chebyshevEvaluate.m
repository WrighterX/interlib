function y = chebyshevEvaluate(handle, x)
alias = interlib.internal.chebyshevAlias();
out = libpointer('doublePtr', 0);
status = calllib(alias, 'interlib_chebyshev_eval', handle, double(x), out);
if status ~= 0
    error('interlib:ChebyshevEvalFailed', '%s', interlib.internal.chebyshevLastError());
end
y = out.Value;
end
