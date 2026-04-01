function y = newtonEvaluate(handle, x)
alias = interlib.internal.newtonAlias();

if isscalar(x)
    out = libpointer('doublePtr', 0);
    status = calllib(alias, 'interlib_newton_eval', handle, double(x), out);
    if status ~= 0
        error('interlib:NewtonEvalFailed', '%s', interlib.internal.newtonLastError());
    end
    y = out.Value;
    return;
end

y = interlib.internal.newtonEvaluateMany(handle, x);
end
