function y = quadraticEvaluate(handle, x)
alias = interlib.internal.quadraticAlias();

if isscalar(x)
    out = libpointer('doublePtr', 0);
    status = calllib(alias, 'interlib_quadratic_eval', handle, double(x), out);
    if status ~= 0
        error('interlib:QuadraticEvalFailed', '%s', interlib.internal.quadraticLastError());
    end
    y = out.Value;
    return;
end

y = interlib.internal.quadraticEvaluateMany(handle, x);
end
