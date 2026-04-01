function y = linearEvaluate(handle, x)
alias = interlib.internal.linearAlias();

if isscalar(x)
    out = libpointer('doublePtr', 0);
    status = calllib(alias, 'interlib_linear_eval', handle, double(x), out);
    if status ~= 0
        error('interlib:LinearEvalFailed', '%s', interlib.internal.linearLastError());
    end
    y = out.Value;
    return;
end

y = interlib.internal.linearEvaluateMany(handle, x);
