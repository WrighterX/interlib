function y = hermiteEvaluate(handle, x)
alias = interlib.internal.hermiteAlias();
x_val = double(x);
out = libpointer('doublePtr', 0);
status = calllib(alias, 'interlib_hermite_eval', handle, x_val, out);
if status ~= 0
    error('interlib:HermiteEvalFailed', '%s', interlib.internal.hermiteLastError());
end
y = out.Value;
end
