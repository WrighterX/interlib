function y = rbfEvaluate(handle, x)
alias = interlib.internal.rbfAlias();
out = libpointer('doublePtr', 0);
status = calllib(alias, 'interlib_rbf_eval', handle, double(x), out);
if status ~= 0
    error('interlib:RBFEvaluateFailed', '%s', interlib.internal.rbfLastError());
end
y = out.Value;
end
