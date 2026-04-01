function weights = rbfWeights(handle)
alias = interlib.internal.rbfAlias();
len = interlib.internal.rbfCount(handle);
if len == 0
    weights = [];
    return;
end

out = libpointer('doublePtr', zeros(len, 1));
status = calllib(alias, 'interlib_rbf_weights', handle, out, len);
if status ~= 0
    error('interlib:RBFWeightsFailed', '%s', interlib.internal.rbfLastError());
end

weights = reshape(out.Value, len, 1);
end
