function count = rbfCount(handle)
alias = interlib.internal.rbfAlias();
count = calllib(alias, 'interlib_rbf_count', handle);
end
