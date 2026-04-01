function rbfDestroy(handle)
if isempty(handle)
    return;
end
alias = interlib.internal.rbfAlias();
calllib(alias, 'interlib_rbf_destroy', handle);
end
