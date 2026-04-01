function hermiteDestroy(handle)
alias = interlib.internal.hermiteAlias();
calllib(alias, 'interlib_hermite_destroy', handle);
end
