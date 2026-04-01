function newtonDestroy(handle)
alias = interlib.internal.newtonAlias();
calllib(alias, 'interlib_newton_destroy', handle);
end
