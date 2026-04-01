function lagrangeDestroy(handle)
alias = interlib.internal.lagrangeAlias();
calllib(alias, 'interlib_lagrange_destroy', handle);
end
