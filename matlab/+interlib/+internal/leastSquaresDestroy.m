function leastSquaresDestroy(handle)
alias = interlib.internal.leastSquaresAlias();
calllib(alias, 'interlib_least_squares_destroy', handle);
end
