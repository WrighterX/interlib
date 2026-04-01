function handle = leastSquaresCreate(degree)
alias = interlib.internal.leastSquaresAlias();
handle = calllib(alias, 'interlib_least_squares_create', uint32(degree));
if isempty(handle)
    error('interlib:LeastSquaresCreateFailed', 'Failed to create a least squares handle.');
end
end
