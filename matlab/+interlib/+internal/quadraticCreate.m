function handle = quadraticCreate()
alias = interlib.internal.quadraticAlias();
handle = calllib(alias, 'interlib_quadratic_create');
if isempty(handle)
    error('interlib:QuadraticCreateFailed', 'Failed to create a quadratic interpolator handle.');
end
end
