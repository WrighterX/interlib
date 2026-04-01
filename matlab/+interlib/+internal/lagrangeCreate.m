function handle = lagrangeCreate()
alias = interlib.internal.lagrangeAlias();
handle = calllib(alias, 'interlib_lagrange_create');
if isempty(handle)
    error('interlib:LagrangeCreateFailed', 'Failed to create a Lagrange interpolator handle.');
end
end
