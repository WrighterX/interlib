function handle = hermiteCreate()
alias = interlib.internal.hermiteAlias();
handle = calllib(alias, 'interlib_hermite_create');
if isempty(handle)
    error('interlib:HermiteCreateFailed', 'Failed to create a Hermite interpolator handle.');
end
end
