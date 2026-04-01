function handle = newtonCreate()
alias = interlib.internal.newtonAlias();
handle = calllib(alias, 'interlib_newton_create');
if isempty(handle)
    error('interlib:NewtonCreateFailed', 'Failed to create a newton interpolator handle.');
end
end
