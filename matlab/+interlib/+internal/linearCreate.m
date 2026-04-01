function handle = linearCreate()
alias = interlib.internal.linearAlias();
handle = calllib(alias, 'interlib_linear_create');
if isempty(handle)
    error('interlib:LinearCreateFailed', 'Failed to create a linear interpolator handle.');
end
