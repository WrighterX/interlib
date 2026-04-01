function handle = rbfCreate(kernelId, epsilon)
alias = interlib.internal.rbfAlias();
handle = calllib(alias, 'interlib_rbf_create', int32(kernelId), double(epsilon));
if isempty(handle)
    error('interlib:RBFCreateFailed', 'Failed to create a new RBF interpolator handle.');
end
end
