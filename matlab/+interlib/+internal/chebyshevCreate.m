function handle = chebyshevCreate(n_points, x_min, x_max, use_clenshaw)
alias = interlib.internal.chebyshevAlias();
handle = calllib(alias, 'interlib_chebyshev_create', uint64(n_points), double(x_min), double(x_max), int32(use_clenshaw));
if isempty(handle)
    error('interlib:ChebyshevCreateFailed', 'Failed to create Chebyshev interpolator handle.');
end
end
