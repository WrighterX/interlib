function chebyshevDestroy(handle)
if isempty(handle)
    return;
end
alias = interlib.internal.chebyshevAlias();
calllib(alias, 'interlib_chebyshev_destroy', handle);
end
