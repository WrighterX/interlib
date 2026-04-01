function linearDestroy(handle)
if isempty(handle)
    return;
end

alias = interlib.internal.linearAlias();
calllib(alias, 'interlib_linear_destroy', handle);
