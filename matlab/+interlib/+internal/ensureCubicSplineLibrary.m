function ensureCubicSplineLibrary(libraryPath)
alias = interlib.internal.nativeAlias();
required_symbol = 'interlib_cubic_spline_create';

if libisloaded(alias)
    functions = libfunctions(alias);
    if any(strcmp(functions, required_symbol))
        return;
    end
    unloadlibrary(alias);
end

interlib.internal.ensureNativeLibrary(libraryPath);
end
