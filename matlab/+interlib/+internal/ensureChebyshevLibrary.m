function ensureChebyshevLibrary(libraryPath)
alias = interlib.internal.chebyshevAlias();
required_symbol = 'interlib_chebyshev_create';

if libisloaded(alias)
    functions = libfunctions(alias);
    if any(strcmp(functions, required_symbol))
        return;
    end
    unloadlibrary(alias);
end

interlib.internal.ensureNativeLibrary(libraryPath);
end
