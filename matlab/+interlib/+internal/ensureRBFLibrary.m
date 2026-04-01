function ensureRBFLibrary(libraryPath)
alias = interlib.internal.rbfAlias();
required_symbol = 'interlib_rbf_create';

if libisloaded(alias)
    functions = libfunctions(alias);
    if any(strcmp(functions, required_symbol))
        return;
    end
    unloadlibrary(alias);
end

interlib.internal.ensureNativeLibrary(libraryPath);
end
