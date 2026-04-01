function ensureQuadraticLibrary(libraryPath)
alias = interlib.internal.quadraticAlias();
if libisloaded(alias)
    return;
end

headerPath = interlib.internal.quadraticHeaderPath();
if ~isfile(libraryPath)
    error('interlib:QuadraticLibraryMissing', 'Shared library not found: %s', libraryPath);
end
if ~isfile(headerPath)
    error('interlib:QuadraticHeaderMissing', 'Header file not found: %s', headerPath);
end

loadlibrary(char(libraryPath), char(headerPath), 'alias', alias);
end
