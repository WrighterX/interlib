function ensureLinearLibrary(libraryPath)
alias = interlib.internal.linearAlias();
if libisloaded(alias)
    return;
end

headerPath = interlib.internal.linearHeaderPath();
if ~isfile(libraryPath)
    error('interlib:LinearLibraryMissing', 'Shared library not found: %s', libraryPath);
end
if ~isfile(headerPath)
    error('interlib:LinearHeaderMissing', 'Header file not found: %s', headerPath);
end

loadlibrary(char(libraryPath), char(headerPath), 'alias', alias);
