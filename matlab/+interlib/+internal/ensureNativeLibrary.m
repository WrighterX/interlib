function ensureNativeLibrary(libraryPath)
alias = interlib.internal.nativeAlias();
if libisloaded(alias)
    return;
end

headerPath = interlib.internal.nativeHeaderPath();
if ~isfile(libraryPath)
    error('interlib:NativeLibraryMissing', 'Shared library not found: %s', libraryPath);
end
if ~isfile(headerPath)
    error('interlib:NativeHeaderMissing', 'Header file not found: %s', headerPath);
end

loadlibrary(char(libraryPath), char(headerPath), 'alias', alias);
end
