function result = verify_installation()
%VERIFY_INSTALLATION Check that the MATLAB wrapper and native library load.
%
% This is the toolbox-facing smoke test. It is intentionally small and
% explicit so users can run it after installation and see a clear failure if
% the MATLAB path or native shared library is not configured correctly.

result = struct('ok', false, 'message', "", 'details', struct());

try
    packageRoot = fileparts(fileparts(mfilename('fullpath')));
    result.details.packageRoot = string(packageRoot);

    packageVisible = exist('interlib.LinearInterpolator', 'class') == 8;
    result.details.packageVisible = packageVisible;
    if ~packageVisible
        error('interlib:InstallationMissingPackage', ...
            'The interlib MATLAB package is not on the path. Add %s to the MATLAB path.', ...
            packageRoot);
    end

    libraryPath = interlib.internal.nativeLibraryPath();
    result.details.libraryPath = string(libraryPath);

    if ~isfile(libraryPath)
        error('interlib:InstallationMissingLibrary', ...
            'Shared library not found: %s', libraryPath);
    end

    interlib.internal.ensureNativeLibrary(libraryPath);
    result.details.nativeAlias = string(interlib.internal.nativeAlias());
    result.details.nativeLoaded = libisloaded(interlib.internal.nativeAlias());

    interp = interlib.LinearInterpolator(libraryPath);
    interp.fit([0; 1; 2], [0; 1; 4]);

    value = interp.evaluate(0.5);
    result.details.sampleValue = value;

    if abs(value - 0.5) > 1e-12
        error('interlib:InstallationVerificationFailed', ...
            'Unexpected verification result: %.16g', value);
    end

    result.ok = true;
    result.message = "interlib MATLAB installation verified.";
    fprintf('%s\n', result.message);
catch err
    result.message = string(err.message);
    fprintf(2, 'interlib installation verification failed: %s\n', err.message);
end
end
