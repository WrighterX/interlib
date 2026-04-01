classdef LinearInterpolator < handle
    % LinearInterpolator MATLAB wrapper around the Rust linear C ABI.
    %
    % This wrapper is intentionally narrow: it exists only for the linear
    % prototype and keeps MATLAB calls at the boundary while the algorithm
    % remains in Rust.

    properties (Access = private)
        Handle
        LibraryPath
        IsFitted = false
    end

    methods
        function obj = LinearInterpolator(libraryPath)
            if nargin < 1 || isempty(libraryPath)
                libraryPath = interlib.internal.linearLibraryPath();
            end

            obj.LibraryPath = string(libraryPath);
            interlib.internal.ensureLinearLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.linearCreate();
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.linearDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, x, y)
            interlib.internal.linearFit(obj.Handle, x, y);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.linearEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.linearEvaluateMany(obj.Handle, x);
        end

        function varargout = subsref(obj, S)
            if numel(S) == 1 && strcmp(S(1).type, '()')
                varargout{1} = obj.evaluate(S(1).subs{1});
                return;
            end

            [varargout{1:nargout}] = builtin('subsref', obj, S);
        end

        function disp(obj)
            if obj.IsFitted
                fprintf('interlib.LinearInterpolator(fitted, library="%s")\n', obj.LibraryPath);
            else
                fprintf('interlib.LinearInterpolator(not fitted, library="%s")\n', obj.LibraryPath);
            end
        end
    end
end
