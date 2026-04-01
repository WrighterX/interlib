classdef NewtonInterpolator < handle
    % NewtonInterpolator MATLAB wrapper around the Rust Newton C ABI.

    properties (Access = private)
        Handle
        LibraryPath
        IsFitted = false
    end

    methods
        function obj = NewtonInterpolator(libraryPath)
            if nargin < 1 || isempty(libraryPath)
                libraryPath = interlib.internal.newtonLibraryPath();
            end

            obj.LibraryPath = string(libraryPath);
            interlib.internal.ensureNewtonLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.newtonCreate();
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.newtonDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, x, y)
            interlib.internal.newtonFit(obj.Handle, x, y);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.newtonEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.newtonEvaluateMany(obj.Handle, x);
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
                fprintf('interlib.NewtonInterpolator(fitted, library="%s")\n', obj.LibraryPath);
            else
                fprintf('interlib.NewtonInterpolator(not fitted, library="%s")\n', obj.LibraryPath);
            end
        end
    end
end
