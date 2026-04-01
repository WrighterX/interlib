classdef QuadraticInterpolator < handle
    % QuadraticInterpolator MATLAB wrapper around the Rust Quadratic C ABI.

    properties (Access = private)
        Handle
        LibraryPath
        IsFitted = false
    end

    methods
        function obj = QuadraticInterpolator(libraryPath)
            if nargin < 1 || isempty(libraryPath)
                libraryPath = interlib.internal.quadraticLibraryPath();
            end

            obj.LibraryPath = string(libraryPath);
            interlib.internal.ensureQuadraticLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.quadraticCreate();
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.quadraticDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, x, y)
            interlib.internal.quadraticFit(obj.Handle, x, y);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.quadraticEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.quadraticEvaluateMany(obj.Handle, x);
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
                fprintf('interlib.QuadraticInterpolator(fitted, library="%s")\n', obj.LibraryPath);
            else
                fprintf('interlib.QuadraticInterpolator(not fitted, library="%s")\n', obj.LibraryPath);
            end
        end
    end
end
