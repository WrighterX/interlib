classdef CubicSplineInterpolator < handle
    % CubicSplineInterpolator MATLAB wrapper around the Rust Cubic Spline C ABI.

    properties (Access = private)
        Handle
        LibraryPath
        IsFitted = false
    end

    methods
        function obj = CubicSplineInterpolator(libraryPath)
            if nargin < 1 || isempty(libraryPath)
                libraryPath = interlib.internal.cubicSplineLibraryPath();
            end

            obj.LibraryPath = string(libraryPath);
            interlib.internal.ensureCubicSplineLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.cubicSplineCreate();
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.cubicSplineDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, x, y)
            interlib.internal.cubicSplineFit(obj.Handle, x, y);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.cubicSplineEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.cubicSplineEvaluateMany(obj.Handle, x);
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
                fprintf('interlib.CubicSplineInterpolator(fitted, library="%s")\n', obj.LibraryPath);
            else
                fprintf('interlib.CubicSplineInterpolator(not fitted, library="%s")\n', obj.LibraryPath);
            end
        end
    end
end
