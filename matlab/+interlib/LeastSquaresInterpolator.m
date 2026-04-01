classdef LeastSquaresInterpolator < handle
    properties (Access = private)
        Handle
        LibraryPath
        Degree
        IsFitted = false
    end

    methods
        function obj = LeastSquaresInterpolator(degree, libraryPath)
            if nargin < 2 || isempty(libraryPath)
                libraryPath = interlib.internal.leastSquaresLibraryPath();
            end
            if nargin < 1
                degree = 2;
            end

            obj.Degree = degree;
            obj.LibraryPath = string(libraryPath);
            interlib.internal.ensureLeastSquaresLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.leastSquaresCreate(uint32(degree));
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.leastSquaresDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, x, y)
            interlib.internal.leastSquaresFit(obj.Handle, x, y);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.leastSquaresEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.leastSquaresEvaluateMany(obj.Handle, x);
        end

        function degree = getDegree(obj)
            degree = obj.Degree;
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
                fprintf('interlib.LeastSquaresInterpolator(degree=%d, library="%s")\n', obj.Degree, obj.LibraryPath);
            else
                fprintf('interlib.LeastSquaresInterpolator(degree=%d, library="%s")\n', obj.Degree, obj.LibraryPath);
            end
        end
    end
end
