classdef ChebyshevInterpolator < handle
    properties (Access = private)
        Handle
        LibraryPath
        NPoints
        XMin
        XMax
        UseClenshaw
        IsFitted = false
    end

    methods
        function obj = ChebyshevInterpolator(n_points, x_min, x_max, use_clenshaw, libraryPath)
            if nargin < 5 || isempty(libraryPath)
                libraryPath = interlib.internal.chebyshevLibraryPath();
            end
            if nargin < 4 || isempty(use_clenshaw)
                use_clenshaw = true;
            end
            if nargin < 3 || isempty(x_max)
                x_max = 1.0;
            end
            if nargin < 2 || isempty(x_min)
                x_min = -1.0;
            end
            if nargin < 1 || isempty(n_points)
                n_points = 10;
            end

            obj.NPoints = uint64(n_points);
            obj.XMin = double(x_min);
            obj.XMax = double(x_max);
            obj.UseClenshaw = logical(use_clenshaw);
            obj.LibraryPath = string(libraryPath);

            interlib.internal.ensureChebyshevLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.chebyshevCreate(obj.NPoints, obj.XMin, obj.XMax, obj.UseClenshaw);
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.chebyshevDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, y)
            interlib.internal.chebyshevFit(obj.Handle, y);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.chebyshevEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.chebyshevEvaluateMany(obj.Handle, x);
        end

        function varargout = subsref(obj, S)
            if numel(S) == 1 && strcmp(S(1).type, '()')
                varargout{1} = obj.evaluate(S(1).subs{1});
                return;
            end
            [varargout{1:nargout}] = builtin('subsref', obj, S);
        end

        function disp(obj)
            method = ternary(obj.UseClenshaw, 'Clenshaw', 'Direct');
            state = ternary(obj.IsFitted, 'fitted', 'not fitted');
            fprintf('interlib.ChebyshevInterpolator(n_points=%d, x_range=[%.2f, %.2f], method=%s, library="%s", %s)\\n', ...
                obj.NPoints, obj.XMin, obj.XMax, method, obj.LibraryPath, state);
        end
    end
end

function out = ternary(cond, trueVal, falseVal)
    if cond
        out = trueVal;
    else
        out = falseVal;
    end
end
