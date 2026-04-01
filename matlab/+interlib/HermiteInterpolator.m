classdef HermiteInterpolator < handle
    properties (Access = private)
        Handle
        LibraryPath
        IsFitted = false
    end

    methods
        function obj = HermiteInterpolator(libraryPath)
            if nargin < 1 || isempty(libraryPath)
                libraryPath = interlib.internal.hermiteLibraryPath();
            end

            obj.LibraryPath = string(libraryPath);
            interlib.internal.ensureHermiteLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.hermiteCreate();
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.hermiteDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, x, y, dy)
            interlib.internal.hermiteFit(obj.Handle, x, y, dy);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.hermiteEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.hermiteEvaluateMany(obj.Handle, x);
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
                fprintf('interlib.HermiteInterpolator(fitted, library="%s")\n', obj.LibraryPath);
            else
                fprintf('interlib.HermiteInterpolator(not fitted, library="%s")\n', obj.LibraryPath);
            end
        end
    end
end
