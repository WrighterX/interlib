classdef RBFInterpolator < handle
    properties (Access = private)
        Handle
        LibraryPath
        Kernel
        KernelId
        Epsilon
        IsFitted = false
    end

    methods
        function obj = RBFInterpolator(kernel, epsilon, libraryPath)
            if nargin < 3 || isempty(libraryPath)
                libraryPath = interlib.internal.rbfLibraryPath();
            end
            if nargin < 2 || isempty(epsilon)
                epsilon = 1.0;
            end
            if nargin < 1 || isempty(kernel)
                kernel = 'gaussian';
            end

            obj.Kernel = string(kernel);
            obj.KernelId = interlib.internal.rbfKernelId(obj.Kernel);
            obj.Epsilon = double(epsilon);
            obj.LibraryPath = string(libraryPath);

            interlib.internal.ensureRBFLibrary(obj.LibraryPath);
            obj.Handle = interlib.internal.rbfCreate(obj.KernelId, obj.Epsilon);
        end

        function delete(obj)
            if ~isempty(obj.Handle)
                interlib.internal.rbfDestroy(obj.Handle);
                obj.Handle = [];
            end
        end

        function fit(obj, x, y)
            interlib.internal.rbfFit(obj.Handle, x, y);
            obj.IsFitted = true;
        end

        function y = evaluate(obj, x)
            y = interlib.internal.rbfEvaluate(obj.Handle, x);
        end

        function y = evaluateMany(obj, x)
            y = interlib.internal.rbfEvaluateMany(obj.Handle, x);
        end

        function weights = get_weights(obj)
            weights = interlib.internal.rbfWeights(obj.Handle);
        end

        function varargout = subsref(obj, S)
            if numel(S) == 1 && strcmp(S(1).type, '()')
                varargout{1} = obj.evaluate(S(1).subs{1});
                return;
            end
            [varargout{1:nargout}] = builtin('subsref', obj, S);
        end

        function disp(obj)
            state = 'not fitted';
            if obj.IsFitted
                state = 'fitted';
            end
            fprintf('interlib.RBFInterpolator(kernel="%s", epsilon=%.2f, library="%s", %s)\n', ...
                obj.Kernel, obj.Epsilon, obj.LibraryPath, state);
        end
    end
end
