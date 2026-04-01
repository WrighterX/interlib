function id = rbfKernelId(kernel)
if nargin < 1 || isempty(kernel)
    kernel = 'gaussian';
end
kernel = lower(string(kernel));
switch char(kernel)
    case 'gaussian'
        id = int32(0);
    case 'multiquadric'
        id = int32(1);
    case {'inverse_multiquadric', 'inverse multiquadric'}
        id = int32(2);
    case {'thin_plate_spline', 'thin plate spline'}
        id = int32(3);
    case 'linear'
        id = int32(4);
    otherwise
        error('interlib:RBFUnknownKernel', 'Unknown kernel: %s', kernel);
end
end
