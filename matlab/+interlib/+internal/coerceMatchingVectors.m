function [x_vec, y_vec] = coerceMatchingVectors(x, y)
x_vec = interlib.internal.coerceNumericVector(x);
y_vec = interlib.internal.coerceNumericVector(y);

if numel(x_vec) ~= numel(y_vec)
    error('interlib:LengthMismatch', 'x and y must have the same length.');
end
