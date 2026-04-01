function values = coerceNumericVector(values)
if ~isnumeric(values) || ~isvector(values)
    error('interlib:InvalidInput', 'Input must be a numeric vector.');
end

values = double(values(:));
