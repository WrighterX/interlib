function message = leastSquaresLastError()
alias = interlib.internal.leastSquaresAlias();
len = calllib(alias, 'interlib_least_squares_last_error', libpointer('charPtr', []), 0);
if len <= 1
    message = 'No error message available.';
    return;
end

buffer = zeros(1, len, 'int8');
calllib(alias, 'interlib_least_squares_last_error', buffer, len);
message = char(buffer(1:end-1));
end
