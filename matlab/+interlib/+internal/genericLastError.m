function message = genericLastError(method)
% GENERICLASTERROR Get the last error message for any method
%
% Parameters:
%   method - String name of the interpolation method
%
% Returns:
%   message - Last error message string

alias = interlib.internal.nativeAlias();
len = calllib(alias, 'interlib_last_error', libpointer('charPtr', []), 0);
if len <= 1
    message = 'No error message available.';
    return;
end

buffer = zeros(1, len, 'int8');
calllib(alias, 'interlib_last_error', buffer, len);
message = char(buffer(1:end-1));
end
