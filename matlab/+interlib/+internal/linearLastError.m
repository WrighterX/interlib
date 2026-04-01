function message = linearLastError()
alias = interlib.internal.linearAlias();
buffer_len = 1024;
buffer = libpointer('int8Ptr', zeros(1, buffer_len, 'int8'));
required = calllib(alias, 'interlib_linear_last_error', buffer, buffer_len);
bytes = buffer.Value;

if required <= 1
    message = "";
    return;
end

usable = bytes(1:min(required - 1, numel(bytes)));
message = string(char(usable));
