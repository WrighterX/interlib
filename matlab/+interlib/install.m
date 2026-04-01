function result = install()
%INSTALL Verify the MATLAB installation and print a short status message.
%
% This is a convenience entrypoint intended for toolbox users. It delegates to
% verify_installation so the install flow stays one command.

result = interlib.verify_installation();
end
