function path = nativeHeaderPath()
matlab_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
path = fullfile(matlab_dir, 'headers', 'interlib_native.h');
end
