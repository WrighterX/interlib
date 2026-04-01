function path = quadraticHeaderPath()
matlab_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
path = fullfile(matlab_dir, 'headers', 'interlib_quadratic.h');
end
