function pythonSetUp(pyExec)
% Sets the python environment for MATLAB, cross-platform.
% pyExec = directory where the python executable is located
%          e.g., Windows: 'C:\Users\Name\miniconda3\envs\myenv\'
%                Linux:   '/home/name/anaconda3/envs/myenv/'

% Determine platform
isWindows = ispc;

% Build python executable path
if isWindows
    pythonBin = fullfile(pyExec, 'python.exe');
else
    pythonBin = fullfile(pyExec, 'bin', 'python');
end

% Set python environment if not already loaded
pythonEnv = pyenv();
if ~strcmp(pythonEnv.Status,'Loaded')
    pyenv('Version', pythonBin);
elseif strcmp(pythonEnv.Status,'Loaded') &&...
        ~strcmp(pythonEnv.Home, pyExec)
    disp("A different python environment has already been loaded.");
    disp("To change it, restart MATLAB, and then call 'pyenv'.");
end

% Add necessary directories to environment variable PATH
p = getenv('PATH');
p = strsplit(p, pathsep);

if isWindows
    pyRoot = fileparts(pyExec);
    addToPath = {
       pyRoot
       fullfile(pyRoot, 'Library', 'mingw-w64', 'bin')
       fullfile(pyRoot, 'Library', 'usr', 'bin')
       fullfile(pyRoot, 'Library', 'bin')
       fullfile(pyRoot, 'Scripts')
       fullfile(pyRoot, 'bin')
       fullfile(pyRoot, 'condabin')
    };
else
    addToPath = {
        fullfile(pyExec, 'bin')
    };
end

p = [addToPath(:); p(:)];
p = unique(p, 'stable');
p = strjoin(p, pathsep);
setenv('PATH', p);

end
