% 示例：JSON 接口端到端。
%
% 生成一份单频带单模态的合成数据，写描述文件，调用 bayoma.io_json，
% 再读回结果并与直接调用 bayoma.bayoma 的结果比对。
%
% 运行：
%   matlab -batch "addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab'))); run('examples\matlab_json_job.m')"

workDir = fullfile(tempdir, 'bayoma_json_demo');
if exist(workDir, 'dir') ~= 7
    mkdir(workDir);
end

fs = 100.0;
dur = 1200.0;
tdata = bayoma.simulate_oma_data([1.0], [0.01], [1; 2; 2] / 3, 1.0, 1.0, ...
    fs, dur, struct('seed', 7));

dataPath = fullfile(workDir, 'data.csv');
dlmwrite(dataPath, tdata, 'delimiter', ',', 'precision', '%.10g');

spec = struct();
spec.data = struct('path', dataPath, 'format', 'csv', 'fs', fs);
spec.bands = [0.9 1.1];
spec.f0 = {{1.0}};
spec.options = struct('mute', 1);
spec.output = struct('path', fullfile(workDir, 'result.json'));

specPath = fullfile(workDir, 'job.json');
fid = fopen(specPath, 'w', 'n', 'UTF-8');
fwrite(fid, jsonencode(spec), 'char');
fclose(fid);

result = bayoma.io_json(specPath);

fprintf('结果文件字段\n');
disp(fieldnames(result));
fprintf('f = %s\n', mat2str(result.f, 6));
fprintf('z = %s\n', mat2str(result.z, 6));
fprintf('Sii = %s\n', mat2str(result.Sii, 6));
fprintf('Se = %s\n', mat2str(result.Se, 6));
fprintf('振型 = %s\n', mat2str(result.phi(:).', 6));
fprintf('n_modes = %d，耗时 %.3f s\n', result.n_modes, result.elapsed);

% 与直接调用比对
IN = struct('tdata', tdata, 'fs', fs, 'f1f2', [0.9 1.1], 'f0', {{1.0}}, 'mute', 1);
OUT = bayoma.bayoma(IN);
fprintf('\n与直接调用的差\n');
fprintf('  |df| = %.3e，|dz| = %.3e，|dSii| = %.3e\n', ...
    abs(result.f(1) - OUT.f(1)), abs(result.z(1) - OUT.z(1)), ...
    abs(result.Sii(1) - OUT.Sii(1)));
