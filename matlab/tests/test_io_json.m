function test_io_json()
%TEST_IO_JSON 校核 JSON 接口层：写描述文件、跑 io_json、读回结果并与直接调用比对。
%
%   校核内容
%     1. 结果文件能否按 UTF-8 写出并被读回；
%     2. 结果字段是否与 bayoma.bayoma 的直接调用一致；
%     3. 描述文件中的频带、初值、选项是否被正确采用。

here = fileparts(mfilename('fullpath'));
root = fileparts(here);

% 校核过程产生的数据与结果写入系统临时目录，不污染工程目录
workDir = fullfile(tempdir, 'bayoma_test_io');
if exist(workDir, 'dir') ~= 7
    mkdir(workDir);
end

data = bayoma.simulate_oma_data([1.0], [0.01], [1; 2; 2] / 3, 1.0, 1.0, ...
    100.0, 1200.0, struct('seed', 7, 'q', 0));

csvPath = fullfile(workDir, 'io_case.csv');
writematrix(data, csvPath);

spec = struct();
spec.data = struct('path', csvPath, 'format', 'csv', 'fs', 100.0);
spec.bands = [0.9, 1.1];
spec.f0 = [1.0];
spec.options = struct('mute', 1, 'q', 0);
spec.output = struct('path', fullfile(workDir, 'io_case_result.json'));

specPath = fullfile(workDir, 'io_case.json');
fid = fopen(specPath, 'w', 'n', 'UTF-8');
fprintf(fid, '%s', jsonencode(spec));
fclose(fid);

result = bayoma.io_json(specPath);
fprintf('io_json  : f=%.6f z=%.6f Sii=%.6f Se=%.6f n_modes=%d\n', ...
    result.f(1), result.z(1), result.Sii(1), result.Se(1), result.n_modes);
fprintf('phi      : %s\n', mat2str(result.phi(:).', 6));
fprintf('coefv.f  : %.6f  coefv.phi: %.6f\n', result.coefv.f(1), result.coefv.phi(1));

direct = bayoma.bayoma(struct('tdata', data, 'fs', 100.0, ...
    'f1f2', [0.9, 1.1], 'f0', {{1.0}}, 'mute', 1));
fprintf('direct   : f=%.6f z=%.6f Sii=%.6f Se=%.6f\n', ...
    direct.f(1), direct.z(1), direct.Sii(1), direct.Se(1));

fprintf('一致判据 : |df|=%.3e |dz|=%.3e |dSii|=%.3e\n', ...
    abs(result.f(1) - direct.f(1)), abs(result.z(1) - direct.z(1)), ...
    abs(result.Sii(1) - direct.Sii(1)));

raw = fileread(specPath);
fprintf('描述文件字节数 = %d\n', numel(raw));
fprintf('结果文件存在   = %d\n', exist(fullfile(workDir, 'io_case_result.json'), 'file') == 2);

% 结果文件必须符合 schema/result.schema.json 的形状约定（Python 侧写入同一份
% 契约）。MATLAB 的 jsondecode 会把长度 1 的 JSON 数组解成标量，故不能用它的
% 返回值判断形状，改为直接核对原始文本中的关键片段：
%   - 一维数组字段写成 [x]（不是裸数字）；
%   - 二维数组字段写成 [[...]]；
%   - 复数按 {"re": [[...]], "im": [[...]]} 给出。
resultText = fileread(fullfile(workDir, 'io_case_result.json'));
compact = regexprep(resultText, '\s+', '');
hasArraySii = ~isempty(regexp(compact, '"Sii":\[[-0-9]', 'once'));
hasArrayBands = ~isempty(regexp(compact, '"bands":\[\[[-0-9]', 'once'));
hasNestedS = ~isempty(regexp(compact, '"S":\[\{"re":\[\[', 'once'));
hasArrayCoefvF = ~isempty(regexp(compact, '"coefv":\{.*?"f":\[[-0-9]', 'once'));
hasMatrixPhi = ~isempty(regexp(compact, '"phi":\[\[[-0-9]', 'once'));
fprintf('契约形状 : Sii 一维=%d  bands 二维=%d  S 嵌套=%d  coefv.f 一维=%d  phi 二维=%d\n', ...
    hasArraySii, hasArrayBands, hasNestedS, hasArrayCoefvF, hasMatrixPhi);
if ~(hasArraySii && hasArrayBands && hasNestedS && hasArrayCoefvF && hasMatrixPhi)
    error('bayoma:badJson', '结果文件的字段形状与 JSON 契约不一致');
end
end
