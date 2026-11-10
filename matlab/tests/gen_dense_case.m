function gen_dense_case()
%GEN_DENSE_CASE 生成密集模态算例并保存，供 Python 侧交叉校核后验协方差。
%
%   数据条件：三通道、f = 1.00/1.02 Hz、z = 1%、模态力相关系数 0.5 相位 π/4、
%   噪声谱 1、采样频率 100 Hz、记录 2400 s、随机种子 3。与
%   examples/matlab_multi_mode.m 的算例不同，此处记录较短，用于比较两侧在
%   短记录下的后验协方差。

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..')));

fTrue = [1.0, 1.02];
zTrue = [0.01, 0.01];
phiTrue = [[1; 2; 2], [-2; 1; 2]] / 3;
phiTrue = phiTrue ./ sqrt(sum(phiTrue .^ 2, 1));
coh = 0.5 * exp(1i * pi / 4);
STrue = [1, coh; conj(coh), 1];
fs = 100.0;
dur = 2400.0;

tdata = bayoma.simulate_oma_data(fTrue, zTrue, phiTrue, STrue, 1.0, ...
    fs, dur, struct('seed', 3, 'q', 0));

here = fileparts(mfilename('fullpath'));
save(fullfile(here, 'dense_case.mat'), 'tdata', 'fs', 'fTrue', 'zTrue', ...
    'phiTrue', 'STrue', '-v7');
fprintf('已保存 %s（%d x %d）\n', fullfile(here, 'dense_case.mat'), ...
    size(tdata, 1), size(tdata, 2));
end
