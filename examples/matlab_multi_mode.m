function matlab_multi_mode()
%MATLAB_MULTI_MODE 示例：多模态识别（含密集模态）。
%
%   生成课本例 13.2 的两阶密集模态数据（1.00 与 1.02 Hz，模态力相关系数 0.5），
%   识别后比较频率、阻尼比、模态力谱矩阵与振型。
%   运行方式：
%       addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab')));
%       addpath(genpath('examples'));
%       matlab_multi_mode

fTrue = [1.0, 1.02];
zTrue = [0.01, 0.01];
phiTrue = [[1; 2; 2], [-2; 1; 2]] / 3;
phiTrue = phiTrue ./ sqrt(sum(phiTrue .^ 2, 1));
coh = 0.5 * exp(1i * pi / 4);
STrue = [1.0, coh; conj(coh), 1.0];
SeTrue = 1.0;
fs = 100.0;
duration = 9600.0;

tdata = bayoma.simulate_oma_data(fTrue, zTrue, phiTrue, STrue, SeTrue, ...
    fs, duration, struct('seed', 3, 'q', 0));
fprintf('数据：%d 样本、%d 通道、%.0f s\n', size(tdata, 1), size(tdata, 2), duration);

IN = struct('tdata', tdata, 'fs', fs, 'f1f2', [0.9, 1.1], ...
    'f0', {{fTrue}}, 'maxiter', 40, 'mute', 1);
OUT = bayoma.bayoma(IN);

fprintf('\n识别结果\n');
fprintf('  频率    : %s（真值 %s）\n', mat2str(OUT.f, 6), mat2str(fTrue, 6));
fprintf('  阻尼比  : %s（真值 %s）\n', mat2str(OUT.z, 6), mat2str(zTrue, 6));
fprintf('  噪声谱  : %.6f（真值 %.6f）\n', OUT.Se(1), SeTrue);

Sest = OUT.S{1};
fprintf('\n模态力谱矩阵\n');
fprintf('  估计\n%s\n', mat2str(Sest, 6));
fprintf('  真值\n%s\n', mat2str(STrue, 6));
fprintf('  互谱模：估计 %.6f，真值 %.6f\n', abs(Sest(1, 2)), abs(coh));

mac = bayoma.mac_matrix(OUT.phi, phiTrue);
fprintf('\n振型 MAC 矩阵\n%s\n', mat2str(mac, 6));

fprintf('\n后验变异系数\n');
fprintf('  频率    : %s\n', mat2str(OUT.coefv.f, 6));
fprintf('  阻尼比  : %s\n', mat2str(OUT.coefv.z, 6));
fprintf('  模态力谱: %s\n', mat2str(OUT.coefv.Sii, 6));
fprintf('  噪声谱  : %s\n', mat2str(OUT.coefv.Se, 6));
fprintf('  振型    : %s\n', mat2str(OUT.coefv.phi, 6));

fprintf('\n说明：密集模态且相干较弱时，模态力谱矩阵的非对角元方差很大，\n');
fprintf('      记录加长后收敛，短记录下不宜与真值逐项比较。\n');
end
