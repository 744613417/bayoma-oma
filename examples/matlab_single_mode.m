function matlab_single_mode()
%MATLAB_SINGLE_MODE 示例：单模态识别。
%
%   生成三通道合成记录，识别 1.0 Hz 附近的一阶模态，并与真值比较。
%   运行方式：
%       addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab')));
%       addpath(genpath('examples'));
%       matlab_single_mode

fTrue = 1.0;
zTrue = 0.01;
phiTrue = [1; 2; 2] / 3;
STrue = 1.0;
SeTrue = 1.0;
fs = 100.0;
duration = 2400.0;

tdata = bayoma.simulate_oma_data(fTrue, zTrue, phiTrue, STrue, SeTrue, ...
    fs, duration, struct('seed', 7, 'q', 0));
fprintf('数据：%d 样本、%d 通道、%.0f s\n', size(tdata, 1), size(tdata, 2), duration);

IN = struct('tdata', tdata, 'fs', fs, 'f1f2', [0.9, 1.1], ...
    'f0', {{fTrue}}, 'mute', 1);
OUT = bayoma.bayoma(IN);

fprintf('\n识别结果\n');
fprintf('  频率    : %.6f Hz（真值 %.6f）\n', OUT.f(1), fTrue);
fprintf('  阻尼比  : %.6f（真值 %.6f）\n', OUT.z(1), zTrue);
fprintf('  模态力谱: %.6f（真值 %.6f）\n', OUT.Sii(1), STrue);
fprintf('  噪声谱  : %.6f（真值 %.6f）\n', OUT.Se(1), SeTrue);
fprintf('  信噪比  : %.1f\n', OUT.sn(1));

mac = bayoma.mac_matrix(OUT.phi, phiTrue);
fprintf('  振型    : %s，MAC = %.6f\n', mat2str(OUT.phi(:).', 6), mac(1, 1));

fprintf('\n后验变异系数\n');
fprintf('  频率    : %.6f\n', OUT.coefv.f(1));
fprintf('  阻尼比  : %.6f\n', OUT.coefv.z(1));
fprintf('  模态力谱: %.6f\n', OUT.coefv.Sii(1));
fprintf('  噪声谱  : %.6f\n', OUT.coefv.Se(1));
fprintf('  振型    : %.6f\n', OUT.coefv.phi(1));

law = bayoma.uncertainty_law(OUT.f(1), OUT.z(1), OUT.Sii(1), OUT.Se(1), ...
    10.0, duration, size(tdata, 2), 0);
fprintf('\n不确定度定律预测（kappa = 10）\n');
fprintf('  频率    : %.6f\n', law.f);
fprintf('  阻尼比  : %.6f\n', law.z);
fprintf('  振型    : %.6f\n', law.phi);
end
