function test_cov_cross()
%TEST_COV_CROSS 单模态后验协方差的 MATLAB/Python 交叉校核。
%
%   读取由 gen_for_python.m 保存的同一份数据与最大后验概率估计，计算解析
%   后验协方差，输出各参数变异系数，供 Python 侧同数据结果逐项比对。

here = fileparts(mfilename('fullpath'));
matPath = fullfile(here, 'synth_case.mat');
if exist(matPath, 'file') ~= 2
    % 数据文件不随工程分发，缺失时由生成脚本现做一份
    gen_for_python();
end
dat = load(matPath);

out = bayoma.single_mode(dat.D, dat.dTotal, dat.freqs, ...
    struct('f0', dat.f, 'q', 0, 'maxiter', 500));
cv = bayoma.single_mode_cov(out.f, out.z, out.S, out.Se, out.phi, ...
    dat.freqs, dat.D, dat.dTotal, 0);

fprintf('MATLAB MPV : f=%.9f z=%.9f S=%.9f Se=%.9f\n', out.f, out.z, out.S, out.Se);
fprintf('MATLAB coefv: f=%.9f z=%.9f S=%.9f Se=%.9f phi=%.9f rms=%.9f\n', ...
    cv.f, cv.z, cv.Sii, cv.Se, cv.phi, cv.rms);
end
