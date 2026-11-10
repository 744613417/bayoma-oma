% 示例：多测组识别。
%
% 四个测组覆盖六自由度结构的不同测点子集，各测组的频率、阻尼比、模态力谱与噪声
% 水平均不同，全局振型共用。识别后比较全局振型与真值，并给出后验变异系数。
%
% 运行：
%   matlab -batch "addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab'))); run('examples\matlab_multi_setup.m')"

uTrue = [0.162; 0.286; 0.315; 0.477; 0.546; 0.516];
uTrue = uTrue / norm(uTrue);

fs = 100;
dur = 2400;
fTrue = 3.0;
zTrue = 0.01;

dofsAll = {[1 2 3], [3 4 5], [4 5], [5 6]};
seAll = [1.0 1.2 0.9 1.0];
sAll = [1.0 0.8 1.1 1.0];

ns = numel(dofsAll);
setups = repmat(struct('freqs', [], 'fft', [], 'dofs', []), 1, ns);
for r = 1:ns
    % 局部振型取全局振型的对应分量，不单独归一化
    local = uTrue(dofsAll{r});
    data = bayoma.simulate_oma_data(fTrue, zTrue, local, sAll(r), seAll(r), ...
        fs, dur, struct('seed', 11));
    Sspec = bayoma.scaled_fft(data, fs, [2.9 3.1]);
    setups(r).freqs = Sspec.freqs;
    setups(r).fft = Sspec.fft;
    setups(r).dofs = dofsAll{r};
    fprintf('测组 %d：自由度 %s，%d 通道\n', r, mat2str(dofsAll{r}), size(data, 2));
end

opts = struct('q', 0, 'maxiter', 200);
res = bayoma.multi_setup(setups, 6, opts);

fprintf('\n各测组的谱参数\n');
for r = 1:numel(res.f)
    fprintf('  测组 %d：f = %.6f Hz，z = %.6f，Se = %.6f，S = %.6f\n', ...
        r, res.f(r), res.z(r), res.Se(r), real(res.S(r)));
end

mac = bayoma.mac_matrix(res.phi, uTrue);
fprintf('\n全局振型\n');
fprintf('  估计 : %s\n', mat2str(res.phi(:).', 6));
fprintf('  真值 : %s\n', mat2str(uTrue(:).', 6));
fprintf('  MAC  : %.6f\n', mac(1, 1));
fprintf('  迭代 : %d 次，收敛 %d\n', res.niter, res.converged);

cov = bayoma.multi_setup_cov(res, setups, opts);
fprintf('\n后验变异系数\n');
fprintf('  f  : %s\n', mat2str(cov.coefv.f, 6));
fprintf('  z  : %s\n', mat2str(cov.coefv.z, 6));
fprintf('  S  : %s\n', mat2str(cov.coefv.S, 6));
fprintf('  Se : %s\n', mat2str(cov.coefv.Se, 6));
fprintf('  振型 : %.6f\n', cov.coefv.phi);

% 定律的带宽因子由实际频带反算：频带上界相对固有频率的偏移除以 zeta。
% 频带较窄时 kappa 小，定律的一阶修正精度下降，与数值协方差的比值会偏离 1；
% 本例的频带半宽仅 3.7 倍阻尼，阻尼比一项的比值约 0.63，属定律的适用性限制。
kappa = (3.1 - res.f(1)) / (res.f(1) * res.z(1));
law = bayoma.uncertainty_law(res.f(1), res.z(1), real(res.S(1)), res.Se(1), ...
    kappa, dur, numel(dofsAll{1}), 0);
fprintf('\n第 15 章定律（第一个测组，kappa = %.2f）\n', kappa);
fprintf('  f = %.6f，z = %.6f，S = %.6f，Se = %.6f，振型 = %.6f\n', ...
    law.f, law.z, law.S, law.Se, law.phi);
