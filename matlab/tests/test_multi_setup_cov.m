function test_multi_setup_cov()
%TEST_MULTI_SETUP_COV 多测组后验协方差的校核。
%
%   单测组六通道：与不确定度定律逐项对照。
%   四测组：检查 Hessian 无负特征值、谱参数与定律一致。

addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

fs = 100;
dur = 2400;
f0 = 3.0;
z0 = 0.01;
nDof = 6;
uTrue = [0.162; 0.286; 0.315; 0.477; 0.546; 0.516];
uTrue = uTrue / norm(uTrue);
band = [2.6, 3.4];

fprintf('=== 情形 1：单测组 6 通道 ===\n');
setups = build_setups(struct('dofs', 1:6, 'S', 1.0, 'Se', 1.0), ...
    uTrue, f0, z0, fs, dur, band, 21);
res = bayoma.multi_setup(setups, nDof, struct('q', 0));
fprintf('MPV f=%.6f z=%.6f Se=%.6f S=%.6f niter=%d conv=%d\n', ...
    res.f(1), res.z(1), res.Se(1), res.S(1), res.niter, res.converged);
cv = bayoma.multi_setup_cov(res, setups, struct('q', 0));
fprintf('coefv f=%.6f z=%.6f S=%.6f Se=%.6f phi=%.6f\n', ...
    cv.coefv.f(1), cv.coefv.z(1), cv.coefv.S(1), cv.coefv.Se(1), cv.coefv.phi);
fprintf('nNegative=%d condition=%.3e\n', cv.nNegative, cv.condition);
kappa = (band(2) - band(1)) / 2 / f0 / res.z(1);
law = bayoma.uncertainty_law(res.f(1), res.z(1), res.S(1), res.Se(1), ...
    kappa, dur, 6, 0);
fprintf('law  f=%.6f z=%.6f S=%.6f Se=%.6f phi=%.6f\n', ...
    law.f, law.z, law.S, law.Se, law.phi);

fprintf('\n=== 情形 2：四测组 ===\n');
specs = [ ...
    struct('dofs', [1 2 3], 'S', 1.0, 'Se', 1.0), ...
    struct('dofs', [3 4 5], 'S', 0.8, 'Se', 1.2), ...
    struct('dofs', [4 5], 'S', 1.1, 'Se', 0.9), ...
    struct('dofs', [5 6], 'S', 1.0, 'Se', 1.0)];
setups = build_setups(specs, uTrue, f0, z0, fs, dur, band, 22);
res = bayoma.multi_setup(setups, nDof, struct('q', 0));
fprintf('MPV f=%s\n', mat2str(res.f, 6));
fprintf('    z=%s\n', mat2str(res.z, 6));
fprintf('    Se=%s\n', mat2str(res.Se, 6));
fprintf('    S=%s\n', mat2str(res.S, 6));
fprintf('MAC=%.6f niter=%d conv=%d\n', ...
    abs(res.phi' * uTrue)^2 / (res.phi' * res.phi), res.niter, res.converged);
cv = bayoma.multi_setup_cov(res, setups, struct('q', 0));
fprintf('coefv f=%s\n', mat2str(cv.coefv.f, 6));
fprintf('      z=%s\n', mat2str(cv.coefv.z, 6));
fprintf('      S=%s\n', mat2str(cv.coefv.S, 6));
fprintf('      Se=%s\n', mat2str(cv.coefv.Se, 6));
fprintf('      phi=%.6f\n', cv.coefv.phi);
fprintf('nNegative=%d condition=%.3e\n', cv.nNegative, cv.condition);
for r = 1:numel(specs)
    kappa = (band(2) - band(1)) / 2 / f0 / res.z(r);
    law = bayoma.uncertainty_law(res.f(r), res.z(r), res.S(r), res.Se(r), ...
        kappa, dur, numel(specs(r).dofs), 0);
    fprintf('  setup%d law f=%.6f z=%.6f S=%.6f Se=%.6f phi=%.6f\n', ...
        r, law.f, law.z, law.S, law.Se, law.phi);
end
end

% ---------------------------------------------------------------- 子函数

function setups = build_setups(specs, uTrue, f0, z0, fs, dur, band, seed)
%BUILD_SETUPS 按测组划分生成合成数据并切出缩放 FFT。
%
%   局部振型取全局振型的对应分量 t_r = L_r*u，不作单独归一：模型中的 S_r 与
%   ||L_r*u||^2 配套，单独归一会使估计出的 S 偏离真值。
ns = numel(specs);
setups = struct('fft', {}, 'freqs', {}, 'dofs', {});
for r = 1:ns
    dofs = specs(r).dofs(:);
    [tdata, ~] = bayoma.simulate_oma_data(f0, z0, uTrue(dofs), ...
        specs(r).S, specs(r).Se, fs, dur, struct('q', 0, 'seed', seed + r));
    sp = bayoma.scaled_fft(tdata, fs, band, 0);
    setups(r).fft = sp.fft;
    setups(r).freqs = sp.freqs;
    setups(r).dofs = dofs;
end
end
