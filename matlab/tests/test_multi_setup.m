function test_multi_setup()
%TEST_MULTI_SETUP 多测组识别的合成数据校核。
%
%   六个自由度的悬臂结构，四个测组覆盖不同的自由度子集。先核对单测组退化
%   情形（应给出与单模态识别一致的结果），再核对四测组的全局振型。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

nDof = 6;
uTrue = [0.162; 0.286; 0.315; 0.477; 0.546; 0.516];
uTrue = uTrue / norm(uTrue);
fTrue = 3.0;
zTrue = 0.01;
fs = 100;
dur = 2400;

% 情形 1：单测组覆盖全部自由度，应与单模态识别退化一致
setups = struct('fft', {}, 'freqs', {}, 'dofs', {});
opts = struct('q', 0, 'seed', 11);
tdata = bayoma.simulate_oma_data(fTrue, zTrue, uTrue, 1.0, 1.0, fs, dur, opts);
Sspec = bayoma.scaled_fft(tdata, fs, [2.7 3.3]);
setups(1).fft = Sspec.fft;
setups(1).freqs = Sspec.freqs;
setups(1).dofs = (1:nDof).';

mopts = struct('q', 0, 'maxiter', 100);
out1 = bayoma.multi_setup(setups, nDof, mopts);
mac1 = abs(out1.phi' * uTrue);
fprintf('单测组：f=%.6f (真值 %.4f)  z=%.6f (真值 %.4f)  Se=%.6f  S=%.6f  MAC=%.6f\n', ...
    out1.f, fTrue, out1.z, zTrue, out1.Se, out1.S, mac1);

% 情形 2：四测组覆盖不同自由度子集
dofSets = {[1 2 3], [3 4 5], [4 5], [5 6]};
SeSets = [1.0 1.2 0.9 1.0];
SSets = [1.0 0.8 1.1 1.0];
setups = struct('fft', {}, 'freqs', {}, 'dofs', {});
for r = 1:numel(dofSets)
    dofs = dofSets{r}(:);
    % 局部振型必须取全局振型的对应分量，不能各自归一：模型中 t_r = L_r u，
    % 各测组的模态力谱 S_r 与之配套，归一化会改变 S_r 的可辨识尺度。
    phiLocal = uTrue(dofs);
    opts = struct('q', 0, 'seed', 20 + r);
    td = bayoma.simulate_oma_data(fTrue, zTrue, phiLocal, SSets(r), SeSets(r), fs, dur, opts);
    Sr = bayoma.scaled_fft(td, fs, [2.7 3.3]);
    setups(r).fft = Sr.fft;
    setups(r).freqs = Sr.freqs;
    setups(r).dofs = dofs;
end

out2 = bayoma.multi_setup(setups, nDof, mopts);
mac2 = abs(out2.phi' * uTrue);
fprintf('四测组：f = [%.5f %.5f %.5f %.5f]\n', out2.f);
fprintf('        z = [%.6f %.6f %.6f %.6f]\n', out2.z);
fprintf('        Se = [%.4f %.4f %.4f %.4f]  真值 [%.2f %.2f %.2f %.2f]\n', ...
    out2.Se, SeSets);
fprintf('        S  = [%.4f %.4f %.4f %.4f]  真值 [%.2f %.2f %.2f %.2f]\n', ...
    out2.S, SSets);
fprintf('        全局振型 MAC = %.6f   niter=%d conv=%d\n', mac2, out2.niter, out2.converged);
end
