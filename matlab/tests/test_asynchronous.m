function test_asynchronous()
%TEST_ASYNCHRONOUS 异步数据单模态识别的合成数据校核（课本第 10.4 节）。
%
%   校核分三组：
%   1. 退化性：全部测点同属一个测组时，异步识别应退化为第 12 章的同步单模态
%      识别，频率、阻尼比、模态力谱密度与预测误差谱密度的相对差应小于 1e-3。
%   2. 识别精度：两个测组各两个测点，组间存在复相干 v12 = 0.5*exp(i*pi/4)。
%      频率与阻尼比的相对误差应小于 5%，振型 MAC 应大于 0.99，|v12| 的估计
%      应落在 0.2~0.8。
%   3. 后验不确定性：数值 Hessian 给出的频率、阻尼比变异系数，与第 15 章
%      不确定度定律的预测值之比应落在 0.5~2.0。定律中的带宽因子按实际选带
%      折算，即 kappa = 半带宽/(f*z)。
%
%   阻尼比的变异系数约为 5%，记录时长 4800 s 时同一批独立实现样本的阻尼比
%   相对误差标准差约 4%，故 5% 判据在单次实现上处于统计波动量级。脚本取固定
%   随机种子以保证结果可复现，取 4800 s 记录以抑制阻尼比的抽样波动。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fTrue = 1.0;
zTrue = 0.01;
STrue = 1.0;
SeTrue = [1.0; 1.5];
fs = 100;
dur = 4800;
band = [0.9 1.1];
seed = 4;

uTrue = [1; 2; 2; 3];
uTrue = uTrue / norm(uTrue);
groups = {[1 2], [3 4]};
off = 0.5 * exp(1i * pi / 4);
vTrue = [1, off; conj(off), 1];

nFail = 0;
mk = @(ok) ternary(ok, '[OK ]', '[FAIL]');

fprintf('异步数据单模态识别校核\n\n');

% ---------------------------------------------------------------- 用例一
% 三个测点同属一个测组，组间无相位差，等同第 12 章的同步单模态问题
uSync = [1; 2; 2] / 3;
data = synth_async(fTrue, zTrue, STrue, SeTrue(1), uSync, 1, {[1 2 3]}, fs, 2400, 7);
Sspec = bayoma.scaled_fft(data, fs, band);
[D, dTotal] = bayoma.psd_matrices(Sspec);
ref = bayoma.single_mode(D, dTotal, Sspec.freqs, struct('q', 0));
out1 = bayoma.asynchronous_single_mode(Sspec.freqs, Sspec.fft, {[1 2 3]}, struct('q', 0));

fprintf('--- q = 1 退化 ---\n');
fprintf('  同步单模态：f = %.8f Hz，z = %.8f，S = %.8f，Se = %.8f\n', ...
    ref.f, ref.z, ref.S, ref.Se);
fprintf('  异步单测组：f = %.8f Hz，z = %.8f，S = %.8f，Se = %.8f\n', ...
    out1.f, out1.z, out1.S, out1.Se(1));
rels = [abs(out1.f / ref.f - 1), abs(out1.z / ref.z - 1), ...
    abs(out1.S / ref.S - 1), abs(out1.Se(1) / ref.Se - 1)];
names = {'频率', '阻尼比', '模态力谱密度', '预测误差谱密度'};
for i = 1:4
    ok = rels(i) < 1e-3;
    fprintf('%s q = 1 时%s与同步识别一致  相对差 %.2e\n', mk(ok), names{i}, rels(i));
    nFail = nFail + ~ok;
end
mac1 = bayoma.mac_matrix(out1.phi, ref.phi);
ok = mac1(1, 1) > 1 - 1e-6;
fprintf('%s q = 1 时振型与同步识别一致  MAC = %.10f\n', mk(ok), mac1(1, 1));
nFail = nFail + ~ok;
fprintf('\n');

% ---------------------------------------------------------------- 用例二
data = synth_async(fTrue, zTrue, STrue, SeTrue, uTrue, vTrue, groups, fs, dur, seed);
Sspec = bayoma.scaled_fft(data, fs, band);
tic;
out2 = bayoma.asynchronous_single_mode(Sspec.freqs, Sspec.fft, groups, struct('q', 0));
tel = toc;

ef = abs(out2.f / fTrue - 1);
ez = abs(out2.z / zTrue - 1);
mac2 = bayoma.mac_matrix(out2.phi, uTrue);
modv = abs(out2.v(1, 2));
phase = angle(out2.v(1, 2));

fprintf('--- q = 2 识别 ---\n');
fprintf('  识别用时 %.2f s，N_f = %d\n', tel, Sspec.nf);
fprintf('  f = %.6f Hz（真值 %.4f，相对误差 %.3f%%）\n', out2.f, fTrue, ef * 100);
fprintf('  z = %.6f（真值 %.4f，相对误差 %.3f%%）\n', out2.z, zTrue, ez * 100);
fprintf('  S = %.6f，Se = [%.6f %.6f]\n', out2.S, out2.Se(1), out2.Se(2));
fprintf('  MAC = %.8f\n', mac2(1, 1));
fprintf('  |v12| = %.6f，相位 = %+.6f rad（真值 0.5 与 pi/4 = %.6f）\n', ...
    modv, phase, pi / 4);
fprintf('  迭代 %d 次，收敛 = %d，负对数似然函数 = %.6f\n', out2.niter, out2.converged, out2.nllf);

tests = {
    '两测组时频率误差小于 5%', ef < 0.05, sprintf('%.3f%%', ef * 100)
    '两测组时阻尼比误差小于 5%', ez < 0.05, sprintf('%.3f%%', ez * 100)
    '两测组时振型 MAC 大于 0.99', mac2(1, 1) > 0.99, sprintf('%.8f', mac2(1, 1))
    '组间相干模落在 0.2~0.8', modv > 0.2 && modv < 0.8, sprintf('|v12| = %.6f', modv)
    '两测组时收敛', logical(out2.converged), sprintf('迭代 %d 次', out2.niter)
    '相干矩阵为对角元 1 的 Hermitian 矩阵', ...
        max(abs(diag(out2.v) - 1)) < 1e-12 && max(max(abs(out2.v - out2.v'))) < 1e-12, ...
        sprintf('max|v - v^H| = %.2e', max(max(abs(out2.v - out2.v'))))
    };
for i = 1:size(tests, 1)
    ok = logical(tests{i, 2});
    fprintf('%s %s  %s\n', mk(ok), tests{i, 1}, tests{i, 3});
    nFail = nFail + ~ok;
end
fprintf('\n');

% ---------------------------------------------------------------- 用例三
cv = out2.coefv;
half = 0.5 * (Sspec.freqs(end) - Sspec.freqs(1));
kappa = half / (out2.f * out2.z);
law = bayoma.uncertainty_law(out2.f, out2.z, out2.S, mean(out2.Se), kappa, dur, 4, 0);

ratioF = cv.f / law.f;
ratioZ = cv.z / law.z;
fprintf('--- 后验不确定性 ---\n');
fprintf('  带宽因子 kappa = %.4f\n', kappa);
fprintf('  变异系数：f = %.6f，z = %.6f，S = %.6f，Se = [%.6f %.6f]\n', ...
    cv.f, cv.z, cv.S, cv.Se(1), cv.Se(2));
fprintf('  不确定度定律：f = %.6f，z = %.6f\n', law.f, law.z);
fprintf('  比值：f %.4f，z %.4f\n', ratioF, ratioZ);

ev = eig((out2.cov + out2.cov') / 2);
ok = max(max(abs(out2.cov - out2.cov'))) < 1e-12 && min(real(ev)) >= -1e-12;
fprintf('%s 协方差矩阵对称正定  最小特征值 %.3e\n', mk(ok), min(real(ev)));
nFail = nFail + ~ok;

ok = ratioF > 0.5 && ratioF < 2.0;
fprintf('%s 频率变异系数与不确定度定律的比值在 0.5~2.0  %.4f\n', mk(ok), ratioF);
nFail = nFail + ~ok;

ok = ratioZ > 0.5 && ratioZ < 2.0;
fprintf('%s 阻尼比变异系数与不确定度定律的比值在 0.5~2.0  %.4f\n', mk(ok), ratioZ);
nFail = nFail + ~ok;

vals = [cv.f, cv.z, cv.S, cv.Se(:).'];
ok = all(isfinite(vals)) && all(vals > 0);
fprintf('%s 变异系数全部为有限正数  f %.4e\n', mk(ok), cv.f);
nFail = nFail + ~ok;
fprintf('\n');

if nFail > 0
    fprintf('失败 %d 项\n', nFail);
    error('bayoma:test:failed', '异步数据单模态识别校核未通过');
end
fprintf('全部通过\n');
end

% ----------------------------------------------------------------------
function out = ternary(cond, a, b)
%TERNARY 条件表达式。
if cond
    out = a;
else
    out = b;
end
end

% ----------------------------------------------------------------------
function tdata = synth_async(f, z, S, Se, u, v, groups, fs, dur, seed)
%SYNTH_ASYNC 按式 (10.50) 的谱矩阵合成异步环境振动数据。
%
%   缩放 FFT 的协方差满足 E[F_k F_k'] = E_k，而 F_k = sqrt(dt/nt)*fft(y)，故
%   fft 的协方差为 E_k*nt/dt。先按此协方差抽取圆对称复高斯样本，再由共轭对称
%   补全负频率半轴，逆变换后乘 nt 还原时程。
nch = numel(u);
u = u(:);
ng = numel(groups);
nt = round(dur * fs);
dt = 1 / fs;
nf = floor(nt / 2) + 1;
freqs = (0:nf-1).' * fs / nt;

dk = bayoma.dynamic_factor(f, z, freqs, 0);
dk(1) = 0;

uop = zeros(nch, ng);
d0 = zeros(nch, 1);
for r = 1:ng
    uop(groups{r}, r) = u(groups{r});
    d0(groups{r}) = Se(r);
end
base = uop * v * uop.';

rng(seed);
spec = complex(zeros(nf, nch));
for k = 1:nf
    Ek = S * dk(k) * base + diag(d0);
    Ek = (Ek + Ek') / 2;
    [Ve, De] = eig(Ek);
    dv = max(real(diag(De)), 0);
    root = Ve * diag(sqrt(dv * nt / dt));
    w = (randn(nch, 1) + 1i * randn(nch, 1)) / sqrt(2);
    spec(k, :) = (root * w).';
end
full = [spec; conj(spec(end-1:-1:2, :))];
tdata = real(ifft(full, [], 1));
end
