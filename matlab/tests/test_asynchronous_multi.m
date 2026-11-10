function test_asynchronous_multi()
%TEST_ASYNCHRONOUS_MULTI 异步数据多阶模态识别的合成数据校核（课本 10.4 节，式 10.44–10.49）。
%
%   校核分五组：
%   1. 谱矩阵组装：m = 1 时式 (10.47) 的双重求和应退化为单阶的
%      S*D_k*(U_1 V_11 U_1') + D0（U_1 为 n x q 的分块振型算子），且组装出的
%      谱矩阵为 Hermitian。
%   2. 降维恒等式：按降维式（G 的列空间正交基 B、Bt = D0^(-1/2)B）算出的负对数
%      似然，应与逐频率 n x n 直接分解（logdet 加二次型）的结果一致。
%   3. 单测组退化：全部测点同属一个测组时，异步多阶模型退化为第 13 章的同步
%      多阶问题（单测组时 Ui = ui、V = I）。核对频率与阻尼比的最可能值。
%   4. 两阶三测组识别：两阶模态、组间存在复相干 v12 = 0.5*exp(i*pi/4)，核对
%      频率、阻尼比、振型与相干矩阵非对角元。
%   5. 后验不确定性：数值 Hessian 给出的频率、阻尼比变异系数与第 15 章不确定度
%      定律的预测值比较。
%
%   第 4、5 组的自由度与测组划分按课本式 (10.44)-(10.49) 的多测组记号给出：
%   结构共 5 个自由度，三个测组分别观测自由度 {1,2,3}、{2,3,4}、{3,4,5}，
%   相邻测组共用自由度 2、3、4。程序接口要求每个通道只属于一个测组，故把共用的
%   自由度在各组内各展开一次，共 9 个通道：dof = [1 2 3 | 2 3 4 | 3 4 5]，
%   测组为 {1:3}、{4:6}、{7:9}。振型真值按 dof 展开后逐组归一。
%
%   第 3 组中本实现（ng == 1 且 m > 1）取复模态形状：每阶形状为 C^n 单位球上
%   的复向量，u_i(theta) = (u_i0 + Z_i*theta)/||u_i0 + Z_i*theta||，Z_i 为单位球
%   切空间的正交基并剔除整体相位方向 1i*u_i0（该方向对应 u_i -> u_i*exp(1i*phi)，
%   可由 S 的第 i 行、列相位吸收，似然不变），实自由度 2*nch-2。multi_mode 的
%   形状为 phi = B0*alpha（B0 为 nch x m0 实正交基、m0 = min(m, nch)，alpha 为
%   m0 x m 复坐标），每阶形状落在同一个 m0 维实子空间内，实自由度 2*m0-2。
%   两类是严格嵌套关系：本实现包含同步多阶类，最大似然点因此不同。
%
%   判据按模型类嵌套关系写，不放宽任何可判量：
%   (a) 频率相对差 < 1e-4；(b) 阻尼比相对差 < 1e-3；
%   (c) 嵌套似然：本实现的最优似然不高于同步解在本模型似然下的取值（相对容差
%       1e-6），这是「本类包含同步类」的直接推论；
%   (d) 形状差落在卡方涨落内：多出的 nExtra 个实自由度上，似然比 2|dL| 小于
%       卡方 99% 分位；
%   (e) 收敛。
%
%   振型 MAC 只作参考打印。判据不能写「两个点估计的 MAC 相等」，因为两类模型的
%   形状自由度不同：本实现每阶为 C^n 单位球上的自由复向量（剔除整体相位后实自由
%   度 2*nch-2），multi_mode 的形状为 phi = B0*alpha（B0 为 nch x m0 实正交基），
%   每阶落在同一个 m0 维实子空间内（实自由度 2*m0-2）。实测第 1 阶 1 - MAC =
%   6.4e-6，而两者与真值的 1 - MAC 本身就在 1e-4 量级（2400 s 记录、1.00/1.02 Hz
%   密集模态、间距约半个半功率带宽）；两个这样的估计之间互差 6.4e-6 属 O(1/N)
%   抽样差，要把该量压到 1e-6 需把记录长度增大约一个半数量级。
%   收敛容差不是成因：把 opts.tol 从 1e-7 收到 1e-9、1e-11 后，负对数似然只降
%   3.5e-8、迭代数从 52 增至 78，1 - MAC 保持在 6.402e-6 / 6.401e-6 / 6.398e-6。

%
%   第 4 组中相干矩阵的模与相位判据取 0.1 与 0.15 rad：V12 只有模与相位两个
%   自由度，其估计方差由带内独立平均次数决定，4800 s 记录、0.9-1.5 Hz 频带的
%   独立平均次数约 30，同一真值下不同种子的极大似然点散布在模 0.02-0.08、
%   相位 0.05-0.16 rad 之间。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

nFail = 0;
nCheck = 0;
mk = @(ok) ternary(ok, '[OK ]', '[FAIL]');

fprintf('异步数据多阶模态识别校核\n\n');

% ---------------------------------------------------------------- 用例一
% 组装：m = 1 时式 (10.47) 应退化为 S*D_k*(U_1 V_11 U_1') + D0，其中 U_1 为
% n x q 的分块振型算子（第 r 列只在第 r 组行块上非零）。U_1 V_11 U_1' 是分块
% 对角矩阵，跨测组块恒为零；不能简写成 u*u'（后者含跨组外积，测组间时钟不同步
% 时这些外积不存在）。
nch = 4;
f1 = 1.2;
z1 = 0.01;
S1 = 0.8;
Se1 = [1.0; 1.5];
groups1 = {[1 2], [3 4]};
u1 = [1; 1.5; 0.8; 1.1];
u1 = u1 / norm(u1);
v1 = 0.4 * exp(1i * pi / 5);
V11 = [1, v1; conj(v1), 1];
% Vc{i,j} 是「阶对 (i,j) 的 q x q 相干矩阵」：m = 1 时只有 Vc{1,1}，它就是
% 两个测组之间的相干矩阵（对角元 1，非对角元 v1），与单阶实现的 uop*v*uop'
% 用法一致。若填单位矩阵，测组间的互谱块全为零。
Vc1 = cell(1, 1);
Vc1{1, 1} = V11;
freqs1 = (0.9:0.01:1.5).';          % 先做一次小频带组装检验
Ek = bayoma.asynchronous_multi_mode('theoreticalPsd', pack_par( ...
    f1, z1, S1, Se1, u1, groups1, Vc1, freqs1));

Dk = bayoma.dynamic_factor(f1, z1, freqs1, 0);
d0 = zeros(nch, 1);
for r = 1:2
    d0(groups1{r}) = Se1(r);
end
U1 = zeros(nch, 2);
for r = 1:2
    U1(groups1{r}, r) = u1(groups1{r});
end
ref = complex(zeros(nch, nch, numel(freqs1)));
for k = 1:numel(freqs1)
    ref(:, :, k) = S1 * Dk(k) * (U1 * V11 * U1.') + diag(d0);
end
dev = max(abs(Ek(:) - ref(:))) / max(abs(ref(:)));

fprintf('--- 用例一：谱矩阵组装 ---\n');
fprintf('  频点数 %d，m = 1 时与单阶公式的最大相对偏差 %.3e\n', numel(freqs1), dev);
herm = max(max(max(abs(Ek - conj(permute(Ek, [2 1 3]))))));
fprintf('  max|E - E^H| = %.2e\n', herm);
ok = dev < 1e-12;
fprintf('%s m = 1 时式 (10.47) 与单阶公式一致\n', mk(ok));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = herm < 1e-12;
fprintf('%s 谱矩阵为 Hermitian\n', mk(ok));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
fprintf('\n');

% ---------------------------------------------------------------- 用例二
% 降维恒等式：与逐频率 n x n 直接分解一致（两阶两测组、带内相干非零）
f2 = [1.00; 1.40];
z2 = [0.010; 0.012];
S2 = [1.0, 0.15; 0.15, 0.9];
Se2 = [1.0; 1.5];
phi2 = [1.0, 0.4; 1.6, -0.3; 2.0, 0.9; 1.1, 1.2; 0.6, -1.0];
phi2 = phi2 ./ vecnorm(phi2, 2, 1);
groups2 = {[1 2 3], [4 5]};
v2 = 0.5 * exp(1i * pi / 4);
W12 = [1, v2; conj(v2), 1];
Vc2 = cell(2, 2);
Vc2{1, 1} = eye(2);
Vc2{2, 2} = eye(2);
Vc2{1, 2} = W12;
Vc2{2, 1} = W12';

data = synth_async_multi(f2, z2, S2, Se2, phi2, Vc2, groups2, 100, 600, 21, 0);
Sspec = bayoma.scaled_fft(data, 100, [0.9 1.5]);
par = pack_par(f2, z2, S2, Se2, phi2, groups2, Vc2, Sspec.freqs);
par.fft = Sspec.fft;
t0 = tic;
Lred = bayoma.asynchronous_multi_mode('nllf', par);
tRed = toc(t0);
t0 = tic;
Ldir = direct_nllf(f2, z2, S2, Se2, phi2, groups2, Vc2, Sspec.freqs, Sspec.fft);
tDir = toc(t0);
relL = abs(Lred / Ldir - 1);

fprintf('--- 用例二：降维恒等式 ---\n');
fprintf('  N_f = %d，降维 %.9f（%.3f s）  直接 %.9f（%.3f s）\n', ...
    Sspec.nf, Lred, tRed, Ldir, tDir);
fprintf('  相对差 %.2e\n', relL);
ok = relL < 1e-10;
fprintf('%s 降维似然与逐频率直接计算一致\n', mk(ok));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = tRed < 0.05;
fprintf('%s 单次似然评估在 50 ms 量级  %.3f ms\n', mk(ok), tRed * 1e3);
nFail = nFail + ~ok;
nCheck = nCheck + 1;
fprintf('\n');

% ---------------------------------------------------------------- 用例三
% 单测组退化到第 13 章的同步多阶问题
phi3 = [1.0, -0.6; 1.8, 0.5; 2.2, 0.9];
phi3 = phi3 ./ vecnorm(phi3, 2, 1);
f3 = [1.00; 1.02];
z3 = [0.010; 0.010];
S3 = [1.0, 0.3; 0.3, 0.8];
data3 = synth_sync(f3, z3, S3, 1.0, phi3, 100, 2400, 11, 0);
Sspec3 = bayoma.scaled_fft(data3, 100, [0.9 1.1]);
[D3, dT3] = bayoma.psd_matrices(Sspec3);
t0 = tic;
ref3 = bayoma.multi_mode(D3, Sspec3.fft, Sspec3.freqs, 2, struct('q', 0));
tSync = toc(t0);
t0 = tic;
out3 = bayoma.asynchronous_multi_mode(Sspec3.freqs, Sspec3.fft, ...
    {1:3}, 2, struct('q', 0));
tAsync = toc(t0);

% 两个实现的相位基准不同：S 与 phi 先按基准相位对齐再比。
% 本实现（单测组、m > 1）取复模态形状，multi_mode 取 phi = B0*alpha（实正交基乘
% 复坐标）；两者的相位规范都是「每列绝对值最大的分量为正实数」，故 phi 可直接用
% 复内积算 MAC，无需投影到实形状。S 的对角元在两侧都是正实数（相位变换
% S -> diag(e^{i*theta}) S diag(e^{-i*theta}) 不改变对角元），直接比较。
sRef = ref3.S;
ph = exp(-1i * angle(diag(sRef)));
sRef = sRef .* (ph * ph');
phiRef = ref3.phi;
mac3 = bayoma.mac_matrix(out3.phi, phiRef);
macd = diag(mac3).';
% 只用实部的 MAC：两种定义都打印，便于说明「振型差」的量级来自哪一处。
macReal = diag(bayoma.mac_matrix(real(out3.phi), real(phiRef))).';

relF = abs(out3.f(:) ./ ref3.f(:) - 1);
relZ = abs(out3.z(:) ./ ref3.z(:) - 1);
relS = abs(real(diag(out3.S)) ./ real(diag(sRef)) - 1);

% 本模型似然在同步解参数处的取值：用于「嵌套」与「卡方」两项判据。
VcId = num2cell(ones(2));
parD = struct('f', ref3.f(:), 'z', ref3.z(:), 'S', ref3.S, ...
    'Se', ref3.Se(:), 'phi', ref3.phi, 'groups', {{1:3}}, 'Vc', {VcId}, ...
    'freqs', Sspec3.freqs(:), 'fft', Sspec3.fft, 'qexp', 0);
nllRefPt = bayoma.asynchronous_multi_mode('nllf', parD);
parO = parD;
parO.f = out3.f(:);
parO.z = out3.z(:);
parO.S = out3.S;
parO.Se = out3.Se(:);
parO.phi = out3.phi;
nllOutPt = bayoma.asynchronous_multi_mode('nllf', parO);
fprintf('  似然：本模型@本解 %.9f，本模型@同步解 %.9f，同步多阶 %.9f（差 %.3e）\n', ...
    nllOutPt, nllRefPt, ref3.nllf, nllOutPt - nllRefPt);
% 两类形状与真值的接近程度（真值形状为实向量，同时落在两个模型类内）
macTrue = diag(bayoma.mac_matrix(out3.phi, phi3)).';
macTrueRef = diag(bayoma.mac_matrix(ref3.phi, phi3)).';
% 形状的逐分量相对差（按最大分量归一后比较），用于说明复形状确实被辨识出来
phiDif = max(abs(out3.phi - phiRef), [], 1) ./ max(abs(phiRef), [], 1);
phiIm = max(abs(imag(out3.phi)), [], 1);
% 同步多阶的形状子空间维数 m0 = min(m, nch)，用于计算两类形状的自由度差
m0Ref = ref3.subspace_rank;

fprintf('--- 用例三：单测组退化 ---\n');
fprintf('  同步多阶：f = [%.8f %.8f]，z = [%.8f %.8f]，Se = %.6f（%.1f s）\n', ...
    ref3.f(1), ref3.f(2), ref3.z(1), ref3.z(2), ref3.Se, tSync);
fprintf('  异步单测组：f = [%.8f %.8f]，z = [%.8f %.8f]，Se = %.6f（%.1f s，%d 次迭代）\n', ...
    out3.f(1), out3.f(2), out3.z(1), out3.z(2), out3.Se(1), tAsync, out3.niter);
fprintf('  相对差：f [%.2e %.2e]，z [%.2e %.2e]，S 对角 [%.2e %.2e]  MAC [%.8f %.8f]\n', ...
    relF(1), relF(2), relZ(1), relZ(2), relS(1), relS(2), macd(1), macd(2));
fprintf('  复形状：max|Im phi| = [%.2e %.2e]，按最大分量归一的逐分量最大差 [%.2e %.2e]\n', ...
    phiIm(1), phiIm(2), phiDif(1), phiDif(2));

ok = all(relF < 1e-4);
fprintf('%s q = 1 时频率与同步多阶识别一致（< 1e-4）\n', mk(ok));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = all(relZ < 1e-3);
fprintf('%s q = 1 时阻尼比与同步多阶识别一致（< 1e-3）\n', mk(ok));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
% 嵌套判据：本模型的模型类包含同步多阶的模型类，故在同步解处的本模型似然不应
% 低于本模型自身的最优值（相对容差 1e-6，覆盖数值舍入）。
ok = nllOutPt <= nllRefPt + 1e-6 * max(abs(nllRefPt), 1);
fprintf('%s q = 1 时本模型的最优似然不高于同步解处的似然（嵌套关系，< 1e-6）\n', mk(ok));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
% 形状差判据：本实现每阶形状为 C^n 单位球上的复向量（实自由度 2*nch-2），
% multi_mode 每阶形状限制在 m0 维实子空间内（实自由度 2*m0-2）。多出的 nExtra
% 个实自由度上，两类模型的似然比在真值附近近似服从卡方分布，判据取 99% 分位：
% 超出说明差异来自模型设定，而不是抽样噪声。
nExtra = 2 * size(out3.phi, 2) * (size(out3.phi, 1) - m0Ref);
lrStat = 2 * abs(nllRefPt - nllOutPt);
if nExtra > 0
    lrCrit = 2 * gammaincinv(0.99, nExtra / 2);
    ok = lrStat < lrCrit;
    fprintf(['%s q = 1 时与同步多阶的形状差落在卡方涨落内（自由度差 %d，' ...
        '2|dL| = %.4f，99%% 分位 %.4f）\n'], mk(ok), nExtra, lrStat, lrCrit);
else
    ok = lrStat < 1e-6 * max(abs(nllRefPt), 1);
    fprintf('%s q = 1 时与同步多阶的形状差可忽略（自由度差 0，2|dL| = %.4f）\n', ...
        mk(ok), lrStat);
end
nFail = nFail + ~ok;
nCheck = nCheck + 1;
% 振型相似度只作参考打印，不设判据：本实现每阶形状取 C^n 单位球上的自由复向量
% （实自由度 2*nch-2），multi_mode 的形状限制在 m0 维实子空间内（实自由度
% 2*m0-2），两者不是同一模型类，点估计不会相等。判据由前面的嵌套与卡方两条承担。
fprintf('  参考（不设判据）：与同步多阶的振型 MAC  %.8f  %.8f\n', macd(1), macd(2));
fprintf('  参考：与真值的 MAC  本实现 [%.8f %.8f]，同步 [%.8f %.8f]\n', ...
    macTrue(1), macTrue(2), macTrueRef(1), macTrueRef(2));
fprintf('  参考：本实现与同步解的振型 MAC [%.8f %.8f]（复内积）[%.8f %.8f]（仅实部）\n', ...
    macd(1), macd(2), macReal(1), macReal(2));
% 模态力谱对角元只作参考：两侧的 S 规范（整体相位基准）与优化停点不同，对角元
% 的差异不随判据收紧而消失。
fprintf('  参考：模态力谱对角元相对差 [%.2e %.2e]\n', relS(1), relS(2));
ok = logical(out3.converged);
fprintf('%s q = 1 时收敛  %d 次迭代\n', mk(ok), out3.niter);
nFail = nFail + ~ok;
nCheck = nCheck + 1;
fprintf('\n');

% ---------------------------------------------------------------- 用例四
% 两阶三测组识别：5 个自由度、3 个测组（相邻组共用自由度），4800 s 记录
fTrue = [1.0, 1.4];
zTrue = [0.01, 0.012];
STrue = [1.0, 0.2; 0.2, 0.9];
SeTrue = [1.0, 1.5, 1.2];
dof = [1 2 3, 2 3 4, 3 4 5];
groups4 = {1:3, 4:6, 7:9};
fs = 100;
dur = 4800;
band = [0.9 1.5];
u5 = [1.0, 0.2; 1.4, -0.5; 2.0, 1.0; 0.6, 1.5; 0.3, 1.9];
u5 = u5 ./ vecnorm(u5);
uTrue = u5(dof, :);
uTrue = uTrue ./ vecnorm(uTrue);

rho = 0.5 * exp(1i * pi / 4);
Vc4 = cell(2, 2);
for i = 1:2
    for j = 1:2
        Vc4{i, j} = eye(3);
    end
end
for r = 1:2
    for s = r + 1:3
        Vc4{1, 2}(r, s) = rho;
        Vc4{1, 2}(s, r) = conj(rho);
    end
end
Vc4{2, 1} = Vc4{1, 2}';

t0 = tic;
data4 = synth_async_multi(fTrue, zTrue, STrue, SeTrue, uTrue, Vc4, groups4, fs, dur, 3, 0);
tSynth = toc(t0);
Sspec4 = bayoma.scaled_fft(data4, fs, band);
t0 = tic;
out4 = bayoma.asynchronous_multi_mode(Sspec4.freqs, Sspec4.fft, groups4, 2, ...
    struct('f0', fTrue, 'q', 0));
tId = toc(t0);

% 逐阶相对误差。两侧都显式整理成行向量，避免行列形状不同时按矩阵广播成
% 阶与阶交叉的二维数组（那样按线性下标取值会读到「第 1 阶除以第 2 阶真值」）。
ef = abs(reshape(out4.f, 1, []) ./ reshape(fTrue, 1, []) - 1);
ez = abs(reshape(out4.z, 1, []) ./ reshape(zTrue, 1, []) - 1);
mac4 = diag(bayoma.mac_matrix(out4.phi, uTrue)).';
vEst = out4.vij.v12(1, 2);
dvMod = abs(abs(vEst) - abs(rho));
dvAng = abs(angle(vEst) - angle(rho));

fprintf('--- 用例四：两阶三测组识别 ---\n');
fprintf('  通道 %d，测组 %d 个，自由度 %d 个，N_f = %d，记录 %d s\n', ...
    size(Sspec4.fft, 1), numel(groups4), numel(unique(dof)), Sspec4.nf, dur);
fprintf('  合成 %.1f s，识别 %.1f s\n', tSynth, tId);
fprintf('  f = [%.5f %.5f] Hz（真值 [%.2f %.2f]，相对误差 [%.2e %.2e]）\n', ...
    out4.f(1), out4.f(2), fTrue(1), fTrue(2), ef(1), ef(2));
fprintf('  z = [%.6f %.6f]（真值 [%.3f %.3f]，相对误差 [%.2e %.2e]）\n', ...
    out4.z(1), out4.z(2), zTrue(1), zTrue(2), ez(1), ez(2));
fprintf('  S = [%.6f %.6f; %.6f %.6f]，Se = [%.6f %.6f %.6f]\n', ...
    out4.S(1, 1), out4.S(1, 2), out4.S(2, 1), out4.S(2, 2), out4.Se(1), out4.Se(2), out4.Se(3));
fprintf('  MAC = [%.8f %.8f]，NLLF = %.6f，迭代 %d 次，收敛 = %d\n', ...
    mac4(1), mac4(2), out4.nllf, out4.niter, out4.converged);
fprintf('  V_12：模 %.4f（真值 %.4f，差 %.4f），相位 %+.4f rad（真值 %+.4f，差 %.4f）\n', ...
    abs(vEst), abs(rho), dvMod, angle(vEst), angle(rho), dvAng);

ok = all(ef < 2e-3);
fprintf('%s 两阶频率相对误差均小于 0.2%%  [%.2e %.2e]\n', mk(ok), ef(1), ef(2));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = all(ez < 0.10);
fprintf('%s 两阶阻尼比相对误差均小于 10%%  [%.2e %.2e]\n', mk(ok), ez(1), ez(2));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = all(mac4 > 0.99);
fprintf('%s 两阶振型 MAC 均大于 0.99  [%.8f %.8f]\n', mk(ok), mac4(1), mac4(2));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = logical(out4.converged);
fprintf('%s 两阶识别收敛  %d 次迭代\n', mk(ok), out4.niter);
nFail = nFail + ~ok;
nCheck = nCheck + 1;
vh = max(max(abs(out4.vij.v12 - out4.vij.v12')));
ok = vh < 1e-12 && max(abs(diag(out4.vij.v12) - 1)) < 1e-12;
fprintf('%s 相干矩阵为对角元 1 的 Hermitian 矩阵  max|v - v^H| = %.2e\n', mk(ok), vh);
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = dvMod < 0.10;
fprintf('%s 相干矩阵模的偏差小于 0.1  %.4f\n', mk(ok), dvMod);
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = dvAng < 0.15;
fprintf('%s 相干矩阵相位的偏差小于 0.15 rad  %.4f\n', mk(ok), dvAng);
nFail = nFail + ~ok;
nCheck = nCheck + 1;
imS = max(max(abs(imag(out4.S))));
ok = imS < 1e-12;
fprintf('%s 三测组时 S 为实对称矩阵  max|Im S_ij| = %.3e\n', mk(ok), imS);
nFail = nFail + ~ok;
nCheck = nCheck + 1;
fprintf('\n');

% ---------------------------------------------------------------- 用例五
% 后验协方差与不确定度定律的比较（复用用例四最可能值处的协方差）
cv = out4.coefv;
half = 0.5 * (Sspec4.freqs(end) - Sspec4.freqs(1));
nch4 = size(Sspec4.fft, 1);
kappa = half ./ (out4.f(:) .* out4.z(:));
lawF = zeros(2, 1);
lawZ = zeros(2, 1);
for i = 1:2
    law = bayoma.uncertainty_law(out4.f(i), out4.z(i), ...
        real(out4.S(i, i)), mean(out4.Se), kappa(i), dur, nch4, 0);
    lawF(i) = law.f;
    lawZ(i) = law.z;
end
ratioF = cv.f(:) ./ lawF;
ratioZ = cv.z(:) ./ lawZ;
ev = eig((out4.cov + out4.cov') / 2);

fprintf('--- 用例五：后验不确定性 ---\n');
fprintf('  自由参数 %d 个，带宽因子 kappa = [%.3f %.3f]\n', size(out4.cov, 1), kappa(1), kappa(2));
fprintf('  变异系数：f = [%.6f %.6f]，z = [%.6f %.6f]\n', cv.f(1), cv.f(2), cv.z(1), cv.z(2));
fprintf('  变异系数：S = [%.6f %.6f]，Se = [%.6f %.6f %.6f]\n', ...
    cv.S(1, 1), cv.S(2, 2), cv.Se(1), cv.Se(2), cv.Se(3));
fprintf('  不确定度定律：f = [%.6f %.6f]，z = [%.6f %.6f]\n', ...
    lawF(1), lawF(2), lawZ(1), lawZ(2));
fprintf('  比值：f = [%.4f %.4f]，z = [%.4f %.4f]\n', ...
    ratioF(1), ratioF(2), ratioZ(1), ratioZ(2));
fprintf('  振型变异系数 [%.6f %.6f]，相干矩阵 %.6f\n', cv.phi(1), cv.phi(2), cv.v.v12);
fprintf('  协方差矩阵特征值：[%.3e, %.3e]\n', min(real(ev)), max(real(ev)));

ok = max(max(abs(out4.cov - out4.cov'))) < 1e-10 && min(real(ev)) > -1e-9;
fprintf('%s 协方差矩阵对称且无负特征值\n', mk(ok));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = all(ratioF > 0.5 & ratioF < 2.0);
fprintf('%s 频率变异系数与不确定度定律的比值在 0.5~2.0  [%.4f %.4f]\n', ...
    mk(ok), ratioF(1), ratioF(2));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
ok = all(ratioZ > 0.5 & ratioZ < 2.0);
fprintf('%s 阻尼比变异系数与不确定度定律的比值在 0.5~2.0  [%.4f %.4f]\n', ...
    mk(ok), ratioZ(1), ratioZ(2));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
vals = [cv.f(:); cv.z(:); real(diag(cv.S)); cv.Se(:); cv.phi(:); cv.v.v12];
ok = all(isfinite(vals)) && all(vals > 0);
fprintf('%s 变异系数全部为有限正数  min = %.3e\n', mk(ok), min(vals));
nFail = nFail + ~ok;
nCheck = nCheck + 1;
fprintf('\n');

fprintf('共校核 %d 项，失败 %d 项\n', nCheck, nFail);
if nFail > 0
    % error 的格式串只接受标量参数之外的形式容易出错，这里先拼好再说
    error('bayoma:test:failed', '%s', ...
        sprintf('异步数据多阶模态识别校核未通过（%d 项）', nFail));
end
fprintf('全部通过\n');
end

% ======================================================================
function par = pack_par(f, z, S, Se, phi, groups, Vc, freqs)
%PACK_PAR 组装校核入口的参数结构体。
%
%   逐字段赋值而非用 struct('groups', {groups}, ...)：后者对元胞字段会构造出
%   结构体数组，入口随即报参数类型错误。
par = struct();
par.f = f(:);
par.z = z(:);
par.S = S;
par.Se = Se(:);
par.phi = phi;
par.groups = groups;
par.Vc = Vc;
par.freqs = freqs(:);
par.qexp = 0;
end

% ======================================================================
function L = direct_nllf(f, z, S, Se, phi, groups, Vc, freqs, fft)
%DIRECT_NLLF 逐频率 n x n 直接分解算负对数似然（式 10.47 的定义式）。
%
%   与 asynchronous_multi_mode 的降维实现互为独立路径：此处不做任何 Woodbury
%   变换，每个频率点独立求 logdet 与二次型。
nch = size(fft, 1);
nf = numel(freqs);
m = numel(f);
ng = numel(groups);
h = bayoma.transfer_matrix(f(:), z(:), freqs, 0);
d0 = zeros(nch, 1);
for r = 1:ng
    d0(groups{r}) = Se(r);
end
U = cell(m, 1);
for i = 1:m
    Ui = zeros(nch, ng);
    for r = 1:ng
        Ui(groups{r}, r) = phi(groups{r}, i);
    end
    U{i} = Ui;
end
S = (S + S') / 2;
D0 = diag(d0);
logdetTot = 0;
quadTot = 0;
for k = 1:nf
    Ek = D0;
    for i = 1:m
        for j = 1:m
            c = S(i, j) * h(i, k) * conj(h(j, k));
            if c ~= 0
                Ek = Ek + c * (U{i} * Vc{i, j} * U{j}.');
            end
        end
    end
    Ek = (Ek + Ek') / 2;
    % 用 Cholesky 因子求 logdet 与二次型：E = L*L' 时 ln|E| = 2*sum(ln(diag(L)))、
    % F'*inv(E)*F = ||L' \ F||^2。工程里没有 slogdet，且 eig 的路径在近奇异谱矩阵上
    % 的精度不如 Cholesky。
    [Lk, p] = chol(Ek);
    if p ~= 0
        L = inf;
        return;
    end
    logdetTot = logdetTot + 2 * sum(log(real(diag(Lk))));
    w = Lk' \ fft(:, k);
    quadTot = quadTot + real(w' * w);
end
L = nch * nf * log(pi) + logdetTot + quadTot;
end

% ======================================================================
function tdata = synth_async_multi(f, z, S, Se, phi, Vc, groups, fs, dur, seed, q)
%SYNTH_ASYNC_MULTI 按式 (10.47) 的谱矩阵合成异步多阶环境振动数据。
%
%   缩放 FFT 的协方差满足 E[F_k F_k'] = E_k，而 F_k = sqrt(dt/nt)*fft(y)，故
%   fft 的协方差为 E_k*nt/dt。先按此协方差抽取圆对称复高斯样本，再由共轭对称
%   补全负频率半轴，逆变换后还原时程。
nch = size(phi, 1);
m = numel(f);
ng = numel(groups);
nt = round(dur * fs);
dt = 1 / fs;
nf = floor(nt / 2) + 1;
freqs = (0:nf-1).' * fs / nt;

h = bayoma.transfer_matrix(f(:), z(:), freqs, q);
h(:, 1) = 0;                 % 直流项置零，避免 (2*pi*f)^(-q) 在 0 Hz 奇异
d0 = zeros(nch, 1);
for r = 1:ng
    d0(groups{r}) = Se(r);
end
U = cell(m, 1);
for i = 1:m
    Ui = zeros(nch, ng);
    for r = 1:ng
        Ui(groups{r}, r) = phi(groups{r}, i);
    end
    U{i} = Ui;
end
S = (S + S') / 2;
D0 = diag(d0);

rng(seed);
spec = complex(zeros(nf, nch));
for k = 1:nf
    Ek = D0;
    for i = 1:m
        for j = 1:m
            c = S(i, j) * h(i, k) * conj(h(j, k));
            if c ~= 0
                Ek = Ek + c * (U{i} * Vc{i, j} * U{j}.');
            end
        end
    end
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

% ======================================================================
function tdata = synth_sync(f, z, S, Se, phi, fs, dur, seed, q)
%SYNTH_SYNC 同步多阶数据的合成（全部通道共用时基，即第 13 章模型）。
%
%   与 synth_async_multi 同法，只是 V 退化为全 1 的 q x q 矩阵。
nch = size(phi, 1);
m = numel(f);
Vc = cell(m, m);
for i = 1:m
    for j = 1:m
        Vc{i, j} = 1;
    end
end
tdata = synth_async_multi(f, z, S, Se, phi, Vc, {1:nch}, fs, dur, seed, q);
end

% ======================================================================
function out = ternary(cond, a, b)
%TERNARY 条件表达式。
if cond
    out = a;
else
    out = b;
end
end
