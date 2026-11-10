function out = single_mode_cov(f, z, S, Se, phi, freqs, D, dTotal, q)
%SINGLE_MODE_COV 单模态后验协方差（第 12.4 节，式 12.21–12.31 与表 12.1–12.6）。
%
%   OUT = bayoma.single_mode_cov(F, Z, S, SE, PHI, FREQS, D, DTOTAL, Q) 返回
%   结构体，字段：
%       cov     (n+4) x (n+4) 协方差矩阵，参数顺序 [f, z, S, Se, phi_1..phi_n]
%       matrix  谱参数块（4 x 4）按最大后验概率值归一后的无量纲协方差，
%               对角元素即各谱参数变异系数的平方
%       f, z, Sii, Se, phi, rms  各参数的变异系数
%
%   模态形状的单位范数约束按式 (12.22)–(12.25) 处理：约束映射 v_c 取
%   blkdiag(I_4, I_n - phi*phi')，拉格朗日乘子取 lambda_1 = -Se^(-1)*lambda，
%   lambda 为 A 的最大特征值。式 (12.30) 的 E_k^(-1) 形式用于避免数值相消。

if nargin < 9 || isempty(q)
    q = 0;
end

freqs = freqs(:);
nf = numel(freqs);
nch = numel(phi);
phi = phi(:);

[dk, der] = bayoma.dynamic_factor_derivs(f, z, freqs, q);
p = S * dk + Se;

% q_k = phi' D_k phi 与 sum_k B_k q_k
qk = zeros(1, nf);
for k = 1:nf
    qk(k) = real(phi' * D(:, :, k) * phi);
end
qsum = sum((S * dk ./ p) .* qk(:));

% ---- L_D 的二阶导数（表 12.4） ----
ld_ff = sum(S * der.ff ./ p) - sum((S * der.f).^2 ./ p.^2);
ld_fz = sum(S * der.fz ./ p) - sum(S^2 * der.f .* der.z ./ p.^2);
ld_fS = sum(der.f * Se ./ p.^2);
ld_fSe = sum(-S * der.f ./ p.^2);
ld_zz = sum(S * der.zz ./ p) - sum((S * der.z).^2 ./ p.^2);
ld_zS = sum(der.z * Se ./ p.^2);
ld_zSe = sum(-S * der.z ./ p.^2);
ld_SS = -sum(dk.^2 ./ p.^2);
ld_SSe = -sum(dk ./ p.^2);
ld_SeSe = -sum(1 ./ p.^2) - (nch - 1) * nf / Se^2;

% ---- B_k = S*D_k/p 的导数（表 12.6 的等价形式） ----
b_d = Se * S * der.f ./ p.^2;
b_z = Se * S * der.z ./ p.^2;
b_S = Se * dk ./ p.^2;
b_Se = -S * dk ./ p.^2;

b_ff = Se * S * der.ff ./ p.^2 - 2 * Se * S^2 * der.f.^2 ./ p.^3;
b_fz = Se * S * der.fz ./ p.^2 - 2 * Se * S^2 * der.f .* der.z ./ p.^3;
b_zz = Se * S * der.zz ./ p.^2 - 2 * Se * S^2 * der.z.^2 ./ p.^3;
b_fS = Se * der.f .* (Se - S * dk) ./ p.^3;
b_fSe = S * der.f .* (S * dk - Se) ./ p.^3;
b_zS = Se * der.z .* (Se - S * dk) ./ p.^3;
b_zSe = S * der.z .* (S * dk - Se) ./ p.^3;
b_SS = -2 * Se * dk.^2 ./ p.^3;
b_SSe = dk .* (S * dk - Se) ./ p.^3;
b_SeSe = 2 * S * dk ./ p.^3;

% ---- L_Q 的二阶导数（表 12.5） ----
lq_ff = -sum(b_ff .* qk(:)) / Se;
lq_fz = -sum(b_fz .* qk(:)) / Se;
lq_fS = -sum(b_fS .* qk(:)) / Se;
lq_fSe = sum(b_d .* qk(:)) / Se^2 - sum(b_fSe .* qk(:)) / Se;
lq_zz = -sum(b_zz .* qk(:)) / Se;
lq_zS = -sum(b_zS .* qk(:)) / Se;
lq_zSe = sum(b_z .* qk(:)) / Se^2 - sum(b_zSe .* qk(:)) / Se;
lq_SS = -sum(b_SS .* qk(:)) / Se;
lq_SSe = sum(b_S .* qk(:)) / Se^2 - sum(b_SSe .* qk(:)) / Se;
lq_SeSe = 2 * (dTotal - qsum) / Se^3 + 2 * sum(b_Se .* qk(:)) / Se^2 ...
    - sum(b_SeSe .* qk(:)) / Se;

h_pp = [
    ld_ff + lq_ff, ld_fz + lq_fz, ld_fS + lq_fS, ld_fSe + lq_fSe
    ld_fz + lq_fz, ld_zz + lq_zz, ld_zS + lq_zS, ld_zSe + lq_zSe
    ld_fS + lq_fS, ld_zS + lq_zS, ld_SS + lq_SS, ld_SSe + lq_SSe
    ld_fSe + lq_fSe, ld_zSe + lq_zSe, ld_SSe + lq_SSe, ld_SeSe + lq_SeSe
    ];

% ---- 混合块 L^(x u)（表 12.3）与 H_uu ----
A = zeros(nch, nch);
for k = 1:nf
    A = A + (S * dk(k) / p(k)) * D(:, :, k);
end
A = (A + A.') / 2;
dphi = zeros(nch, nf);
for k = 1:nf
    dphi(:, k) = D(:, :, k) * phi;
end

row_f = -2 * (dphi * b_d(:)) / Se;
row_z = -2 * (dphi * b_z(:)) / Se;
row_S = -2 * (dphi * b_S(:)) / Se;
row_Se = -2 * (dphi * b_Se(:)) / Se + 2 * (A * phi) / Se^2;
h_pu = [row_f, row_z, row_S, row_Se].';
h_uu = -2 * A / Se;

nh = 4 + nch;
H = zeros(nh, nh);
H(1:4, 1:4) = h_pp;
H(1:4, 5:end) = h_pu;
H(5:end, 1:4) = h_pu.';
H(5:end, 5:end) = h_uu;

% ---- 约束处理：G = 1 - phi'phi，lambda_1 = -Se^(-1) lambda ----
lam = max(eig(A));
lam1 = -lam / Se;

vmap = zeros(nh, nh);
vmap(1:4, 1:4) = eye(4);
vmap(5:end, 5:end) = eye(nch) - phi * phi.';

G = zeros(nh, nh);
G(5:end, 5:end) = -2 * eye(nch);

Hc = vmap.' * (H + lam1 * G) * vmap;
Hc = (Hc + Hc.') / 2;

% 雅可比预条件：谱参数（S、Se 的量级随数据单位变化，可从 1e-12 到 1e6）与振型分量
% （无量纲）相差若干数量级，直接求伪逆会把小量级方向判成零曲率，对应参数的方差退化
% 为零。先按对角元缩放到同一量级，求逆后再还原。
dscale = sqrt(abs(diag(Hc)));
dscale(dscale <= 0) = 1;
Hcs = Hc ./ (dscale * dscale.');
% 伪逆：约束方向上的零特征值对应模态形状沿 phi 的分量，该方向不参与不确定性
covScaled = pinv(Hcs, 1e-12);
covFree = covScaled ./ (dscale * dscale.');
cov = vmap * covFree * vmap.';
cov = (cov + cov.') / 2;

% ---- 变异系数 ----
dg = diag(cov);
out.cov = cov;
out.f = sqrt(max(dg(1), 0)) / f;
out.z = sqrt(max(dg(2), 0)) / z;
out.Sii = sqrt(max(dg(3), 0)) / S;
out.Se = sqrt(max(dg(4), 0)) / Se;
out.phi = sqrt(max(sum(dg(5:end)), 0));

% 模态加速度 rms 的变异系数按一阶小扰动由 f、z、S 及其相关性传播
rmsv = sqrt(pi * S * f / (4 * z));
grad = [0.5 * rmsv / f, -0.5 * rmsv / z, 0.5 * rmsv / S];
varRms = grad * cov(1:3, 1:3) * grad.';
out.rms = sqrt(max(varRms, 0)) / rmsv;

% 无量纲协方差：谱参数按最大后验概率值归一，故对角元素即变异系数的平方
scale = [f; z; S; Se];
Dsc = 1 ./ scale;
Cs = cov(1:4, 1:4);
out.matrix = (Dsc * ones(1, 4)) .* Cs .* (ones(4, 1) * Dsc.');
end
