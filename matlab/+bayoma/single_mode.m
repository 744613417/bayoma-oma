function out = single_mode(D, dTotal, freqs, opts)
%SINGLE_MODE 单模态贝叶斯识别（课本第 12 章）。
%
%   OUT = bayoma.single_mode(D, DTOTAL, FREQS) 执行单模态识别。D 为
%   nch x nch x Nf 的样本谱矩阵 D_k = Re(F_k F_k')，DTOTAL = sum_k F_k'F_k，
%   FREQS 为频率坐标（Hz）。
%
%   OUT = bayoma.single_mode(..., OPTS) 接受结构体 OPTS，字段：
%       f0        频率初值（Hz），缺省取奇异值谱峰
%       z0        阻尼比初值，缺省 0.01
%       q         模态频响函数的指数，缺省 0（0 加速度 / 1 速度 / 2 位移）
%       maxiter   MPV 迭代上限，缺省 500
%       tol       收敛容差，缺省 1e-6
%
%   返回结构体字段：
%       f, z, S, Se  最大后验概率值
%       phi          模态形状（列向量，欧氏范数 1）
%       sn           模态信噪比 S/(4*Se*z^2)
%       nllf         负对数似然函数值
%       niter        迭代次数
%       converged    是否满足收敛判据
%
%   算法按第 12.3 节：交替更新谱参数 {f, z, S, Se}（最小化式 12.13 的降维
%   目标函数）与模态形状（取 A 的最大特征值对应的特征向量）。

if nargin < 4
    opts = struct();
end
opts = fill_defaults(opts, struct( ...
    'f0', [], 'z0', 0.01, 'q', 0, 'maxiter', 500, 'tol', 1e-6));

nch = size(D, 1);
nf = size(D, 3);

if isempty(opts.f0)
    sv = bayoma.sv_curves(D);
    [~, idx] = max(sv(1, :));
    f0 = freqs(idx);
else
    f0 = opts.f0;
end

% 初值按高信噪比渐近式 (12.20)：Se 由总能量与最大特征值之差估计，
% S 由各频率点主方向能量与动态因子的比值平均得到。
A0 = sum(D, 3);
A0 = (A0 + A0.') / 2;
[v0, d0] = eig(A0);
[~, imax] = max(diag(d0));
phi = v0(:, imax);
phi = phi / norm(phi);
lambda0 = phi' * A0 * phi;

Se = max((dTotal - lambda0) / ((nch - 1) * nf), 1e-12);
Dk = bayoma.dynamic_factor(f0, opts.z0, freqs, opts.q);
S = max(mean(lambda0 ./ (nf * Dk)) / 2, 1e-12);

x = [log(f0); log(opts.z0); log(S); log(Se)];
niter = 0;
converged = false;
best = inf;

while niter < opts.maxiter
    niter = niter + 1;
    xold = x;

    % 谱参数子问题：在 log 空间用有界拟牛顿法最小化降维目标函数。
    obj = @(v) spectrum_objective(v, freqs, D, dTotal, opts.q);
    options = optimset('Display', 'off', 'MaxIter', 400, 'TolFun', 1e-14, ...
        'TolX', 1e-12, 'MaxFunEvals', 4000);
    x = fminsearch(obj, x, options);

    % 模态形状子问题：A 的最大特征值对应的特征向量。
    f = exp(x(1)); z = exp(x(2)); S = exp(x(3)); Se = exp(x(4));
    Dk = bayoma.dynamic_factor(f, z, freqs, opts.q);
    weight = 1 ./ (1 + Se ./ (S * Dk));
    A = zeros(nch, nch);
    for k = 1:nf
        A = A + weight(k) * D(:, :, k);
    end
    [v, dA] = eig((A + A.') / 2);
    [~, imax] = max(diag(dA));
    phi = v(:, imax);
    phi = phi / norm(phi);
    if phi(1) < 0
        phi = -phi;
    end

    val = spectrum_objective(x, freqs, D, dTotal, opts.q);
    if val < best
        best = val;
    end
    if max(abs(x - xold)) < opts.tol
        converged = true;
        break;
    end
end

f = exp(x(1)); z = exp(x(2)); S = exp(x(3)); Se = exp(x(4));
[L, ~] = bayoma.nllf_reduced(f, z, S, Se, freqs, D, dTotal, opts.q);

out.f = f;
out.z = z;
out.S = S;
out.Se = Se;
out.phi = phi;
out.sn = S / (4 * Se * z^2);
out.nllf = L;
out.niter = niter;
out.converged = converged;

% 后验协方差按第 12.4 节的解析形式计算，与多模态输出字段保持一致
cv = bayoma.single_mode_cov(f, z, S, Se, phi, freqs, D, dTotal, opts.q);
out.coefv = cv;
out.cmat = cv.matrix;

% 结果输出扩充所需的补充字段：振型分量的协方差块与参数排列描述。
% cv.cov 的参数顺序为 [f, z, S, Se, phi_1, ..., phi_n]，故振型块自第 5 项起；
% 单模态的振型张满整个通道空间，切空间维数为 nch-1。
out.full = cv.cov;
out.cov_phi = {cv.cov(5:end, 5:end)};
out.sizes = nch - 1;
out.n_off = 0;
out.complex = false;
% 单模态的参数排列直接给出：频率、阻尼比、模态力谱、预测误差谱与各通道的
% 振型分量，共 4+nch 项，与 out.full 的维数一致
out.index = bayoma.result_index('single', nch);
end

function L = spectrum_objective(v, freqs, D, dTotal, q)
%SPECTRUM_OBJECTIVE 降维目标函数：参数向量取 log 空间，越界返回大值。
f = exp(v(1)); z = exp(v(2)); S = exp(v(3)); Se = exp(v(4));
if z < 1e-6 || z > 0.5 || S <= 0 || Se <= 0
    L = 1e300;
    return;
end
L = bayoma.nllf_reduced(f, z, S, Se, freqs, D, dTotal, q);
if ~isfinite(L)
    L = 1e300;
end
end

function s = fill_defaults(s, d)
names = fieldnames(d);
for i = 1:numel(names)
    if ~isfield(s, names{i}) || isempty(s.(names{i}))
        s.(names{i}) = d.(names{i});
    end
end
end
