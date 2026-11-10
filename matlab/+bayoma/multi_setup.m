function out = multi_setup(setups, nDof, opts)
%MULTI_SETUP 多测组的贝叶斯模态识别（课本第 14 章）。
%
%   OUT = bayoma.multi_setup(SETUPS, NDOF) 对各测组数据作联合识别。
%
%   SETUPS 为结构体数组，每个元素对应一个测组，字段：
%       fft     缩放 FFT，n_r x Nf
%       freqs   频率坐标（Hz），Nf x 1
%       dofs    该测组各通道对应的自由度编号，n_r x 1
%   各测组的频率坐标可以不同；自由度编号用于构造选择矩阵 L_r。
%
%   NDOF 为全局自由度总数。
%
%   OUT = bayoma.multi_setup(..., OPTS) 接受结构体 OPTS，字段：
%       q         模态频响函数的指数，缺省 0
%       f0        全局频率初值（Hz），缺省取各测组奇异值谱峰
%       maxiter   每个测组谱参数子问题的迭代上限，缺省 200
%       tol       外层迭代收敛容差，缺省 1e-8
%       condense  是否启用参考自由度凝聚（式 14.17--14.20），缺省 true
%
%   返回结构体字段：
%       f, z      各测组的频率与阻尼比，1 x q
%       S, Se     各测组的模态力谱与通道噪声谱，1 x q
%       phi       全局振型，nDof x 1，欧氏范数为 1
%       local     各测组的局部振型，元胞数组，每项 n_r x 1
%       lam, c    式 (14.41) 与式 (14.37) 的辅助量
%       nllf      负对数似然函数值
%       niter     迭代轮数
%       converged 是否满足收敛判据
%
%   算法按第 14.3.3 节：逐测组更新谱参数 {f^(r), z^(r), S0^(r), Se^(r)}，再以
%   式 (14.40) 的最小特征值特征向量更新全局振型，最后按式 (14.41) 与式 (14.37)
%   更新辅助量 lam 与 c，循环至收敛。

if nargin < 3
    opts = struct();
end
opts = fill_defaults(opts, struct('q', 0, 'f0', [], 'maxiter', 200, 'tol', 1e-8, 'condense', true));

q = opts.q;
ns = numel(setups);

Lc = [];
if opts.condense
    dofsPerSetup = cell(1, ns);
    for r = 1:ns
        dofsPerSetup{r} = setups(r).dofs;
    end
    Lc = bayoma.condensation_matrix(dofsPerSetup, nDof);
end
nch = zeros(1, ns);
nf = zeros(1, ns);
dTotal = zeros(1, ns);
L = cell(1, ns);
for r = 1:ns
    [nch(r), nf(r)] = size(setups(r).fft);
    dTotal(r) = sum(abs(setups(r).fft(:)).^2);
    setups(r).dofs = setups(r).dofs(:);
    if numel(setups(r).dofs) ~= nch(r)
        error('bayoma:multi_setup:badDofs', '第 %d 个测组的自由度编号数与通道数不一致', r);
    end
    L{r} = bayoma.selection_matrix(setups(r).dofs, nDof);
end

% 频率初值：缺省取第一个测组奇异值谱的最高峰
if isempty(opts.f0)
    D1 = local_psd(setups(1).fft);
    sv1 = bayoma.sv_curves(D1);
    [~, imax] = max(sv1(1, :));
    f0 = setups(1).freqs(imax);
else
    f0 = opts.f0(1);
end

% 初值：按式 (14.43) 的高信噪比近似给出 S0 与 Se，全局振型由式 (14.48) 给出
fval = f0 * ones(1, ns);
zval = 0.01 * ones(1, ns);
S0 = zeros(1, ns);
Se = zeros(1, ns);
A0 = cell(1, ns);
aMax = zeros(1, ns);
Binit = zeros(nDof, nDof);
for r = 1:ns
    D = local_psd(setups(r).fft);
    A0{r} = (sum(D, 3) + sum(D, 3)') / 2;
    [v, d] = eig(A0{r});
    [~, imax] = max(diag(d));
    aMax(r) = max(diag(d));
    tvec = v(:, imax);
    tvec = tvec / max(norm(tvec), 1e-300);
    % 式 (14.43)：Se 由总能量扣除模态部分得到，S0 由 d_k / D_k 的平均得到
    dk = reshape(bayoma.dynamic_factor(fval(r), zval(r), setups(r).freqs, q), 1, []);
    quad = zeros(1, nf(r));
    for k = 1:nf(r)
        quad(k) = real(tvec' * D(:, :, k) * tvec);
    end
    dh = sum(quad);
    Se(r) = max((dTotal(r) - dh) / max((nch(r) - 1) * nf(r), 1), 1e-30);
    S0(r) = max(sum(quad ./ max(dk, 1e-300)) / max(nf(r), 1), 1e-30);
    Binit = Binit + L{r}' * (aMax(r) * eye(nch(r)) - A0{r}) * L{r};
end
Binit = (Binit + Binit') / 2;
if ~isempty(Lc)
    [u, ~] = bayoma.project_condensed((Binit + Binit') / 2, Lc);
else
    [vu, du] = eig((Binit + Binit') / 2);
    [~, imin] = min(diag(du));
    u = vu(:, imin);
    u = u / norm(u);
end

lam = zeros(1, ns);
c = zeros(1, ns);
best = inf;
converged = false;
niter = 0;
while niter < 200
    niter = niter + 1;
    uOld = u;
    S0Old = S0;
    SeOld = Se;
    fOld = fval;
    zOld = zval;

    % Step 1：逐测组更新谱参数
    for r = 1:ns
        tr = L{r} * u;
        if norm(tr) <= 0
            continue;
        end
        D = local_psd(setups(r).fft);
        [fval(r), zval(r), S0(r), Se(r)] = spectrum_subproblem( ...
            setups(r).freqs, D, dTotal(r), tr, fval(r), zval(r), S0(r), Se(r), q, opts.maxiter);
    end

    % 用当前全局振型计算各测组的局部振型、式 (14.32) 的加权谱矩阵与式 (14.41) 的 lam。
    % lam 必须在本轮构造式 (14.40) 的 B 之前更新：若沿用上一轮的 lam，首轮
    % lam 全为零，B 只剩负定项，全局振型会落到无物理意义的方向上。
    tvec = cell(1, ns);
    avec = cell(1, ns);
    for r = 1:ns
        tr = L{r} * u;
        nr = norm(tr);
        if nr <= 0
            error('bayoma:multi_setup:zeroShape', '第 %d 个测组上的全局振型为零', r);
        end
        tvec{r} = tr / nr;
        D = local_psd(setups(r).fft);
        avec{r} = weighted_spectral(D, setups(r).freqs, fval(r), zval(r), S0(r), Se(r), q);
        lam(r) = (tvec{r}' * avec{r} * tvec{r}) / max(Se(r) * (tvec{r}' * tvec{r})^2, 1e-300);
    end

    % Step 2：式 (14.40) 的最小特征值特征向量更新全局振型
    Bmat = zeros(nDof, nDof);
    for r = 1:ns
        nr2 = tvec{r}' * tvec{r};
        Bmat = Bmat + lam(r) * (L{r}' * L{r}) ...
            - (L{r}' * avec{r} * L{r}) / max(nr2 * Se(r), 1e-300);
    end
    Bmat = (Bmat + Bmat') / 2;
    if ~isempty(Lc)
        [u, ~] = bayoma.project_condensed(Bmat, Lc);
    else
        [vu, du] = eig(Bmat);
        [~, imin] = min(diag(du));
        u = vu(:, imin);
        u = u / norm(u);
    end

    % Step 3：式 (14.37) 更新 c_r
    for r = 1:ns
        c(r) = u' * (L{r}' * L{r}) * u;
    end

    val = 0;
    for r = 1:ns
        tr = L{r} * u;
        tn = tr / max(norm(tr), 1e-300);
        D = local_psd(setups(r).fft);
        val = val + setup_nllf(fval(r), zval(r), S0(r), Se(r), setups(r).freqs, D, ...
            dTotal(r), tn, q);
    end
    stepU = max(abs(u - uOld));
    stepS = max([max(abs(S0 - S0Old)), max(abs(Se - SeOld)), ...
        max(abs(fval - fOld) ./ max(abs(fOld), 1e-300)), ...
        max(abs(zval - zOld) ./ max(abs(zOld), 1e-300))]);
    step = max(stepU, stepS);
    if isfinite(best)
        rel = abs(best - val) / max(abs(best), 1);
    else
        rel = inf;
    end
    if isfinite(val) && val < best
        best = val;
    end
    if step < opts.tol || rel < max(opts.tol * 1e-2, 1e-12)
        converged = true;
        break;
    end
end

% 由式 (14.27) 反算各测组的模态力谱 S_r = S0_r / ||L_r u||^2
Sval = zeros(1, ns);
local = cell(1, ns);
for r = 1:ns
    tr = L{r} * u;
    nr2 = tr' * tr;
    Sval(r) = S0(r) / max(nr2, 1e-300);
    local{r} = tr / max(sqrt(nr2), 1e-300);
end

out.f = fval;
out.z = zval;
out.S = Sval;
out.Se = Se;
out.phi = u;
out.local = local;
out.lam = lam;
out.c = c;
out.nllf = best;
out.niter = niter;
out.converged = converged;
end

% ---------------------------------------------------------------- 子函数

function D = local_psd(fft)
%LOCAL_PSD 由缩放 FFT 计算样本谱矩阵序列 D_k = Re(F_k F_k')。
[nch, nf] = size(fft);
D = zeros(nch, nch, nf);
for k = 1:nf
    fk = fft(:, k);
    D(:, :, k) = real(fk * fk');
end
end

function dh = spectral_mean(D, freqs, f, z, t, q)
%SPECTRAL_MEAN 计算式 (14.43) 中的 d_hat = sum_k t' D_k t。
nf = numel(freqs);
tn = t / max(norm(t), 1e-300);
dh = 0;
for k = 1:nf
    dh = dh + real(tn' * D(:, :, k) * tn);
end
end

function [f, z, s0, se] = spectrum_subproblem(freqs, D, dTotal, tr, f0, z0, s00, se0, q, maxiter)
%SPECTRUM_SUBPROBLEM 单个测组的谱参数子问题（式 14.31）。
%
%   四个参数 {f, z, S0, Se} 同时优化，均在无约束的对数空间中搜索。
tn = tr / max(norm(tr), 1e-300);
obj = @(v) setup_nllf(exp(v(1)), exp(v(2)), exp(v(3)), exp(v(4)), freqs, D, ...
    dTotal, tn, q);

v0 = [log(f0); log(z0); log(max(s00, 1e-30)); log(max(se0, 1e-30))];
% 有界搜索：频率限制在频带内，阻尼比限制在物理范围，谱密度限制在正数域。
% 无界搜索会让阻尼比漂到 1、谱密度溢出到 1e60 量级。
lb = [log(freqs(1)); log(1e-6); log(1e-30); log(1e-30)];
ub = [log(freqs(end)); log(0.5); log(1e30); log(1e30)];
v0 = min(max(v0, lb), ub);
opt = optimset('Display', 'off', 'MaxIter', maxiter, 'MaxFunEvals', 20 * maxiter, ...
    'TolFun', 1e-12, 'TolX', 1e-10, 'Algorithm', 'interior-point');
v = fmincon(obj, v0, [], [], [], [], lb, ub, [], opt);

f = exp(v(1));
z = exp(v(2));
s0 = exp(v(3));
se = exp(v(4));
% 频率必须落在频带内，越界时退回初值
if f < freqs(1) || f > freqs(end) || ~isfinite(f)
    f = f0;
end
end

function L = setup_nllf(f, z, s0, se, freqs, D, dTotal, tn, q)
%SETUP_NLLF 单个测组的负对数似然函数（式 14.31）。
%
%   L_r = n N_f ln(pi) + sum_k ln(S0 D_k + Se) + (n-1) N_f ln(Se)
%         + [d - sum_k B_k q_k] / Se
%   其中 B_k = 1/(1 + Se/(S0 D_k))，q_k = t' D_k t。
nf = numel(freqs);
nch = size(D, 1);
if ~isfinite(f) || f <= 0 || ~isfinite(z) || z <= 0 || z >= 1 ...
        || ~isfinite(s0) || s0 <= 0 || ~isfinite(se) || se <= 0
    L = 1e300;
    return;
end
h = bayoma.dynamic_factor(f, z, freqs, q);
h = reshape(h, 1, []);
if any(~isfinite(h))
    L = 1e300;
    return;
end
dk = s0 * h;
p = dk + se;
if any(p <= 0)
    L = 1e300;
    return;
end
w = reshape(1 ./ (1 + se ./ max(dk, 1e-300)), 1, []);
qk = zeros(1, nf);
for k = 1:nf
    qk(k) = real(tn' * D(:, :, k) * tn);
end
L = nch * nf * log(pi) + sum(log(p)) + (nch - 1) * nf * log(se) ...
    + (dTotal - sum(w .* qk)) / se;
if ~isfinite(L)
    L = 1e300;
end
end

function A = weighted_spectral(D, freqs, f, z, S0, Se, q)
%WEIGHTED_SPECTRAL 式 (14.32) 的加权谱矩阵 A。
nf = numel(freqs);
h = reshape(bayoma.dynamic_factor(f, z, freqs, q), 1, []);
dk = S0 * h;
w = reshape(1 ./ (1 + Se ./ max(dk, 1e-300)), 1, []);
A = zeros(size(D, 1));
for k = 1:nf
    A = A + w(k) * D(:, :, k);
end
A = (A + A') / 2;
end

function s = fill_defaults(s, d)
%FILL_DEFAULTS 用缺省值补全结构体字段。
fn = fieldnames(d);
for i = 1:numel(fn)
    if ~isfield(s, fn{i}) || isempty(s.(fn{i}))
        s.(fn{i}) = d.(fn{i});
    end
end
end
