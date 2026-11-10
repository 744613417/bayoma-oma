function out = asynchronous_single_mode(freqs, fft, groups, opts)
%ASYNCHRONOUS_SINGLE_MODE 异步数据的单模态贝叶斯识别（课本第 10.4 节）。
%
%   OUT = bayoma.asynchronous_single_mode(FREQS, FFT, GROUPS) 对存在组间不同
%   步的单模态环境振动数据作贝叶斯识别。FREQS 为频带内频率坐标（Hz，Nf x 1），
%   FFT 为 nch x Nf 的缩放 FFT（第 k 列为 F_k，约定 E[F_k F_k'] = E_k），
%   GROUPS 为测组划分的元胞数组，GROUPS{r} 给出第 r 组各通道的索引，各组互不
%   重叠且覆盖全部通道。组内通道同步采集，组间采集时钟不同步。
%
%   OUT = bayoma.asynchronous_single_mode(..., OPTS) 接受结构体 OPTS：
%       q        模态频响函数的指数，缺省 0（0 加速度 / 1 速度 / 2 位移）
%       f0       固有频率初值（Hz），缺省取奇异值谱峰
%       z0       阻尼比初值，缺省 0.01
%       maxIter  交替极小化的外层迭代上限，缺省 200
%       tol      收敛容差，缺省 1e-8
%
%   返回结构体字段：
%       f, z, S   固有频率（Hz）、阻尼比与模态力谱密度
%       Se        各测组的预测误差谱密度 S_er（ng x 1）
%       phi       模态形状（列向量，欧氏范数 1），第 r 段即局部振型 t_r
%       v         复相干矩阵（ng x ng，Hermitian，对角元为 1）
%       nllf      负对数似然函数值，式 (10.42)
%       niter     外层迭代次数
%       converged 是否满足收敛判据
%       coefv     各参数的后验变异系数，字段 f、z、S、Se、phi、v
%       cov       自由参数的后验协方差矩阵（谱参数块按最大后验概率值归一）
%
%   模型按式 (10.43)、(10.45)、(10.50)、(10.51)：模态形状按测组分块
%   u = [t_1; ...; t_q]，组间相关性由复相干矩阵 v 表征，
%
%       E_k = S*D_k*(u~*v*u~') + diag{S_er*I_{n_r}},  r = 1..q
%
%   其中 u~ 为 nch x ng 的分块矩阵，第 r 个行块只有第 r 列为 t_r。似然函数取
%   式 (10.42)，参数集 {f, z, S, {S_er}, u, v}，模态形状受单位范数约束
%   u'u = 1。全部测点同属一个测组时 v 退化为 1，问题退化为第 12 章的同步单模态
%   识别。
%
%   算法与 Python 实现一致：谱参数在对数空间作有界拟牛顿极小化，模态形状在
%   u 的正交补空间内搜索（自动满足单位范数约束），相干矩阵的非对角元用模与
%   相位参数化，模约束在 [0, 0.99] 内。后验协方差取数值 Hessian 的正则化逆。
%
%   参见 bayoma.single_mode、bayoma.uncertainty_law。

if nargin < 4 || isempty(opts)
    opts = struct();
end
opts = fill_defaults(opts, struct('q', 0, 'f0', [], 'z0', 0.01, ...
    'maxIter', 200, 'tol', 1e-8));

q = opts.q;
freqs = double(freqs(:));
fft = double(fft);
[nch, nf] = size(fft);
if numel(freqs) ~= nf
    error('bayoma:asynchronous:badSize', '频率点数必须等于 fft 的列数');
end
groups = validate_groups(groups, nch);
ng = numel(groups);

% 谱密度的上下界按数据量纲取：缩放 FFT 的平均谱能量给出 S 与 S_er 的自然尺度，
% 相差八个数量级足以覆盖实际识别范围。若改用与数据无关的宽定界，拟牛顿法的线
% 搜索会因区间与曲率尺度失配而在初值处停滞。
dscale = mean(abs(fft(:)).^2);
if ~isfinite(dscale) || dscale <= 0
    dscale = 1;
end
smin = max(dscale * 1e-8, 1e-30);
smax = min(dscale * 1e8, 1e30);

[f_cur, z_cur, S_cur, Se_cur, u, v] = initial_guess(freqs, fft, groups, q, opts.f0, opts.z0);

x = log([f_cur; z_cur; S_cur; Se_cur(:)]);
lb = log([max(freqs(1) * 0.5, 1e-12); 1e-6; smin * ones(ng + 1, 1)]);
ub = log([freqs(end) * 2.0; 0.5; smax * ones(ng + 1, 1)]);

optSpec = optimoptions('fmincon', 'Display', 'off', ...
    'Algorithm', 'sqp', 'MaxIterations', 200, ...
    'MaxFunctionEvaluations', 4000, 'OptimalityTolerance', 1e-10, ...
    'StepTolerance', 1e-12);
optShape = optimoptions('fmincon', 'Display', 'off', ...
    'Algorithm', 'sqp', 'MaxIterations', 200, ...
    'MaxFunctionEvaluations', 4000, 'OptimalityTolerance', 1e-10, ...
    'StepTolerance', 1e-12);

nllfval = nllf_async(f_cur, z_cur, S_cur, Se_cur, u, v, freqs, fft, groups, q);
if ~isfinite(nllfval)
    error('bayoma:asynchronous:badStart', '初值处的负对数似然函数非有限');
end

pairlist = pairs_of(ng);
npair = size(pairlist, 1);
step_tol = max(sqrt(opts.tol), 1e-6);
converged = false;
niter = 0;

for niter = 1:opts.maxIter
    nllf_old = nllfval;
    x_old = x;
    u_old = u;
    v_old = v;

    % ---- 谱参数子问题：对数空间的有界拟牛顿极小化 ----
    obj = @(xx) guard(nllf_async(exp(xx(1)), exp(xx(2)), exp(xx(3)), ...
        exp(xx(4:end)), u, v, freqs, fft, groups, q));
    [xn, ~, ~, ~] = fmincon(obj, x, [], [], [], [], lb, ub, [], optSpec);
    if all(isfinite(xn))
        x = xn(:);
    end
    f_cur = exp(x(1));
    z_cur = exp(x(2));
    S_cur = exp(x(3));
    Se_cur = exp(x(4:end));

    % ---- 模态形状与相干矩阵子问题 ----
    Zb = null(u.');
    ntan = size(Zb, 2);
    y0 = [zeros(ntan, 1); zeros(2 * npair, 1)];
    lby = [-ones(ntan, 1); repmat([0; -pi], npair, 1)];
    uby = [ones(ntan, 1); repmat([0.99; pi], npair, 1)];

    objs = @(yy) guard(shape_objective(yy, u, Zb, ntan, pairlist, ng, ...
        f_cur, z_cur, S_cur, Se_cur, freqs, fft, groups, q));
    [yn, ~, ~, ~] = fmincon(objs, y0, [], [], [], [], lby, uby, [], optShape);
    if all(isfinite(yn))
        y = yn(:);
    else
        y = y0;
    end
    u_new = u + Zb * y(1:ntan);
    nrm = norm(u_new);
    if nrm > 1e-12
        u_new = u_new / nrm;
    end
    v_new = polar_to_v(y(ntan + 1:end), pairlist, ng);
    u = u_new;
    v = v_new;

    nllfval = nllf_async(f_cur, z_cur, S_cur, Se_cur, u, v, freqs, fft, groups, q);
    if ~isfinite(nllfval)
        u = u_old;
        v = v_old;
        x = x_old;
        f_cur = exp(x(1));
        z_cur = exp(x(2));
        S_cur = exp(x(3));
        Se_cur = exp(x(4:end));
        nllfval = nllf_old;
        break;
    end

    rel_l = abs(nllf_old - nllfval) / max(abs(nllf_old), 1.0);
    step_spec = max(abs([f_cur / exp(x_old(1)) - 1; z_cur / exp(x_old(2)) - 1]));
    nz = abs(x_old(3:end)) > 0;
    if any(nz)
        step_spec = max(step_spec, max(abs(x(3:end) - x_old(3:end))));
    end
    step = max([step_spec; max(abs(u - u_old)); max(abs(v(:) - v_old(:)))]);
    if rel_l < opts.tol && step < step_tol
        converged = true;
        break;
    end
end

[u, v] = fix_signs(u, v, groups);

out.f = f_cur;
out.z = z_cur;
out.S = S_cur;
out.Se = Se_cur(:);
out.phi = u;
out.groups = groups;
out.v = v;
out.nllf = nllfval;
out.niter = niter;
out.converged = converged;

[cov, coefv] = posterior_cov(f_cur, z_cur, S_cur, Se_cur, u, v, ...
    freqs, fft, groups, q, 1e-4);
out.cov = cov;
out.coefv = coefv;
end

% ----------------------------------------------------------------------
function uop = block_op(u, groups)
%BLOCK_OP 构造式 (10.50) 的分块振型矩阵 u~，第 r 个行块只有第 r 列非零。
u = u(:);
ng = numel(groups);
uop = zeros(numel(u), ng);
for r = 1:ng
    idx = groups{r};
    uop(idx, r) = u(idx);
end
end

% ----------------------------------------------------------------------
function d0 = noise_diagonal(Se, groups)
%NOISE_DIAGONAL 把各测组的预测误差谱密度铺成逐通道的对角元。
nch = sum(cellfun(@numel, groups));
Se = Se(:);
if numel(Se) ~= numel(groups)
    error('bayoma:asynchronous:badSe', '预测误差谱密度的个数应等于测组数');
end
d0 = zeros(nch, 1);
for r = 1:numel(groups)
    d0(groups{r}) = Se(r);
end
end

% ----------------------------------------------------------------------
function val = nllf_async(f, z, S, Se, u, v, freqs, fft, groups, q)
%NLLF_ASYNC 式 (10.42) 的负对数似然函数。
%
%   谱矩阵写作 E_k = D0^(1/2)*(I + S*D_k*M)*D0^(1/2)，其中
%   M = D0^(-1/2)*(u~*v*u~')*D0^(-1/2) 为 Hermitian 半正定矩阵，秩不超过 ng。
%   对 M 作一次特征分解后，行列式与二次型都化为逐频率的标量累加，
%   单次评估为 O(nch^2*Nf)：
%       ln|E_k| = sum_i ln D0_ii + sum_i ln(1 + S*D_k*lambda_i)
%       F_k'*E_k^(-1)*F_k = sum_i |(Q'*D0^(-1/2)*F_k)_i|^2 / (1 + S*D_k*lambda_i)
[~, nf] = size(fft);
val = inf;
if ~(isfinite(f) && f > 0 && isfinite(z) && z > 0 && z < 1 && isfinite(S) && S > 0)
    return;
end
Se = Se(:);
if numel(Se) ~= numel(groups) || any(~isfinite(Se)) || any(Se <= 0)
    return;
end
u = u(:);

d0 = noise_diagonal(Se, groups);
dk = bayoma.dynamic_factor(f, z, freqs, q);
dk = dk(:).';
uop = block_op(u, groups);
r0 = sqrt(d0);
M = (uop * v * uop.') ./ (r0 * r0.');
% 强制 M 精确 Hermitian：只取上三角再补共轭转置。若仅写作 (M + M')/2，舍入
% 会使 ishermitian 判为假，eig 随后按一般矩阵处理，特征向量不再正交，行列式与
% 二次型的分解随之失效。
M = triu(M) + triu(M, 1)';
[Q, Lam] = eig(M);
lam = real(diag(Lam));
denom = 1 + S * (dk.' * lam.');            % nf x ng
if any(~isfinite(denom(:))) || any(denom(:) <= 0)
    return;
end
G = Q' * (fft ./ r0);                      % ng x nf
logdet = nf * sum(log(d0)) + sum(log(denom(:)));
quad = sum(sum((abs(G).^2) ./ denom.'));
val = size(fft, 1) * nf * log(pi) + logdet + quad;
if ~isfinite(val)
    val = inf;
end
end

% ----------------------------------------------------------------------
function val = shape_objective(y, u, Zb, ntan, pairlist, ng, f, z, S, Se, freqs, fft, groups, q)
%SHAPE_OBJECTIVE 给定谱参数时的振动型相关参数子问题。
unew = u + Zb * y(1:ntan);
nrm = norm(unew);
if nrm <= 1e-12
    val = 1e300;
    return;
end
unew = unew / nrm;
vnew = polar_to_v(y(ntan + 1:end), pairlist, ng);
val = nllf_async(f, z, S, Se, unew, vnew, freqs, fft, groups, q);
if ~isfinite(val)
    val = 1e300;
end
end

% ----------------------------------------------------------------------
function val = guard(val)
%GUARD 把非有限的目标函数值替换为惩罚值，供优化器使用。
if ~isfinite(val)
    val = 1e300;
end
end

% ----------------------------------------------------------------------
function idx = pairs_of(ng)
%PAIRS_OF 相干矩阵上三角非对角元的编号，顺序与 nchoosek 一致。
if ng < 2
    idx = zeros(0, 2);
    return;
end
idx = nchoosek(1:ng, 2);
end

% ----------------------------------------------------------------------
function v = polar_to_v(par, pairlist, ng)
%POLAR_TO_V 由模与相位参数构造对角元为 1 的 Hermitian 相干矩阵（式 10.51）。
v = eye(ng);
par = par(:);
for t = 1:size(pairlist, 1)
    m = par(2 * t - 1);
    ph = par(2 * t);
    v(pairlist(t, 1), pairlist(t, 2)) = m * exp(1i * ph);
    v(pairlist(t, 2), pairlist(t, 1)) = conj(v(pairlist(t, 1), pairlist(t, 2)));
end
end

% ----------------------------------------------------------------------
function [u, v] = fix_signs(u, v, groups)
%FIX_SIGNS 固定各测组局部振型的整体符号：绝对值最大的分量为正。
%
%   局部振型的整体符号不可辨识：t_r -> -t_r 同时把相干矩阵第 r 行与第 r 列
%   反号，式 (10.50) 的谱矩阵不变。统一符号后，同一模态在不同数据组、不同程序
%   实现之间可直接比较。
u = u(:);
for r = 1:numel(groups)
    idx = groups{r};
    loc = u(idx);
    [~, k] = max(abs(loc));
    if loc(k) < 0
        u(idx) = -loc;
        v(r, :) = -v(r, :);
        v(:, r) = -v(:, r);
    end
end
v(1:size(v, 1) + 1:end) = 1;
end

% ----------------------------------------------------------------------
function [f0, z0, S0, Se0, u, v] = initial_guess(freqs, fft, groups, q, f0_in, z0_in)
%INITIAL_GUESS 按第 12.3.1 节的高信噪比渐近式构造初值。
%
%   频率初值取奇异值谱峰；各测组的局部振型初值取本组样本谱矩阵之和的主特征
%   向量（组内通道同步，该求和矩阵仍具有同步单模态的秩一结构）；模态力谱密度
%   与各组预测误差谱密度由组内主特征值与迹估计。
[nch, nf] = size(fft);
ng = numel(groups);

s1 = zeros(1, nf);
for k = 1:nf
    s = svd(fft(:, k));
    s1(k) = s(1);
end
[~, kpk] = max(s1);

if isempty(f0_in)
    f0 = freqs(kpk);
else
    f0 = double(f0_in);
    [~, kpk] = min(abs(freqs - f0));
end
z0 = double(z0_in);

u = zeros(nch, 1);
peak = zeros(ng, 1);
tr = zeros(ng, 1);
for r = 1:ng
    idx = groups{r};
    blk = fft(idx, :);
    amat = real(blk * blk');
    amat = (amat + amat') / 2;
    [vec, ev] = eig(amat);
    [ev, order] = sort(real(diag(ev)), 'descend');
    tvec = vec(:, order(1));
    [~, kmax] = max(abs(tvec));
    if tvec(kmax) < 0
        tvec = -tvec;
    end
    u(idx) = tvec;
    peak(r) = ev(1);
    tr(r) = real(trace(amat));
end
u = u / max(norm(u), 1e-300);

dk = bayoma.dynamic_factor(f0, z0, freqs, q);
dk = dk(:);
asum = sum(dk);
Se0 = zeros(ng, 1);
sest = zeros(ng, 1);
for r = 1:ng
    idx = groups{r};
    nr = numel(idx);
    if nr > 1
        Se0(r) = max((tr(r) - peak(r)) / ((nr - 1) * nf), 1e-30);
    else
        Se0(r) = max(0.5 * tr(r) / nf, 1e-30);
    end
    loc = u(idx);
    sest(r) = max(peak(r) - Se0(r) * nf, 0) / max(asum * (loc' * loc), 1e-300);
end
S0 = min(max(mean(sest), 1e-30), 1e30);

v = eye(ng);
end

% ----------------------------------------------------------------------
function [cov, coefv] = posterior_cov(f, z, S, Se, u, v, freqs, fft, groups, q, rel_step)
%POSTERIOR_COV 后验协方差矩阵（数值 Hessian 的正则化逆）。
%
%   自由参数向量取
%       y = [f/f_hat, z/z_hat, S/S_hat, {S_er/S_er_hat}, theta, {Re dv_ij, Im dv_ij}]
%   谱参数按最大后验概率值无量纲化；theta 为模态形状在 u 的正交补空间中的坐标，
%   自动满足单位范数约束；dv_ij 为相干矩阵非对角元的实、虚部。由此得到的
%   协方差矩阵前 3+ng 个对角元即各谱参数变异系数的平方。
ng = numel(groups);
Se = Se(:);
u = u(:);
scale = [f; z; S; Se];
nspec = numel(scale);
Zb = null(u.');
ntan = size(Zb, 2);
pairlist = pairs_of(ng);
npair = size(pairlist, 1);
nfree = nspec + ntan + 2 * npair;

y0 = [ones(nspec, 1); zeros(ntan, 1); zeros(2 * npair, 1)];

obj = @(yy) guard(posterior_objective(yy, scale, nspec, u, Zb, ntan, ...
    pairlist, ng, freqs, fft, groups, q));

steps = rel_step * ones(nfree, 1);
H = numerical_hessian(obj, y0, steps);
cov = regularised_inverse(H);

dg = max(diag(cov), 0);
coefv = struct();
coefv.f = sqrt(dg(1));
coefv.z = sqrt(dg(2));
coefv.S = sqrt(dg(3));
coefv.Se = sqrt(dg(4:nspec));
coefv.phi = sqrt(max(sum(dg(nspec + 1:nspec + ntan)), 0));
coefv.v = sqrt(max(sum(dg(nspec + ntan + 1:end)), 0));
end

% ----------------------------------------------------------------------
function val = posterior_objective(yy, scale, nspec, u, Zb, ntan, pairlist, ng, freqs, fft, groups, q)
%POSTERIOR_OBJECTIVE 无量纲自由参数向量对应的负对数似然函数。
spec = yy(1:nspec) .* scale;
f = spec(1);
z = spec(2);
S = spec(3);
Se = spec(4:nspec);
if ~(f > 0 && z > 0 && z < 1 && S > 0) || any(Se <= 0)
    val = 1e300;
    return;
end
unew = u + Zb * yy(nspec + 1:nspec + ntan);
nrm = norm(unew);
if nrm <= 1e-12
    val = 1e300;
    return;
end
unew = unew / nrm;
v = eye(ng);
par = yy(nspec + ntan + 1:end);
for t = 1:size(pairlist, 1)
    dv = par(2 * t - 1) + 1i * par(2 * t);
    v(pairlist(t, 1), pairlist(t, 2)) = dv;
    v(pairlist(t, 2), pairlist(t, 1)) = conj(dv);
end
val = nllf_async(f, z, S, Se, unew, v, freqs, fft, groups, q);
if ~isfinite(val)
    val = 1e300;
end
end

% ----------------------------------------------------------------------
function H = numerical_hessian(func, y, steps)
%NUMERICAL_HESSIAN 中心差分的 Hessian 矩阵。
n = numel(y);
H = zeros(n, n);
f0 = func(y);
for i = 1:n
    yp = y; yp(i) = yp(i) + steps(i);
    ym = y; ym(i) = ym(i) - steps(i);
    H(i, i) = (func(yp) - 2 * f0 + func(ym)) / steps(i)^2;
end
for i = 1:n - 1
    for j = i + 1:n
        ypp = y; ypp(i) = ypp(i) + steps(i); ypp(j) = ypp(j) + steps(j);
        ypm = y; ypm(i) = ypm(i) + steps(i); ypm(j) = ypm(j) - steps(j);
        ymp = y; ymp(i) = ymp(i) - steps(i); ymp(j) = ymp(j) + steps(j);
        ymm = y; ymm(i) = ymm(i) - steps(i); ymm(j) = ymm(j) - steps(j);
        val = (func(ypp) - func(ypm) - func(ymp) + func(ymm)) / (4 * steps(i) * steps(j));
        H(i, j) = val;
        H(j, i) = val;
    end
end
end

% ----------------------------------------------------------------------
function cov = regularised_inverse(H)
%REGULARISED_INVERSE 对称矩阵的正则化逆。
%
%   数值微分的 Hessian 在弱可辨识方向上的特征值接近零，直接求逆会把舍入误差放大
%   成虚假的巨大方差；带截断阈值的伪逆则把这类方向判成零方差，给出参数精确已知
%   的错误结论。这里按最大特征值取相对下限，只把非正与近零特征值抬到该下限，
%   使弱可辨识方向得到有限的、量级合理的大方差。
H = (H + H.') / 2;
[V, D] = eig(H);
w = real(diag(D));
wmax = max(abs(w));
if ~isfinite(wmax) || wmax <= 0
    error('bayoma:asynchronous:badHessian', '负对数似然的 Hessian 非正定');
end
floor_ = wmax * 1e-12;
w = max(w, floor_);
cov = V * diag(1 ./ w) * V.';
cov = (cov + cov.') / 2;
end

% ----------------------------------------------------------------------
function groups = validate_groups(groups, nch)
%VALIDATE_GROUPS 校验测组划分并统一为索引列向量的元胞数组。
if isnumeric(groups)
    groups = {groups(:)};
end
if ~iscell(groups) || isempty(groups)
    error('bayoma:asynchronous:badGroups', '测组划分必须是索引向量的元胞数组');
end
out = cell(numel(groups), 1);
used = false(nch, 1);
for r = 1:numel(groups)
    idx = double(groups{r}(:));
    if isempty(idx)
        error('bayoma:asynchronous:emptyGroup', '第 %d 个测组不含任何通道', r);
    end
    if any(idx < 1) || any(idx > nch) || any(idx ~= round(idx))
        error('bayoma:asynchronous:badIndex', '第 %d 个测组的通道索引越界', r);
    end
    if any(used(idx))
        error('bayoma:asynchronous:overlap', '通道 %d 同时属于多个测组', find(used(idx), 1));
    end
    used(idx) = true;
    out{r} = idx;
end
if ~all(used)
    error('bayoma:asynchronous:incomplete', '测组划分必须覆盖全部通道');
end
groups = out;
end

% ----------------------------------------------------------------------
function s = fill_defaults(s, d)
%FILL_DEFAULTS 用缺省值补齐结构体字段。
names = fieldnames(d);
for i = 1:numel(names)
    if ~isfield(s, names{i}) || isempty(s.(names{i}))
        s.(names{i}) = d.(names{i});
    end
end
end
