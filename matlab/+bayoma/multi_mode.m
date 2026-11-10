function out = multi_mode(D, fft, freqs, m, opts)
%MULTI_MODE 多模态贝叶斯识别（课本第 13 章）。
%
%   OUT = bayoma.multi_mode(D, FFT, FREQS, M) 执行 M 阶模态的贝叶斯识别。
%   D 为 nch x nch x Nf 的样本谱矩阵 D_k = Re(F_k F_k')，FFT 为 nch x Nf 的
%   缩放 FFT，FREQS 为频率坐标（Hz）。
%
%   OUT = bayoma.multi_mode(..., OPTS) 接受结构体 OPTS，字段：
%       f0        各阶频率初值（Hz），长度 m，缺省取奇异值谱峰
%       q         模态频响函数的指数，缺省 0
%       maxiter   交替极小化轮数上限，缺省 40
%       tol       收敛容差，缺省 1e-7
%
%   返回结构体字段：
%       f, z      各阶频率与阻尼比
%       S         模态力谱密度矩阵（m x m，Hermitian）
%       Se        通道噪声谱
%       phi       振型，nch x m，每列欧氏范数为 1
%       basis     模态形状子空间基 B0
%       alpha     振型在子空间基下的坐标
%       coefv     各参数的后验变异系数
%       cmat      无量纲参数协方差矩阵
%       nllf      负对数似然函数值
%       niter     迭代轮数
%       converged 是否满足收敛判据
%
%   算法按第 13.5 节的交替极小化：谱参数 {f, z, S, Se} 与振型坐标 alpha 交替
%   更新，最后对谱参数、坐标与子空间超角作联合精化。

if nargin < 5
    opts = struct();
end
opts = fill_defaults(opts, struct('f0', [], 'q', 0, 'maxiter', 40, 'tol', 1e-7));

[nch, nf] = size(fft);
m = round(m);
m0 = min(m, nch);
dTotal = sum(abs(fft(:)).^2);
q = opts.q;

% 子空间基：A0 = sum_k D_k 的前 m0 个最大特征向量（第 13.6.1 节）
A0 = sum(D, 3);
A0 = (A0 + A0.') / 2;
[vecs, vals] = eig(A0);
[~, order] = sort(diag(vals), 'descend');
basis = orth_complete(vecs(:, order(1:m0)));

% 频率初值：第一条奇异值谱曲线上的显著谱峰
if isempty(opts.f0)
    sv = bayoma.sv_curves(D);
    fstart = initial_frequencies(freqs, sv(1, :), m);
else
    fstart = sort(opts.f0(:)).';
    while numel(fstart) < m
        fstart(end+1) = fstart(end) * 1.01; %#ok<AGROW>
    end
    fstart = fstart(1:m);
end
zstart = 0.01 * ones(1, m);

alpha = initial_coordinates(D, basis, freqs, fstart);
scales = initial_scales(D, freqs, fstart, zstart, q);
Se = noise_guess(D, dTotal, m0, nf);
corrParams = 0.1 * ones(1, m * (m - 1));

best = inf;
converged = false;
niter = 0;
xs = pack_spectrum(fstart, zstart, Se, scales, corrParams);

while niter < opts.maxiter
    niter = niter + 1;
    xsOld = xs;
    aOld = alpha(:);

    objS = @(v) spectrum_objective(v, alpha, basis, fft, freqs, dTotal, q, m, m0);
    xs = fminunc(objS, xs, optimset('Display', 'off', 'MaxIter', 800, ...
        'MaxFunEvals', 8000, 'TolFun', 1e-12, 'TolX', 1e-10, 'LargeScale', 'off'));

    objA = @(v) alpha_objective(v, xs, basis, fft, freqs, dTotal, q, m, m0);
    a = fminunc(objA, pack_alpha(alpha), optimset('Display', 'off', 'MaxIter', 400, ...
        'MaxFunEvals', 4000, 'TolFun', 1e-12, 'TolX', 1e-10, 'LargeScale', 'off'));
    alpha = normalise_columns(unpack_alpha(a, m0, m));

    val = spectrum_objective(xs, alpha, basis, fft, freqs, dTotal, q, m, m0);
    % 相对下降量必须以更新前的历史最优值为基准，否则首轮恒为零而误判收敛
    if isfinite(best)
        rel = abs(best - val) / max(abs(best), 1);
    else
        rel = inf;
    end
    if isfinite(val) && val < best
        best = val;
    end
    step = max(abs([xs - xsOld; alpha(:) - aOld]));
    if step < opts.tol || rel < max(opts.tol * 1e-2, 1e-10)
        converged = true;
        break;
    end
end

% 联合精化：谱参数、坐标与子空间超角同时调整
try
    [xs2, alpha2, basis2, val2] = joint_polish(xs, alpha, basis, fft, freqs, ...
        dTotal, q, m, m0);
    if isfinite(val2) && val2 <= best + 1e-6
        xs = xs2; alpha = alpha2; basis = basis2; best = val2;
        converged = true;
    end
catch
end

[fv, zv, sev, scv, cpv] = unpack_spectrum(xs, m);
Smat = bayoma.structured_modal_psd(cpv, m, scv);
phi = basis * alpha;

% 按频率升序排列，使结果与初值顺序无关
[fs, order] = sort(fv);
fv = fs;
zv = zv(order);
scv = scv(order);
phi = phi(:, order);
alpha = alpha(:, order);
Smat = Smat(order, order);

% 相位基准归一：取每列模最大分量为正实数，并把相位差并回 S。
% phi、alpha、S 三者必须同步变换，否则后续似然计算中互不一致。
theta = zeros(1, m);
for i = 1:m
    [~, idx] = max(abs(phi(:, i)));
    if abs(phi(idx, i)) > 0
        theta(i) = angle(phi(idx, i));
    end
end
phase = exp(1i * theta);
alpha = alpha .* repmat(conj(phase), m0, 1);
phi = basis * alpha;
Smat = (phase(:) * ones(1, m)) .* Smat .* (ones(m, 1) * conj(phase(:)).');
for i = 1:m
    nr = norm(phi(:, i));
    if nr > 0
        phi(:, i) = phi(:, i) / nr;
        alpha(:, i) = alpha(:, i) / nr;
    end
end

out.f = fv;
out.z = zv;
out.S = Smat;
out.Se = sev;
out.phi = phi;
out.basis = basis;
out.alpha = alpha;
out.nllf = best;
out.niter = niter;
out.converged = converged;
out.subspace_rank = m0;

cv = covariance_estimate(fv, zv, Smat, sev, phi, fft, freqs, dTotal, q, m, m0, basis);
out.coefv = cv;
out.cmat = cv.matrix;
% 完整协方差（谱参数块 + 各阶振型切空间块）供联合协方差装配；cmat 仍只保留
% 谱参数块，保持与参考 pcode 的输出约定一致。
out.full = cv.full;
out.cov_phi = cv.cov_phi;
out.sizes = cv.sizes;
out.n_off = cv.n_off;
out.complex = cv.complex;
end

% ---------------------------------------------------------------- 子函数

function v = pack_spectrum(f, z, Se, scales, corrParams)
%PACK_SPECTRUM 把谱参数打包为无约束向量（各正量取对数）。
v = [log(f(:)); log(z(:)); log(Se); log(scales(:)); corrParams(:)];
end

function [f, z, Se, scales, corrParams] = unpack_spectrum(v, m)
%UNPACK_SPECTRUM 由无约束向量还原谱参数。
f = exp(v(1:m)).';
z = exp(v(m+1:2*m)).';
Se = exp(v(2*m+1));
scales = exp(v(2*m+2:3*m+1)).';
corrParams = v(3*m+2:end).';
end

function L = eval_nllf(v, alpha, basis, fft, freqs, dTotal, q, m)
%EVAL_NLLF 由谱参数向量与坐标计算降维似然函数。
[f, z, Se, scales, cp] = unpack_spectrum(v, m);
if any(~isfinite(f)) || any(f <= 0) || any(z <= 1e-6) || any(z >= 0.5) ...
        || ~isfinite(Se) || Se <= 0 || any(scales <= 0) || any(~isfinite(cp))
    L = 1e300;
    return;
end
S = bayoma.structured_modal_psd(cp, m, scales);
L = bayoma.nllf_multi(f, z, S, Se, alpha, basis, fft, freqs, dTotal, q);
if ~isfinite(L)
    L = 1e300;
end
end

function L = spectrum_objective(v, alpha, basis, fft, freqs, dTotal, q, m, m0) %#ok<INUSD>
%SPECTRUM_OBJECTIVE 谱参数子问题的目标函数。
L = eval_nllf(v, alpha, basis, fft, freqs, dTotal, q, m);
end

function L = alpha_objective(v, xs, basis, fft, freqs, dTotal, q, m, m0)
%ALPHA_OBJECTIVE 振型坐标子问题的目标函数。
%   模态形状的坐标可为复数，而 fminunc 只处理实变量，故实部与虚部分开打包。
alpha = unpack_alpha(v, m0, m);
L = eval_nllf(xs, alpha, basis, fft, freqs, dTotal, q, m);
end

function v = pack_alpha(alpha)
%PACK_ALPHA 把复数坐标矩阵打包为实向量（先实部后虚部，均按列优先）。
v = [real(alpha(:)); imag(alpha(:))];
end

function alpha = unpack_alpha(v, m0, m)
%UNPACK_ALPHA 由实向量还原复数坐标矩阵。
n = m0 * m;
alpha = reshape(v(1:n), m0, m) + 1i * reshape(v(n+1:2*n), m0, m);
end

function f = initial_frequencies(freqs, main, m)
%INITIAL_FREQUENCIES 在第一条奇异值谱曲线上取显著谱峰作为频率初值。
fr = freqs(:);
nf = numel(fr);
df = fr(2) - fr(1);
span = fr(end) - fr(1);
minSep = max(3 * df, 0.02 * span);

cand = [];
prom = [];
for i = 2:nf-1
    if main(i) > main(i-1) && main(i) >= main(i+1)
        cand(end+1) = i; %#ok<AGROW>
        lmin = min(main(max(1, i-200):i));
        rmin = min(main(i:min(nf, i+200)));
        prom(end+1) = main(i) - max(lmin, rmin); %#ok<AGROW>
    end
end
chosen = [];
if ~isempty(cand)
    [~, ord] = sort(prom, 'descend');
    for j = ord
        if numel(chosen) >= m
            break;
        end
        if isempty(chosen) || all(abs(fr(cand(j)) - fr(chosen)) >= minSep)
            chosen(end+1) = cand(j); %#ok<AGROW>
        end
    end
end
if numel(chosen) < m
    [~, base] = max(main);
    if isempty(chosen)
        chosen = base;
    end
    step = 1;
    while numel(chosen) < m && step * minSep < span
        for s = [1, -1]
            idx = base + s * step;
            if idx >= 1 && idx <= nf
                chosen(end+1) = idx; %#ok<AGROW>
            end
            if numel(chosen) >= m
                break;
            end
        end
        step = step + 1;
    end
end
f = sort(fr(chosen(1:m))).';
for i = 2:numel(f)
    if f(i) - f(i-1) < minSep
        f(i) = min(f(i-1) + minSep, fr(end));
    end
end
end

function alpha = initial_coordinates(D, basis, freqs, f0)
%INITIAL_COORDINATES 各阶初始频率处主特征向量在子空间内的坐标，再作正交化。
m0 = size(basis, 2);
m = numel(f0);
cols = zeros(m0, m);
for i = 1:m
    [~, idx] = min(abs(freqs - f0(i)));
    dk = (D(:, :, idx) + D(:, :, idx).') / 2;
    [v, ~] = eig(dk);
    cols(:, i) = basis.' * v(:, end);
end
if m > 1
    s = svd(cols);
    degenerate = s(end) < 1e-3 * s(1);
else
    degenerate = false;
end
if degenerate
    A0 = sum(D, 3);
    A0 = (A0 + A0.') / 2;
    [v, d] = eig(A0);
    [~, ord] = sort(diag(d), 'descend');
    cols = basis.' * v(:, ord(1:m));
end
[q, r] = qr(cols, 0);
sg = sign(diag(r));
sg(sg == 0) = 1;
alpha = q .* repmat(sg.', m0, 1);
end

function scales = initial_scales(D, freqs, f0, z, q)
%INITIAL_SCALES 各阶模态力谱密度尺度初值（第 13.6.1 节的高信噪比近似）。
m = numel(f0);
scales = zeros(1, m);
for i = 1:m
    [~, idx] = min(abs(freqs - f0(i)));
    dk = (D(:, :, idx) + D(:, :, idx).') / 2;
    peak = max(eig(dk));
    scales(i) = sqrt(max(peak, 1e-30)) * 2 * z(i) * (2 * pi * f0(i))^q;
end
scales = max(scales, 1e-12);
end

function Se = noise_guess(D, dTotal, m0, nf)
%NOISE_GUESS 噪声谱初值，按式 (13.60) 的思路用高阶奇异值估计。
nch = size(D, 1);
A0 = sum(D, 3);
vals = sort(eig((A0 + A0.') / 2), 'descend');
d0 = sum(vals(1:m0));
Se = max((dTotal - d0) / max((nch - m0) * nf, 1), 1e-30);
end

function alpha = normalise_columns(alpha)
%NORMALISE_COLUMNS 把各列缩放到欧氏范数 1。
m = size(alpha, 2);
for i = 1:m
    nr = norm(alpha(:, i));
    if nr > 0
        alpha(:, i) = alpha(:, i) / nr;
    end
end
end

function [xs, alpha, basis, best] = joint_polish(xs, alpha, basis, fft, freqs, ...
    dTotal, q, m, m0)
%JOINT_POLISH 谱参数、振型坐标与子空间超角的联合精化（式 13.22–13.23）。
n = size(basis, 1);
nAng = m0 * (n - m0);
if nAng < 1
    best = eval_nllf(xs, alpha, basis, fft, freqs, dTotal, q, m);
    return;
end
bFull = full_basis(basis);
nSpec = numel(xs);
v0 = [xs; pack_alpha(alpha); zeros(nAng, 1)];

    function L = obj(v)
        bs = rotate_basis(bFull, m0, v(nSpec+2*m0*m+1:end));
        a = unpack_alpha(v(nSpec+1:nSpec+2*m0*m), m0, m);
        L = eval_nllf(v(1:nSpec), a, bs, fft, freqs, dTotal, q, m);
    end

% 联合精化用拟牛顿法。fminsearch 在十余个参数上收敛太慢，常停在后验协方差
% 所需的驻点之外（Hessian 出现负特征值），故改用 fminunc。
v = fminunc(@obj, v0, optimset('Display', 'off', 'MaxIter', 400, ...
    'MaxFunEvals', 20000, 'TolFun', 1e-14, 'TolX', 1e-12, ...
    'LargeScale', 'off'));
xs = v(1:nSpec);
alpha = normalise_columns(unpack_alpha(v(nSpec+1:nSpec+2*m0*m), m0, m));
basis = rotate_basis(bFull, m0, v(nSpec+2*m0*m+1:end));
best = obj(v);
end

function b = full_basis(basis)
%FULL_BASIS 把列正交基扩充为整个空间的方阵正交基。
n = size(basis, 1);
m0 = size(basis, 2);
if m0 >= n
    b = basis(:, 1:n);
    return;
end
comp = null(basis.');
b = [basis, comp(:, 1:n-m0)];
end

function b = rotate_basis(bfull, m0, angles)
%ROTATE_BASIS 由超角构造旋转后的子空间基（式 13.22–13.23）。
n = size(bfull, 1);
nOut = n - m0;
rot = eye(n);
idx = 1;
for i = m0:-1:1
    for j = nOut:-1:1
        p = m0 + j;
        c = cos(angles(idx));
        s = sin(angles(idx));
        g = eye(n);
        g(i, i) = c; g(i, p) = -s;
        g(p, i) = s; g(p, p) = c;
        rot = rot * g;
        idx = idx + 1;
    end
end
b = bfull * rot;
b = b(:, 1:m0);
end

function out = covariance_estimate(f, z, S, Se, phi, fft, freqs, dTotal, q, m, m0, basis)
%COVARIANCE_ESTIMATE 后验协方差的数值估计（切空间参数化）。
%
%   自由参数取无量纲谱参数与振型在切空间中的坐标：
%
%       谱参数 = [f, z, diag(S), Re(offdiag S), Im(offdiag S), Se]，各自按最可能值归一；
%       振型   = 子空间坐标 alpha 的切空间坐标 theta，alpha_i(theta) 归一化后使用。
%
%   因为谱参数按最可能值归一，协方差矩阵对角线元素直接是变异系数的平方。
%   振型的变异系数取振型后验协方差矩阵迹的平方根（课本 11.3.3 节）；子空间基
%   列正交，故迹可由 alpha 空间的迹直接得到。
%
%   振型初值必须取最可能值处的子空间坐标，不能取零：取零时振型为零向量，似然
%   在该点对振型方向没有定义，Hessian 无意义。

nOff = m * (m - 1) / 2;
specVec = [f(:); z(:); real(diag(S)); real(offdiag_vec(S)); imag(offdiag_vec(S)); Se];
nSpec = numel(specVec);
specScale = max(abs(specVec), 1e-300);

alpha0 = basis.' * phi;
for i = 1:m
    nr = norm(alpha0(:, i));
    if nr > 0
        alpha0(:, i) = alpha0(:, i) / nr;
    end
end

% 每阶振型在归一化约束下剩余 m0-1 个自由度
Ztan = cell(1, m);
nTan = 0;
for i = 1:m
    Ztan{i} = tangent_basis(alpha0(:, i));
    nTan = nTan + size(Ztan{i}, 2);
end

    function L = obj(y)
        sv = y(1:nSpec) .* specScale;
        fv = sv(1:m).';
        zv = sv(m+1:2*m).';
        sv3 = sv(2*m+1:3*m);
        reOff = sv(3*m+1:3*m+nOff);
        imOff = sv(3*m+nOff+1:3*m+2*nOff);
        sev = sv(end);
        Sm = diag(sv3);
        k = 0;
        for ii = 2:m
            for jj = 1:ii-1
                k = k + 1;
                Sm(ii, jj) = reOff(k) + 1i * imOff(k);
                Sm(jj, ii) = reOff(k) - 1i * imOff(k);
            end
        end

        al = alpha0;
        k = nSpec;
        for ii = 1:m
            nt = size(Ztan{ii}, 2);
            th = y(k+1:k+nt);
            k = k + nt;
            ai = alpha0(:, ii) + Ztan{ii} * th;
            nr = norm(ai);
            if nr > 0
                al(:, ii) = ai / nr;
            end
        end

        L = bayoma.nllf_multi(fv, zv, Sm, sev, al, basis, fft, freqs, dTotal, q);
        if ~isfinite(L)
            L = 1e300;
        end
    end

y0 = [ones(nSpec, 1); zeros(nTan, 1)];
h = 1e-4;
nFree = nSpec + nTan;

% 求 Hessian 之前先把自由参数精化到驻点。交替极小化与联合精化给出的点可能仍在
% 下降途中，此时 Hessian 会出现负特征值，取逆得到的方差会被整体放大。
yRef = fminunc(@obj, y0, optimset('Display', 'off', 'MaxIter', 2000, ...
    'MaxFunEvals', 60000, 'TolFun', 1e-16, 'TolX', 1e-12, ...
    'LargeScale', 'off'));
if isfinite(obj(yRef)) && obj(yRef) <= obj(y0) + 1e-6
    y0 = yRef;
end

H = zeros(nFree, nFree);
f0 = obj(y0);
for i = 1:nFree
    yp = y0; yp(i) = yp(i) + h;
    ym = y0; ym(i) = ym(i) - h;
    H(i, i) = (obj(yp) - 2 * f0 + obj(ym)) / h^2;
    for j = i+1:nFree
        ypp = y0; ypp(i) = ypp(i) + h; ypp(j) = ypp(j) + h;
        ypm = y0; ypm(i) = ypm(i) + h; ypm(j) = ypm(j) - h;
        ymp = y0; ymp(i) = ymp(i) - h; ymp(j) = ymp(j) + h;
        ymm = y0; ymm(i) = ymm(i) - h; ymm(j) = ymm(j) - h;
        H(i, j) = (obj(ypp) - obj(ypm) - obj(ymp) + obj(ymm)) / (4 * h^2);
        H(j, i) = H(i, j);
    end
end

% 对称化后作特征分解求逆；把接近零的特征值抬到下限，避免把弱可辨识方向判成零方差
[V, W] = eig((H + H.') / 2);
w = real(diag(W));
wmax = max(abs(w));
if wmax <= 0
    wmax = 1.0;
end
floorW = wmax * 1e-12;
w(w < floorW) = floorW;
C = V * diag(1 ./ w) * V.';

% 谱参数已按最可能值归一，故对角线元素即变异系数的平方
Cs = C(1:nSpec, 1:nSpec);

out.matrix = Cs;

% 结果输出扩充所需的补充字段
%   full      nFree x nFree 的完整协方差矩阵（谱参数块 + 各阶振型切空间块）
%   sizes     各阶振型在切空间中的实维数，实数基为 m0-1，复数基为 2*m0-2
%   cov_phi   各阶振型在通道空间中的协方差矩阵，由切空间块经通道映射得到
%   n_off     模态力谱矩阵非对角元个数
%   complex   振型是否为复数（单测组多阶时）
sizes = zeros(1, m);
covPhi = cell(1, m);
pos = nSpec;
for i = 1:m
    nt = size(Ztan{i}, 2);
    sizes(i) = nt;
    blk = C(pos + 1:pos + nt, pos + 1:pos + nt);
    mat = bayoma.channel_map(Ztan{i}, basis);
    covPhi{i} = mat * blk * mat.';
    pos = pos + nt;
end
out.full = C;
out.sizes = sizes;
out.cov_phi = covPhi;
out.n_off = nOff;
out.complex = ~isreal(phi);

out.f = sqrt(max(real(diag(Cs(1:m, 1:m))), 0)).';
out.z = sqrt(max(real(diag(Cs(m+1:2*m, m+1:2*m))), 0)).';
sv3 = sqrt(max(real(diag(Cs(2*m+1:3*m, 2*m+1:3*m))), 0));
out.Sii = sv3.';
out.Se = sqrt(max(real(Cs(end, end)), 0));
out.phi = sqrt(max(real(trace(C(nSpec+1:end, nSpec+1:end))), 0));
out.rms = 0.5 * sqrt(out.f.^2 + out.z.^2 + sv3.'.^2);
end

function Z = tangent_basis(a)
%TANGENT_BASIS 构造单位向量 A 处的切空间正交基。
%
%   返回矩阵 Z，使 A + Z*theta 归一化后覆盖单位球面在 A 处的邻域。实数 A 的
%   切空间维数为 n-1；复数 A 按实部、虚部展开为 2n 维实向量，切空间维数为
%   2n-1，其中「整体相位」方向 i*A 上似然不变（相位差可被模态力谱矩阵吸收），
%   属不可辨识方向，必须剔除，故实际返回 2n-2 列。
%
%   若把复数 A 当作 n 维实向量处理，会漏掉振型的虚部自由度，并使切空间包含
%   相位方向，后验协方差在该方向上的近零曲率会把其余参数的方差一并放大。

a = a(:);
n = numel(a);
if isreal(a)
    m = n;
    u = a;
else
    m = 2 * n;
    u = [real(a); imag(a)];
end
if m <= 1
    Z = zeros(n, 0);
    return;
end

% 在 R^m 中构造与 u 正交的 m-1 维正交基（Householder 把 u 映到 e1）
e1 = zeros(m, 1);
e1(1) = 1.0;
v = e1 - u;
nv = norm(v);
if nv <= 1e-14
    e1 = zeros(m, 1);
    e1(min(2, m)) = 1.0;
    v = e1 - u;
    nv = norm(v);
end
v = v / nv;
Hm = eye(m) - 2 * (v * v.');
Zr = Hm(:, 2:m);                       % m x (m-1)，列与 u 正交

% 剔除整体相位方向：复数情形下该方向在 R^{2n} 中为 [-Im a; Re a]
if ~isreal(a)
    ph = [-imag(a); real(a)];
    nph = norm(ph);
    if nph > 1e-12
        ph = ph / nph;
        Zr = Zr - ph * (ph.' * Zr);
        [Q, ~] = qr(Zr, 0);
        Zr = Q(:, 1:min(size(Zr, 2), m - 2));
    end
end

if isreal(a)
    Z = Zr;
else
    Z = Zr(1:n, :) + 1i * Zr(n+1:2*n, :);
end
end

function v = offdiag_vec(S)
%OFFDIAG_VEC 提取严格上三角元素（按行优先）。
m = size(S, 1);
v = zeros(1, m * (m - 1) / 2);
k = 0;
for i = 1:m
    for j = i+1:m
        k = k + 1;
        v(k) = S(i, j);
    end
end
end

function b = orth_complete(basis)
%ORTH_COMPLETE 对列作正交归一化。
[Q, ~] = qr(basis, 0);
b = Q;
end

function s = fill_defaults(s, d)
%FILL_DEFAULTS 用缺省值补齐结构体字段。
names = fieldnames(d);
for i = 1:numel(names)
    if ~isfield(s, names{i}) || isempty(s.(names{i}))
        s.(names{i}) = d.(names{i});
    end
end
end
