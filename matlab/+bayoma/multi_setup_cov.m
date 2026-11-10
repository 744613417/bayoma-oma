function out = multi_setup_cov(res, setups, opts)
%MULTI_SETUP_COV 多测组问题的后验协方差（式 14.51–14.57）。
%
%   OUT = bayoma.multi_setup_cov(RES, SETUPS) 在 RES（bayoma.multi_setup 的返回
%   结构体）处计算后验协方差。
%
%   SETUPS 为结构体数组，字段 fft、freqs、dofs，与 bayoma.multi_setup 的输入相同。
%
%   OUT = bayoma.multi_setup_cov(..., OPTS) 接受结构体 OPTS，字段：
%       q        模态频响函数的指数，缺省 0
%       relStep  无量纲坐标中的数值微分相对步长，缺省 1e-4
%       verbose  为真时打印无量纲参数向量与 Hessian 的诊断信息，缺省假
%
%   自由参数取
%
%       [f_1..f_ns, z_1..z_ns, S0_1..S0_ns, Se_1..Se_ns, theta]
%
%   其中 S0_r = S_r*||L_r*u||^2 按式 (14.27) 换算；全局振型 u 经切空间参数化满足
%   单位范数约束，各测组的局部振型由 t_r = L_r*u 归一化得到，其单位范数约束随之
%   自动满足。谱参数按最可能值归一，故协方差矩阵对角线元素直接是变异系数的平方。
%
%   返回结构体字段 cov、coefv（结构体，含 f/z/S/Se/phi）、matrix（无量纲参数协
%   方差矩阵）、eigenvalues、nNegative、condition。

if nargin < 3
    opts = struct();
end
if ~isfield(opts, 'q') || isempty(opts.q)
    opts.q = 0;
end
if ~isfield(opts, 'relStep') || isempty(opts.relStep)
    opts.relStep = 1e-4;
end
q = opts.q;

ns = numel(setups);
nDof = numel(res.phi);
u0 = res.phi(:);
u0 = u0 / max(norm(u0), 1e-300);

L = cell(1, ns);
nch = zeros(1, ns);
dTotal = zeros(1, ns);
for r = 1:ns
    nch(r) = size(setups(r).fft, 1);
    dTotal(r) = sum(abs(setups(r).fft(:)).^2);
    L{r} = bayoma.selection_matrix(setups(r).dofs(:), nDof);
end

% 按式 (14.27) 把 S 换算成 S0
S0 = zeros(1, ns);
for r = 1:ns
    tr = L{r} * u0;
    S0(r) = res.S(r) * (tr' * tr);
end

specVec = [res.f(:); res.z(:); S0(:); res.Se(:)];
nSpec = numel(specVec);
specScale = max(abs(specVec), 1e-300);

Ztan = tangent_basis_real(u0);
nTan = size(Ztan, 2);

    function Lv = obj(y)
        sv = y(1:nSpec) .* specScale;
        th = y(nSpec+1:end);
        uu = u0 + Ztan * th;
        uu = uu / max(norm(uu), 1e-300);
        fv = sv(1:ns);
        zv = sv(ns+1:2*ns);
        s0v = sv(2*ns+1:3*ns);
        sev = sv(3*ns+1:4*ns);
        total = 0.0;
        for rr = 1:ns
            tr = L{rr} * uu;
            tn = tr / max(norm(tr), 1e-300);
            total = total + setup_nllf(fv(rr), zv(rr), s0v(rr), sev(rr), ...
                setups(rr).freqs(:), setups(rr).fft, dTotal(rr), tn, q);
        end
        if ~isfinite(total)
            total = 1e300;
        end
        Lv = total;
    end

y0 = [specVec ./ specScale; zeros(nTan, 1)];
nFree = nSpec + nTan;

% 不在求 Hessian 前作额外的无约束精化。交替极小化的输出点已经满足一维扰动检验：
% 沿各谱参数方向作 ±1e-4 相对扰动都使似然上升。似然函数在最可能值附近的变化量
% 已接近双精度舍入水平，再作精化会沿数值噪声漂移，把 Hessian 推出正定区域。

h = opts.relStep;
H = zeros(nFree, nFree);
f0v = obj(y0);
if isfield(opts, 'verbose') && opts.verbose
    fprintf('[multi_setup_cov] nSpec=%d nTan=%d y0=%s\n', nSpec, nTan, ...
        mat2str(y0.', 8));
    fprintf('[multi_setup_cov] specVec=%s specScale=%s\n', ...
        mat2str(specVec.', 8), mat2str(specScale.', 8));
    fprintf('[multi_setup_cov] obj(y0)=%.10f\n', f0v);
end
for i = 1:nFree
    yp = y0; yp(i) = yp(i) + h;
    ym = y0; ym(i) = ym(i) - h;
    H(i, i) = (obj(yp) - 2 * f0v + obj(ym)) / h^2;
    for j = i+1:nFree
        ypp = y0; ypp(i) = ypp(i) + h; ypp(j) = ypp(j) + h;
        ypm = y0; ypm(i) = ypm(i) + h; ypm(j) = ypm(j) - h;
        ymp = y0; ymp(i) = ymp(i) - h; ymp(j) = ymp(j) + h;
        ymm = y0; ymm(i) = ymm(i) - h; ymm(j) = ymm(j) - h;
        H(i, j) = (obj(ypp) - obj(ypm) - obj(ymp) + obj(ymm)) / (4 * h^2);
        H(j, i) = H(i, j);
    end
end

H = (H + H.') / 2;
if isfield(opts, 'verbose') && opts.verbose
    fprintf('[multi_setup_cov] diag(H)=%s\n', mat2str(diag(H).', 6));
    fprintf('[multi_setup_cov] eig(H)=%s\n', ...
        mat2str(sort(real(eig(H)), 'descend').', 6));
end
[V, D] = eig(H);
wv = real(diag(D));
wmax = max(abs(wv));
floorVal = max(wmax * 1e-12, 1e-300);
wv = max(wv, floorVal);
cov = (V ./ wv.') * V.';
cov = (cov + cov.') / 2;

dg = max(real(diag(cov)), 0);
cv = sqrt(dg);

out.cov = cov;
out.matrix = cov;
out.coefv = struct( ...
    'f', cv(1:ns).', ...
    'z', cv(ns+1:2*ns).', ...
    'S', cv(2*ns+1:3*ns).', ...
    'Se', cv(3*ns+1:4*ns).', ...
    'phi', sqrt(max(sum(dg(nSpec+1:end)), 0)));
out.eigenvalues = sort(real(diag(D)), 'descend');
out.nNegative = sum(real(diag(D)) < -1e-8 * max(wmax, 1));
out.condition = wmax / max(min(abs(real(diag(D)))), 1e-300);
end

% ---------------------------------------------------------------- 子函数

function Z = tangent_basis_real(u)
%TANGENT_BASIS_REAL 单位实向量 U 处的切空间正交基。
%
%   返回 n x (n-1) 矩阵 Z，使 (U + Z*theta) 归一化后覆盖单位球面在 U 处的邻域。
u = u(:);
n = numel(u);
if n <= 1
    Z = zeros(n, 0);
    return;
end
e1 = zeros(n, 1);
e1(1) = 1.0;
v = e1 - u;
nv = norm(v);
if nv <= 1e-14
    e1 = zeros(n, 1);
    e1(2) = 1.0;
    v = e1 - u;
    nv = norm(v);
end
v = v / nv;
Hm = eye(n) - 2 * (v * v.');
Z = Hm(:, 2:n);
[Q, ~] = qr(Z, 0);
Z = Q(:, 1:n-1);
end

function Lv = setup_nllf(f, z, s0, se, freqs, fft, dTotal, tn, q)
%SETUP_NLLF 单个测组的负对数似然函数（式 14.31）。
nf = numel(freqs);
nch = size(fft, 1);
if ~isfinite(f) || f <= 0 || ~isfinite(z) || z <= 0 || z >= 1 ...
        || ~isfinite(s0) || s0 <= 0 || ~isfinite(se) || se <= 0
    Lv = 1e300;
    return;
end
h = bayoma.dynamic_factor(f, z, freqs, q);
h = reshape(h, 1, []);
if any(~isfinite(h))
    Lv = 1e300;
    return;
end
dk = s0 * h;
p = dk + se;
if any(p <= 0)
    Lv = 1e300;
    return;
end
w = reshape(1 ./ (1 + se ./ max(dk, 1e-300)), 1, []);
qk = zeros(1, nf);
for k = 1:nf
    fk = fft(:, k);
    qk(k) = real(tn' * (fk * fk') * tn);
end
Lv = nch * nf * log(pi) + sum(log(p)) + (nch - 1) * nf * log(se) ...
    + (dTotal - sum(w .* qk)) / se;
if ~isfinite(Lv)
    Lv = 1e300;
end
end
