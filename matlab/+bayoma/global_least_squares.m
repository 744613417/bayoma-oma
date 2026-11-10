function out = global_least_squares(tildeShapes, selection, weights, opts)
%GLOBAL_LEAST_SQUARES 多测组全局最小二乘（课本 14.2 节，式 14.3--14.20）
%   out = bayoma.global_least_squares(tildeShapes, selection)
%   out = bayoma.global_least_squares(tildeShapes, selection, weights, opts)
%
%   输入
%     tildeShapes : 元胞数组，各测组识别并归一的局部振型 ~t_r
%     selection   : 元胞数组，各测组选择矩阵 L_r（n_r x n）
%     weights     : 权重 w_r，缺省等权 1/q
%     opts        : 结构体，字段 condense（缺省 true）、maxIter（200）、tol（1e-10）
%   输出
%     out.u 全局振型、out.c 各测组范数平方、out.lam 各测组乘子、
%     out.nIter、out.converged、out.condensed、out.info
%
%   迭代按 14.2.3 节：由当前 u 更新 c_r（式 14.6）与 lambda_r，再解
%   A u + b = lambda_0 u（式 14.11--14.12）；启用凝聚时在 Lc 子空间内
%   解式 14.20。局部振型的整体正负号不可辨识，每次迭代按当前的 u 把
%   它们对齐到 t~_r' * L_r * u > 0。

if nargin < 3 || isempty(weights)
    weights = ones(1, numel(tildeShapes)) / numel(tildeShapes);
end
if nargin < 4
    opts = struct();
end
opts = fill_defaults(opts);

nSetup = numel(tildeShapes);
nDof = size(selection{1}, 2);
w = weights(:).' / sum(weights);

A0 = zeros(nDof, nDof);
for r = 1:nSetup
    L = selection{r};
    t = tildeShapes{r}(:);
    A0 = A0 + w(r) * (L' * (eye(numel(t)) - t * t') * L);
end
[V, Dg] = eig((A0 + A0') / 2);
[dg, idx] = sort(real(diag(Dg)), 'ascend');
u = V(:, idx(1));
u = u / norm(u);

Lc = [];
info = struct('reference', [], 'nonReference', [], 'nCondensed', nDof);
if opts.condense
    dofsPerSetup = cell(1, nSetup);
    for r = 1:nSetup
        dofsPerSetup{r} = find(any(selection{r} ~= 0, 1));
    end
    [Lc, info] = bayoma.condensation_matrix(dofsPerSetup, nDof);
end

c = zeros(nSetup, 1);
lam = zeros(nSetup, 1);
converged = false;
nIter = 0;
for nIter = 1:opts.maxIter
    for r = 1:nSetup
        tr = selection{r} * u;
        c(r) = real(tr' * tr);
        if c(r) > 0
            lam(r) = w(r) * (1 - abs(tildeShapes{r}(:)' * tr) / sqrt(c(r)));
        else
            lam(r) = w(r);
        end
    end
    lam = min(max(lam, 0), w(:));

    Amat = zeros(nDof, nDof);
    bvec = zeros(nDof, 1);
    for r = 1:nSetup
        L = selection{r};
        t = tildeShapes{r}(:);
        tr = L * u;
        sgn = 1;
        if t' * tr < 0
            sgn = -1;
        end
        Amat = Amat + (w(r) + lam(r)) * (L' * L);
        bvec = bvec - w(r) * sqrt(max(c(r), 0)) * (L' * (sgn * t));
    end

    if ~isempty(Lc) && size(Lc, 2) > 0
        Ac = Lc' * Amat * Lc;
        bc = Lc' * bvec;
        nc = size(Ac, 2);
        Dm = zeros(2 * nc, 2 * nc);
        Dm(1:nc, 1:nc) = Ac;
        Dm(1:nc, nc+1:end) = bc * bc';
        Dm(nc+1:end, 1:nc) = eye(nc);
        Dm(nc+1:end, nc+1:end) = Ac;
        uNew = Lc * smallest_real_vector(Dm, nc);
    else
        Dm = zeros(2 * nDof, 2 * nDof);
        Dm(1:nDof, 1:nDof) = Amat;
        Dm(1:nDof, nDof+1:end) = bvec * bvec';
        Dm(nDof+1:end, 1:nDof) = eye(nDof);
        Dm(nDof+1:end, nDof+1:end) = Amat;
        uNew = smallest_real_vector(Dm, nDof);
    end

    nr = norm(uNew);
    if nr > 0
        uNew = uNew / nr;
    end
    stepU = max(abs(uNew - u));
    u = uNew;
    if stepU < opts.tol
        converged = true;
        break;
    end
end

out = struct('u', u, 'c', c, 'lam', lam, 'nIter', nIter, ...
             'converged', converged, 'condensed', ~isempty(Lc) && size(Lc, 2) > 0, ...
             'info', info);
end

function v = smallest_real_vector(M, nHead)
[V, Dg] = eig(M);
dg = diag(Dg);
scale = max(abs(real(dg)), 1.0);
mask = abs(imag(dg)) <= 1e-8 * scale;
index = find(mask);
if isempty(index)
    index = (1:numel(dg)).';
end
[~, loc] = min(real(dg(index)));
v = real(V(1:nHead, index(loc)));
end

function s = fill_defaults(s)
if ~isfield(s, 'condense') || isempty(s.condense)
    s.condense = true;
end
if ~isfield(s, 'maxIter') || isempty(s.maxIter)
    s.maxIter = 200;
end
if ~isfield(s, 'tol') || isempty(s.tol)
    s.tol = 1.0e-10;
end
end
