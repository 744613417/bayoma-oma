function Z = tangent_basis(a, forceComplex)
%TANGENT_BASIS 单位向量 A 所在单位球面的切空间正交基（Householder 构造）。
%
%   Z = bayoma.tangent_basis(A) 返回矩阵 Z，使 (a + Z*theta) 归一化后覆盖 a
%   邻域的单位球面。实向量情形 Z 为 n x (n-1)，切空间维数 n-1。
%
%   复向量情形把 a 按实部、虚部展开为 R^(2n) 的单位向量，Householder 给出
%   2n-1 个正交方向，再剔除整体相位方向 [-Im a; Re a]：该方向对应
%   a -> a*exp(1i*phi)，似然不变（相位差可由模态力谱矩阵吸收），属规范自由度；
%   若不剔除，后验海森矩阵在该方向的曲率恰为零，方差发散。剔除后返回
%   Z（n x (2n-2)）。
%
%   Z = bayoma.tangent_basis(A, FORCECOMPLEX) 中 FORCECOMPLEX 为真时，即便 a
%   当前恰为实向量也按复向量构造 2n-2 维切空间。识别过程中振型可能从实初值
%   出发，若按 isreal 判定，第一步就退化为实形状类，复形状不可达。

a = a(:);
n = numel(a);
if nargin < 2 || isempty(forceComplex)
    forceComplex = ~isreal(a);
elseif forceComplex && isreal(a)
    % 显式要求复形状：保持复切空间，不做实数化简
    forceComplex = true;
elseif ~forceComplex && ~isreal(a)
    % 未要求复形状却收到复向量：按实向量处理会丢虚部自由度，故强制走复分支
    forceComplex = true;
end

if forceComplex
    m = 2 * n;
    u = [real(a); imag(a)];
else
    m = n;
    u = a;
end
nu = norm(u);
if nu <= 1e-14 || m <= 1
    Z = zeros(n, 0);
    return;
end
u = u / nu;

% Householder 把 u 映到 e1，其前 m-1 列与 u 正交
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
Zr = Hm(:, 2:m);

if forceComplex
    ph = [-imag(a); real(a)];
    nph = norm(ph);
    if nph > 1e-12
        ph = ph / nph;
        Zr = Zr - ph * (ph.' * Zr);
        [Q, ~] = qr(Zr, 0);
        Zr = Q(:, 1:min(size(Zr, 2), m - 2));
    end
    Z = Zr(1:n, :) + 1i * Zr(n + 1:2 * n, :);
else
    Z = Zr;
end
end
