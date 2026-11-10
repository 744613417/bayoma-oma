function [u, lam] = project_condensed(A, Lc)
%PROJECT_CONDENSED 在凝聚子空间内求最小特征值向量（课本式 14.18--14.20）
%   [u, lam] = bayoma.project_condensed(A, Lc)
%
%   A  对称矩阵（式 14.40 的 B，或式 14.8 的 A）
%   Lc 凝聚矩阵（见 bayoma.condensation_matrix）
%   u  单位欧氏范数的全局振型，lam 对应特征值

A = (A + A') / 2;
if isempty(Lc) || size(Lc, 2) == 0
    [V, Dg] = eig(A);
    dg = real(diag(Dg));
    [lam, idx] = min(dg);
    u = V(:, idx);
else
    Ar = Lc' * A * Lc;
    Ar = (Ar + Ar') / 2;
    [V, Dg] = eig(Ar);
    dg = real(diag(Dg));
    [lam, idx] = min(dg);
    u = Lc * V(:, idx);
end
nrm = norm(u);
if nrm > 0
    u = u / nrm;
end
end
