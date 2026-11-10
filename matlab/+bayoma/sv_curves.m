function sv = sv_curves(D)
%SV_CURVES 奇异值谱：Re(D_k) 的特征值按降序排列（课本第 7.3 节）。
%
%   SV = bayoma.sv_curves(D) 返回 nch x Nf 的奇异值谱曲线，第 1 行为最高阶
%   曲线。共振带内显著高于其余曲线的条数等于该频带内主导模态的数目。
%
%   注意：判定模态数须使用**经平均的**谱矩阵（如 Welch 平均的谱密度矩阵）。
%   单次实现的 Re(F_k F_k') 秩为 1，其高阶奇异值恒为零，不能用于判定模态数。

nch = size(D, 1);
nf = size(D, 3);
sv = zeros(nch, nf);
for k = 1:nf
    mat = (D(:, :, k) + D(:, :, k).') / 2;
    vals = sort(eig(mat), 'descend');
    sv(:, k) = vals;
end
end
