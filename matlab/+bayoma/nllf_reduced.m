function [L, A] = nllf_reduced(f, z, S, Se, freqs, D, dTotal, q)
%NLLF_REDUCED 单模态降维负对数似然函数（式 12.13）与矩阵 A（式 12.11）。
%
%   [L, A] = bayoma.nllf_reduced(F, Z, S, SE, FREQS, D, DTOTAL, Q) 返回降维
%   负对数似然函数 L~ 与加权谱矩阵 A = sum_k (1 + SE/(S*D_k))^{-1} D_k。
%
%   D 为 nch x nch x Nf 的样本谱矩阵 D_k = Re(F_k F_k')，DTOTAL = sum_k F_k'F_k。
%   模态形状已被解析消去：式 (12.13) 中的 u'Au 取 A 的最大特征值。
%
%   参数说明见 bayoma.bayoma 的输入字段 p 与 q。

if nargin < 8 || isempty(q)
    q = 0;
end

nch = size(D, 1);
nf = size(D, 3);

Dk = bayoma.dynamic_factor(f, z, freqs, q);
p = S * Dk + Se;
if any(p <= 0) || ~isfinite(S) || ~isfinite(Se) || S <= 0 || Se <= 0
    L = inf;
    A = zeros(nch, nch);
    return;
end

weight = 1 ./ (1 + Se ./ (S * Dk));
A = zeros(nch, nch);
for k = 1:nf
    A = A + weight(k) * D(: , :, k);
end
lambda = max(eig((A + A.') / 2));

L = nch * nf * log(pi) + sum(log(p)) + (nch - 1) * nf * log(Se) ...
    + (dTotal - lambda) / Se;
end
