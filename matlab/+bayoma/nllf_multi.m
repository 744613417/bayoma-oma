function L = nllf_multi(f, z, S, Se, alpha, basis, fft, freqs, dTotal, q)
%NLLF_MULTI 多模态降维负对数似然函数（式 13.15 / 13.47）。
%
%   L = bayoma.nllf_multi(F, Z, S, SE, ALPHA, BASIS, FFT, FREQS, DTOTAL, Q)
%   返回降维负对数似然函数值。
%
%   参数
%       F       各阶频率（Hz），长度 m
%       Z       各阶阻尼比，长度 m
%       S       模态力谱密度矩阵，m x m，Hermitian
%       SE      通道噪声谱，标量
%       ALPHA   模态形状在子空间基下的坐标，m0 x m
%       BASIS   子空间基 B0，nch x m0，列正交
%       FFT     缩放 FFT，nch x Nf
%       FREQS   频率坐标（Hz）
%       DTOTAL  sum_k F_k' F_k
%       Q       模态频响函数的指数（0 加速度 / 1 速度 / 2 位移）
%
%   式 (13.47) 的最后一项是复值二次型 sum_k F'^*_k E0_k^{-1} F'_k，其中
%   F'_k = B0' F_k。该项必须保留 F_k 的相位：若改用实值
%   D'_k = B0' Re(F_k F_k') B0，则互谱虚部信息全部丢失，模态力谱密度矩阵 S 的
%   非对角元失去约束，密集模态下尤其如此。

if nargin < 10 || isempty(q)
    q = 0;
end

[nch, nf] = size(fft);
m0 = size(basis, 2);
m = numel(f);

if ~isfinite(Se) || Se <= 0
    L = inf;
    return;
end
if any(f <= 0) || any(z <= 0) || any(z >= 1)
    L = inf;
    return;
end

h = bayoma.transfer_matrix(f, z, freqs, q);
if any(~isfinite(h(:)))
    L = inf;
    return;
end

fproj = basis.' * fft;            % m0 x Nf，复值投影
d0 = sum(abs(fproj(:)).^2);
eyeM0 = eye(m0);

logdetSum = 0;
quadSum = 0;
eyeM0 = eye(m0);
for k = 1:nf
    ah = alpha .* repmat(h(:, k).', m0, 1);   % m0 x m
    e0 = ah * S * ah' + Se * eyeM0;
    e0 = (e0 + e0') / 2;
    % 用特征分解同时得到 log|E0_k| 与二次型，避免 chol 在病态情形下失败
    [Ve, De] = eig(e0);
    dv = real(diag(De));
    if any(dv <= 0) || any(~isfinite(dv))
        L = inf;
        return;
    end
    logdetSum = logdetSum + sum(log(dv));
    fk = fproj(:, k);
    quadSum = quadSum + real(fk' * (Ve * (diag(1 ./ dv) * (Ve' * fk))));
end

L = nch * nf * log(pi) + (nch - m0) * nf * log(Se) ...
    + (dTotal - d0) / Se + logdetSum + quadSum;
if ~isfinite(L)
    L = inf;
end
end
