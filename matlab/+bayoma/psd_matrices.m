function [D, dTotal] = psd_matrices(S)
%PSD_MATRICES 由缩放 FFT 计算样本谱矩阵 D_k = Re(F_k F_k') 与总量 d。
%
%   [D, DTOTAL] = bayoma.psd_matrices(S) 接受 bayoma.scaled_fft 返回的结构体，
%   返回 D 为 nch x nch x Nf 的实对称矩阵序列，DTOTAL = sum_k F_k' F_k
%   （式 12.12 / 14.33）。

if ~isfield(S, 'fft')
    error('bayoma:psd_matrices:badInput', '输入必须是 bayoma.scaled_fft 的返回值');
end

nch = S.nch;
nf = S.nf;
D = zeros(nch, nch, nf);
for k = 1:nf
    fk = S.fft(:, k);
    D(:, :, k) = real(fk * fk');
end
dTotal = sum(abs(S.fft(:)).^2);
end
