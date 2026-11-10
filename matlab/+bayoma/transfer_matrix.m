function h = transfer_matrix(f, z, freqs, q)
%TRANSFER_MATRIX 各阶模态在频带内的传递函数矩阵 h（式 10.8）。
%
%   H = bayoma.transfer_matrix(F, Z, FREQS, Q) 返回 m x Nf 的复矩阵，
%       h_ik = (2*pi*i*f_k)^(-q) / (1 - beta_ik^2 - 2*i*z_i*beta_ik)
%   其中 beta_ik = f_i/f_k，Q = 0 加速度 / 1 速度 / 2 位移。

if nargin < 4 || isempty(q)
    q = 0;
end

f = f(:);
z = z(:);
fr = freqs(:).';
beta = f ./ fr;
om = 2 * pi * fr;
h = (om .^ (-q)) ./ (1 - beta.^2 - 2i * z .* beta);
h(~isfinite(h)) = 0;
end
