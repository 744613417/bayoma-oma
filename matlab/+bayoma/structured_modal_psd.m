function S = structured_modal_psd(params, m, scales)
%STRUCTURED_MODAL_PSD 由自由参数重建模态力谱密度矩阵（式 13.50–13.57）。
%
%   S = bayoma.structured_modal_psd(PARAMS, M, SCALES) 返回 M x M 的 Hermitian
%   半正定矩阵 S = diag(s) S0 diag(s)，其中 S0_ij = exp(i*v_ij) sin(u_ij)
%   （i > j），对角元为 1。
%
%   PARAMS 长度 M*(M-1)：前一半为 u_ij，后一半为 v_ij。SCALES 为长度 M 的
%   各阶模态力谱密度尺度 sqrt(S_ii)。

half = m * (m - 1) / 2;
u = params(1:half);
v = params(half+1:2*half);
corr = eye(m);
idx = 1;
for i = 1:m
    for j = 1:i-1
        amp = sin(u(idx));
        ph = exp(1i * v(idx));
        corr(i, j) = amp * ph;
        corr(j, i) = amp * conj(ph);
        idx = idx + 1;
    end
end
scales = scales(:);
S = (scales * scales.') .* corr;
end
