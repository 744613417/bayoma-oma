function test_single_mode()
%TEST_SINGLE_MODE 单模态识别的合成数据校核。
%
%   用与 Python 实现相同的合成参数生成数据，核对频率、阻尼比与振型。
%   数据生成采用频域法：先按式 (10.10) 构造谱密度矩阵，再逐频率抽取圆对称
%   复高斯样本，最后逆变换回时域。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fs = 100;
dur = 2400;
nt = round(dur * fs);
f0 = 1.0;
z0 = 0.01;
phi = [1; 2; 2] / 3;
S = 1.0;
Se = 1.0;
q = 0;

rng(5);
tdata = synth_data(f0, z0, phi, S, Se, fs, nt, q);

Sspec = bayoma.scaled_fft(tdata, fs, [0.9 1.1]);
[D, dTotal] = bayoma.psd_matrices(Sspec);

opts = struct('f0', f0, 'q', q, 'maxiter', 200);
out = bayoma.single_mode(D, dTotal, Sspec.freqs, opts);

fprintf('f   = %.6f  (真值 %.4f)\n', out.f, f0);
fprintf('z   = %.6f  (真值 %.4f)\n', out.z, z0);
fprintf('S   = %.6f  (真值 %.4f)\n', out.S, S);
fprintf('Se  = %.6f  (真值 %.4f)\n', out.Se, Se);
fprintf('phi = [%.5f %.5f %.5f]  (真值 [%.5f %.5f %.5f])\n', ...
    out.phi(1), out.phi(2), out.phi(3), phi(1), phi(2), phi(3));
fprintf('sn  = %.1f   niter = %d   converged = %d\n', out.sn, out.niter, out.converged);
end

function tdata = synth_data(f, z, phi, S, Se, fs, nt, q)
%SYNTH_DATA 按式 (10.10) 的谱密度矩阵合成环境振动数据。
%
%   缩放 FFT 的协方差满足 E[F_k F_k'] = E_k，而 F_k = sqrt(dt/N) * rfft(y)，
%   故 rfft 的协方差为 E_k * N/dt。先按此协方差抽取圆对称复高斯样本，再由
%   共轭对称补全负频率半轴，逆变换后乘 N 还原时程。
nch = numel(phi);
nf = floor(nt / 2) + 1;
freqs = (0:nf-1).' * fs / nt;
dt = 1 / fs;

beta = f ./ freqs;
h = (2 * pi * freqs).^(-q) ./ (1 - beta.^2 - 2i * z * beta);
h(1) = 0;

spec = zeros(nf, nch);
for k = 1:nf
    Ek = S * abs(h(k))^2 * (phi * phi') + Se * eye(nch);
    Ek = (Ek + Ek') / 2;
    L = chol(Ek * nt / dt, 'lower');
    w = (randn(nch, 1) + 1i * randn(nch, 1)) / sqrt(2);
    spec(k, :) = (L * w).';
end
% 由单边谱补全双边谱后逆变换
full = [spec; conj(spec(end-1:-1:2, :))];
tdata = real(ifft(full, [], 1));
end
