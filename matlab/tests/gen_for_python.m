function gen_for_python()
%GEN_FOR_PYTHON 生成一份合成数据并保存，供 Python 估计器交叉校核。
%
%   目的是把"数据合成"与"估计器实现"分开：同一份 D 矩阵分别送入 MATLAB 与
%   Python 的单模态估计器，若两者结果一致，则差异来自数据本身而非实现。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fs = 100;
dur = 2400;
nt = round(dur * fs);
f = 1.0;
z = 0.01;
phi = [1; 2; 2] / 3;
S = 1.0;
Se = 1.0;

rng(5);
tdata = synth_data(f, z, phi, S, Se, fs, nt, 0);

Sspec = bayoma.scaled_fft(tdata, fs, [0.9 1.1]);
[D, dTotal] = bayoma.psd_matrices(Sspec);

opts = struct('f0', f, 'q', 0, 'maxiter', 200);
out = bayoma.single_mode(D, dTotal, Sspec.freqs, opts);

fprintf('MATLAB: f=%.6f z=%.6f S=%.6f Se=%.6f niter=%d\n', ...
    out.f, out.z, out.S, out.Se, out.niter);
fprintf('MATLAB: phi=[%.6f %.6f %.6f]\n', out.phi(1), out.phi(2), out.phi(3));

freqs = Sspec.freqs;
save(fullfile(fileparts(mfilename('fullpath')), 'synth_case.mat'), ...
    'D', 'dTotal', 'freqs', 'f', 'z', 'phi', 'S', 'Se', 'fs', 'nt', '-v7');
end

function tdata = synth_data(f, z, phi, S, Se, fs, nt, q)
%SYNTH_DATA 按式 (10.10) 的谱密度矩阵合成环境振动数据。
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
full = [spec; conj(spec(end-1:-1:2, :))];
tdata = real(ifft(full, [], 1));
end
