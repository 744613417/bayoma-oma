function test_synth_scale()
%TEST_SYNTH_SCALE 校核合成数据的幅度标定。
%
%   生成已知 S、Se 的数据后，直接比较缩放 FFT 的样本谱矩阵与理论谱矩阵，
%   确认合成与识别两侧的标定一致。远离共振处 D_k 应约等于 Se*I。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fs = 100;
nt = 60000;
phi = [1; 2; 2] / 3;
f = 1.0;
z = 0.01;
S = 1.0;
Se = 1.0;

rng(5);
tdata = synth_data(f, z, phi, S, Se, fs, nt, 0);

Sspec = bayoma.scaled_fft(tdata, fs, [0.3 0.5]);
[D, ~] = bayoma.psd_matrices(Sspec);
fprintf('远离共振（0.3-0.5 Hz）D 的对角均值 = %.6g   (理论 Se = %.6g)\n', ...
    mean(mean(real(diag(D(:, :, 1))))), Se);

Sspec = bayoma.scaled_fft(tdata, fs, [0.95 1.05]);
[D, ~] = bayoma.psd_matrices(Sspec);
[~, imid] = min(abs(Sspec.freqs - f));
fprintf('共振处（%.3f Hz）D 的最大特征值 = %.6g\n', Sspec.freqs(imid), max(eig(D(:, :, imid))));
fprintf('  理论值 = %.6g\n', S / (4 * z^2) * max(eig(phi * phi')) + Se);
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
