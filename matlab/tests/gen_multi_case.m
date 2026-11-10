function gen_multi_case()
%GEN_MULTI_CASE 生成多模态合成数据并保存，供 Python 侧交叉校核。
%
%   保存缩放 FFT、频率坐标与真值参数，使两侧估计器在完全相同的输入上比较。

addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

fs = 100; dur = 2400; nt = round(dur * fs);
f = [1.0; 1.02]; z = [0.01; 0.01];
phi = [1 -2; 2 1; 2 2] ./ 3;
phi = phi ./ sqrt(sum(phi.^2, 1));
S = [1.0, 0.5*exp(1i*pi/4); 0.5*exp(-1i*pi/4), 1.0];
Se = 1.0;

rng(3);
tdata = synth_data(f, z, phi, S, Se, fs, nt);
Sspec = bayoma.scaled_fft(tdata, fs, [0.9 1.1]);
freqs = Sspec.freqs;
fft = Sspec.fft;
dTotal = sum(abs(fft(:)).^2);

save(fullfile(fileparts(mfilename('fullpath')), 'multi_case.mat'), ...
    'fft', 'freqs', 'dTotal', 'f', 'z', 'phi', 'S', 'Se', 'fs', 'nt', '-v7');
fprintf('saved: nch=%d nf=%d dTotal=%.6f\n', size(fft, 1), size(fft, 2), dTotal);
end

function tdata = synth_data(f, z, phi, S, Se, fs, nt)
nch = size(phi, 1); nf = floor(nt/2)+1;
freqs = (0:nf-1).'*fs/nt; dt = 1/fs;
beta = f./freqs.';
om = 2*pi*freqs.';
h = 1 ./ (1 - beta.^2 - 2i*z.*beta);
h(:, 1) = 0;
spec = zeros(nf, nch);
for k = 1:nf
    hk = h(:, k);
    Ek = (phi.*hk.')*S*(phi.*conj(hk).')' + Se*eye(nch);
    Ek = (Ek + Ek')/2;
    [Ve, De] = eig(Ek);
    De = max(real(diag(De)), 0);
    Lf = Ve*diag(sqrt(De*nt/dt));
    w = (randn(nch, 1) + 1i*randn(nch, 1))/sqrt(2);
    spec(k, :) = (Lf*w).';
end
full = [spec; conj(spec(end-1:-1:2, :))];
tdata = real(ifft(full, [], 1));
end
