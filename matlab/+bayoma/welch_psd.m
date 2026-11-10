function [freqs, psd] = welch_psd(tdata, fs, nblock, overlap)
%WELCH_PSD 分块平均的单边功率谱密度矩阵，用于奇异值谱与自动选带。
%
%   [FREQS, PSD] = bayoma.welch_psd(TDATA, FS) 返回分块平均的单边谱密度
%   矩阵 PSD，尺寸 nch x nch x Nf。TDATA 为 nt x nch 的环境振动数据。
%
%   [FREQS, PSD] = bayoma.welch_psd(TDATA, FS, NBLOCK, OVERLAP) 指定每块样本数
%   与重叠率（0 至 1），缺省按总长度的 1/8 取 2 的整数幂、重叠率 0.5。
%
%   判定模态数与自动选带须使用平均谱矩阵：单次实现的 Re(F_k F_k') 秩为 1，
%   高阶奇异值恒为零，噪声基线会被估成零，宽带噪声起伏会被误判为谱峰。

if nargin < 3 || isempty(nblock)
    nblock = min(size(tdata, 1), max(256, 2 ^ floor(log2(size(tdata, 1) / 8))));
end
if nargin < 4 || isempty(overlap)
    overlap = 0.5;
end

data = double(tdata);
nt = size(data, 1);
nch = size(data, 2);
data = data - repmat(mean(data, 1), nt, 1);

nblock = min(nblock, nt);
step = max(1, round(nblock * (1 - overlap)));
starts = 1:step:(nt - nblock + 1);
if isempty(starts)
    starts = 1;
end

win = hann(nblock);
winPower = sum(win .^ 2);
nf = floor(nblock / 2) + 1;
acc = zeros(nch, nch, nf);
for s = starts
    seg = data(s:s + nblock - 1, :) .* repmat(win, 1, nch);
    spec = fft(seg, [], 1);
    spec = spec(1:nf, :).';
    for k = 1:nf
        fk = spec(:, k);
        acc(:, :, k) = acc(:, :, k) + (fk * fk');
    end
end
acc = acc * (2 / (fs * winPower)) / numel(starts);
acc(:, :, 1) = acc(:, :, 1) / 2;
if mod(nblock, 2) == 0
    acc(:, :, end) = acc(:, :, end) / 2;
end

freqs = (0:nf-1).' * fs / nblock;
psd = real(acc);
end
