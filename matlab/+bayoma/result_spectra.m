function out = result_spectra(data, fs, bands, f0List, maxPoints, nblock)
%RESULT_SPECTRA 时程数据的功率谱与奇异值谱，并标注频带与代入计算的频率初值。
%
%   OUT = bayoma.result_spectra(DATA, FS, BANDS, F0LIST) 对 DATA（行对应时刻、
%   列对应通道）作分块平均的单边谱估计，给出各通道自谱与奇异值谱。默认抽样到
%   不超过 1024 个频点写入结果。
%
%   OUT = bayoma.result_spectra(..., MAXPOINTS, NBLOCK) 指定频点上限与每块样本数。
%
%   返回结构体字段：freqs（频率轴）、psd_diag（各通道自谱）、sv（奇异值谱）、
%   stride（抽稀倍数）、n_freq_full（原始频点数）、n_freq_kept（写入频点数）、
%   reduction（抽稀方式）、bands（各频带的标注）、note。
%
%   抽稀采用块内平均而非简单抽样：窄带谱峰在简单抽样下可能整点丢失，块内平均
%   保持谱峰幅值的同时把频点压到目标数量。抽稀倍数与原始频点数一并记录，便于
%   复算频率轴。

if nargin < 5 || isempty(maxPoints)
    maxPoints = 1024;
end
if nargin < 6
    nblock = [];
end

data = double(data);
[freqs, psd] = bayoma.welch_psd(data, fs, nblock);

% 各通道自谱：psd 的第三个下标为频点，取每个频点上谱矩阵的对角元
nch = size(psd, 1);
nf = size(psd, 3);
diagPsd = zeros(nch, nf);
for k = 1:nf
    diagPsd(:, k) = real(diag(psd(:, :, k)));
end
sv = bayoma.sv_curves(psd);

stride = max(1, ceil(nf / maxPoints));
if stride == 1
    freqsRed = freqs(:).';
    diagRed = diagPsd;
    svRed = sv;
else
    keep = floor(nf / stride) * stride;
    diagRed = block_mean(diagPsd(:, 1:keep), stride);
    svRed = block_mean(sv(:, 1:keep), stride);
    freqsRed = mean(reshape(freqs(1:keep), stride, []), 1);
end

marks = cell(1, size(bands, 1));
for i = 1:size(bands, 1)
    lo = double(bands(i, 1));
    hi = double(bands(i, 2));
    inBand = (freqs >= lo) & (freqs <= hi);
    rec = struct('band_index', i - 1, 'lower', lo, 'upper', hi);
    if i <= numel(f0List) && ~isempty(f0List{i})
        rec.f0 = double(f0List{i}(:).');
    else
        rec.f0 = [];
    end
    rec.n_freq_points = sum(inBand);
    if any(inBand)
        % 峰值取频带内第一通道自谱的最大值所在频点
        seg = diagPsd(1, inBand);
        idx = find(inBand);
        [~, loc] = max(seg);
        rec.peak_freq = freqs(idx(loc));
    else
        rec.peak_freq = bayoma.json_null();
    end
    marks{i} = rec;
end

out = struct();
out.freqs = freqsRed;
out.psd_diag = diagRed;
out.sv = svRed;
out.stride = stride;
out.n_freq_full = nf;
out.n_freq_kept = numel(freqsRed);
out.reduction = '块内平均';
out.bands = marks;
out.note = ['psd_diag 为各通道自谱（Welch 平均），sv 为奇异值谱（按降序）；' ...
    '行列顺序与 channel_index、模态阶次一一对应。'];
end

% ----------------------------------------------------------------------
function red = block_mean(mat, stride)
%BLOCK_MEAN 沿第二维按 stride 个点一组取平均。
n = size(mat, 2);
keep = floor(n / stride) * stride;
red = squeeze(mean(reshape(mat(:, 1:keep), size(mat, 1), stride, []), 2));
if size(mat, 1) == 1
    red = reshape(red, 1, []);
end
end
