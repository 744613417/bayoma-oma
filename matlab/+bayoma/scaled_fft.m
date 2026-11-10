function S = scaled_fft(tdata, fs, band, p, detrend)
%SCALED_FFT 按式 (10.1) 计算缩放 FFT 并截取指定频带。
%
%   S = bayoma.scaled_fft(TDATA, FS, BAND) 返回频带 BAND = [f1 f2] 内的缩放
%   FFT。TDATA 为 nt x nch 的环境振动数据，FS 为采样频率（Hz）。
%
%   S = bayoma.scaled_fft(TDATA, FS, BAND, P) 对各通道乘以 (i*2*pi*f)^P(k)，
%   P 为标量或长度 nch 的向量。P = 0 表示原始数据为加速度，1 为速度，2 为位移。
%
%   S = bayoma.scaled_fft(TDATA, FS, BAND, P, DETREND) 指定是否去均值与线性
%   趋势，缺省为 true。
%
%   返回结构体字段：
%       freqs  频带内频率坐标（列向量，Hz）
%       fft    缩放 FFT，nch x Nf，第 k 列为 F_k
%       dfreq  频率分辨率 1/(N*dt)
%       nch    通道数
%       nf     频带内 FFT 点数 N_f
%       band   频带上界与下界
%
%   缩放方式使 E[F_k F_k'] 直接等于单边功率谱密度矩阵，单位与 Sii、Se 一致。

if nargin < 4 || isempty(p)
    p = 0;
end
if nargin < 5 || isempty(detrend)
    detrend = true;
end

if ~isnumeric(tdata) || ndims(tdata) ~= 2
    error('bayoma:scaled_fft:badData', 'tdata 必须是二维数值矩阵，行对应时刻、列对应通道');
end
if ~isscalar(fs) || ~isfinite(fs) || fs <= 0
    error('bayoma:scaled_fft:badFs', 'fs 必须是正有限标量');
end

nt = size(tdata, 1);
nch = size(tdata, 2);
if nt < 16
    error('bayoma:scaled_fft:tooShort', '样本数过少，无法进行频域识别');
end

data = double(tdata);
data = data - repmat(mean(data, 1), nt, 1);
if detrend
    t = (0:nt-1).';
    t = t - mean(t);
    slope = (t.' * data) / (t.' * t);
    data = data - t * slope;
end

dt = 1 / fs;
spec = fft(data, [], 1);
spec = spec * sqrt(dt / nt);
freqsAll = (0:nt-1).' / (nt * dt);

pvec = double(p(:));
if numel(pvec) == 1
    pvec = repmat(pvec, nch, 1);
end
if numel(pvec) ~= nch
    error('bayoma:scaled_fft:badP', 'p 的长度 %d 与通道数 %d 不一致', numel(pvec), nch);
end
if any(pvec ~= 0)
    omega = 2 * pi * freqsAll;
    factor = complex(zeros(nt, nch));
    for c = 1:nch
        factor(:, c) = (1i * omega).^pvec(c);
    end
    factor(1, :) = 0;
    spec = spec .* factor;
end

lo = band(1);
hi = band(2);
sel = freqsAll >= lo & freqsAll <= hi & freqsAll > 0;
if ~any(sel)
    error('bayoma:scaled_fft:emptyBand', '频带 [%g, %g] Hz 内没有 FFT 点', lo, hi);
end

S.freqs = freqsAll(sel);
S.fft = spec(sel, :).';
S.dfreq = 1 / (nt * dt);
S.nch = nch;
S.nf = numel(S.freqs);
S.band = [lo hi];
end
