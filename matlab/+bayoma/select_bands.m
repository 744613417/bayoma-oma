function sel = select_bands(freqs, psd, opts)
%SELECT_BANDS 按奇异值谱自动划分识别频带并给出各频带模态阶数。
%
%   SEL = bayoma.select_bands(FREQS, PSD) 在平均谱矩阵 PSD（nch x nch x Nf）
%   上自动选带，返回结构体字段：
%       bands     nb x 2 的频带上下界矩阵（Hz）
%       nmodes    nb x 1 的模态阶数
%       peaks     nb x 1 的谱峰频率（Hz）
%       sv        nch x Nf 的奇异值谱
%       floor     噪声基线
%
%   SEL = bayoma.select_bands(..., OPTS) 接受结构体 OPTS，字段：
%       kappa       频带半宽相对半功率带宽的倍数，缺省 10
%       min_sep_hz  相邻谱峰的最小间隔（Hz），缺省取 3 倍频率分辨率与 2%
%                   频率跨度中的较大者
%       prominence  谱峰相对突出度阈值，缺省 1e-3
%       max_modes   单个频带允许的最大模态阶数，缺省 4
%       noise_factor 判定模态数时相对噪声基线的倍数，缺省 3
%       peak_factor 谱峰须高于噪声基线的倍数，缺省 5
%       rel_peak    谱峰相对最高谱峰的比例下限，缺省 5e-2
%       smooth_hz   谱峰搜索前的滑动平均窗宽（Hz）
%
%   步骤：由奇异值谱估计噪声基线；对最高阶曲线做滑动平均后提取谱峰；每个
%   谱峰对应一个频带，半宽取 kappa 倍半功率带宽（阻尼比未知时按 1% 估计），
%   相邻频带重叠时按中点切开；在谱峰处计数显著曲线得模态阶数。

if nargin < 3
    opts = struct();
end
opts = fill_defaults(opts, struct('kappa', 10, 'min_sep_hz', [], ...
    'prominence', 1e-3, 'max_modes', 4, 'noise_factor', 3, ...
    'peak_factor', 5, 'rel_peak', 5e-2, 'smooth_hz', []));

fr = freqs(:);
nf = numel(fr);
df = fr(2) - fr(1);
span = fr(end) - fr(1);
sv = bayoma.sv_curves(psd);
floorVal = median(sv(end, :));

width = opts.smooth_hz;
if isempty(width)
    width = max(4 * df, 0.002 * span);
end
nwin = max(round(width / df), 1);
main = moving_average(sv(1, :), nwin);

sep = opts.min_sep_hz;
if isempty(sep)
    sep = max(3 * df, 0.002 * span);
end

above = max(main - floorVal, 0);
[pk, props] = findpeaks_local(above, opts.prominence * max(max(above), 1e-300), ...
    max(round(sep / df), 1));

% 只保留显著高于噪声基线的谱峰：宽带噪声起伏不得判为模态。仅用噪声基线
% 作判据不够，噪声起伏可高出基线数倍却远低于真实共振峰，故再加一条相对
% 最高谱峰的比例下限。
strong = pk(main(pk) > opts.peak_factor * floorVal);
if ~isempty(strong)
    pk = strong;
end
if ~isempty(pk)
    pk = pk(main(pk) >= opts.rel_peak * max(main(pk)));
end
if isempty(pk)
    [~, idx] = max(main);
    pk = idx;
end

halfWidth = opts.kappa * 0.01 * fr(pk);
lo = fr(pk) - halfWidth;
hi = fr(pk) + halfWidth;

% 相邻频带重叠时按中点切开，保证各频带互不重叠。
for i = 2:numel(pk)
    mid = (fr(pk(i-1)) + fr(pk(i))) / 2;
    if lo(i) < mid
        lo(i) = mid;
    end
    if hi(i-1) > mid
        hi(i-1) = mid;
    end
end
lo = max(lo, fr(1));
hi = min(hi, fr(end));

nmodes = zeros(numel(pk), 1);
for i = 1:numel(pk)
    vals = sv(:, pk(i));
    top = vals(1);
    thr = max([floorVal * opts.noise_factor, 1e-2 * top, 1e-300]);
    nmodes(i) = min(sum(vals > thr), opts.max_modes);
end

sel.bands = [lo(:), hi(:)];
sel.nmodes = nmodes;
sel.peaks = fr(pk);
sel.sv = sv;
sel.floor = floorVal;
end

function out = moving_average(x, n)
%MOVING_AVERAGE 中心滑动平均，两端按可用点数归一。
x = x(:).';
n = max(round(n), 1);
if n <= 1
    out = x;
    return;
end
k = ones(1, n);
num = conv(x, k, 'same');
den = conv(ones(1, numel(x)), k, 'same');
out = num ./ max(den, 1);
end

function [pk, props] = findpeaks_local(y, minProm, minDist)
%FINDPEAKS_LOCAL 局部极大值提取，按突出度与最小间隔筛选。
y = y(:).';
n = numel(y);
cand = [];
for i = 2:n-1
    if y(i) > y(i-1) && y(i) >= y(i+1)
        cand(end+1) = i; %#ok<AGROW>
    end
end
if isempty(cand)
    pk = [];
    props = struct();
    return;
end
% 突出度：与两侧最近谷底的较小高差。
prom = zeros(size(cand));
for j = 1:numel(cand)
    i = cand(j);
    left = y(1:i);
    right = y(i:end);
    lmin = min(left(max(1, i - 200):i));
    rmin = min(right(1:min(numel(right), 201)));
    prom(j) = y(i) - max(lmin, rmin);
end
keep = prom >= minProm;
cand = cand(keep);
prom = prom(keep);

% 按突出度从大到小贪心选取，保证间隔不小于 minDist。
[~, order] = sort(prom, 'descend');
chosen = [];
for j = order
    if isempty(chosen) || all(abs(cand(j) - chosen) >= minDist)
        chosen(end+1) = cand(j); %#ok<AGROW>
    end
end
pk = sort(chosen);
props = struct('prominence', prom);
end

function s = fill_defaults(s, d)
%FILL_DEFAULTS 用缺省值补齐结构体字段。
names = fieldnames(d);
for i = 1:numel(names)
    if ~isfield(s, names{i}) || isempty(s.(names{i}))
        s.(names{i}) = d.(names{i});
    end
end
end
