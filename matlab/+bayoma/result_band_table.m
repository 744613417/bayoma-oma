function items = result_band_table(bands, f0List, nModsPerBand, peaks, excluded)
%RESULT_BAND_TABLE 频带表：每段的上下界、模态阶数、代入计算的频率初值。
%
%   ITEMS = bayoma.result_band_table(BANDS, F0LIST, NMODES) 返回元胞数组，每项
%   一个频带的记录，字段为 band_index、lower、upper、bandwidth、n_modes、f0。
%   BANDS 为 nb x 2 矩阵（每行一段的下界与上界，单位 Hz）；F0LIST 为元胞，每项
%   是该频带内各阶固有频率的初值；NMODES 为各频带的模态阶数。
%
%   ITEMS = bayoma.result_band_table(..., PEAKS) 另给出各频带的谱峰频率。
%   ITEMS = bayoma.result_band_table(..., PEAKS, EXCLUDED) 另标注该频带内被剔除的
%   子区间是否存在；EXCLUDED 为逻辑向量。
%
%   频带序号自 0 起计，与结果文件的其余部分一致。

if nargin < 4
    peaks = [];
end
if nargin < 5
    excluded = [];
end

nb = size(bands, 1);
items = cell(1, nb);
for i = 1:nb
    lo = double(bands(i, 1));
    hi = double(bands(i, 2));
    rec = struct('band_index', i - 1, 'lower', lo, 'upper', hi, ...
        'bandwidth', hi - lo, 'n_modes', 0, 'f0', []);
    if i <= numel(nModsPerBand)
        rec.n_modes = double(nModsPerBand(i));
    end
    if i <= numel(f0List) && ~isempty(f0List{i})
        rec.f0 = double(f0List{i}(:).');
    end
    if ~isempty(peaks) && i <= numel(peaks)
        % 谱峰频率缺省为 NaN，此时该字段不写入，保持契约里「可选」的语义
        if isfinite(peaks(i))
            rec.peak = double(peaks(i));
        end
    end
    if ~isempty(excluded) && i <= numel(excluded)
        rec.excluded = logical(excluded(i));
    end
    items{i} = rec;
end
end
