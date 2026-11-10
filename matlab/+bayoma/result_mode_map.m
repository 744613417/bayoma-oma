function items = result_mode_map(f, bandOfOrder, offsetOfBand, bands, f0List)
%RESULT_MODE_MAP 每阶模态落在哪个频带、在频带内的次序、以及该频带的上下界与初值。
%
%   ITEMS = bayoma.result_mode_map(F, BANDOFORDER, OFFSETOFBAND, BANDS, F0LIST)
%   返回元胞数组，每项一阶模态，字段为 order、frequency、band_index、
%   order_in_band，并在频带表覆盖该阶时另给 band_lower、band_upper 与 f0。
%
%   F 为各阶识别频率；BANDOFORDER 为长度与 F 相同的频带序号；OFFSETOFBAND
%   为各频带首阶在全局的序号；BANDS 为 nb x 2 的频带矩阵；F0LIST 为元胞，
%   每项是该频带的频率初值。
%
%   识别频率、频带下界、上界与代入计算的初值放在同一条记录里，按阶次即可
%   查回所用的频带，不必再关联频带表。

if nargin < 4
    bands = [];
end
if nargin < 5
    f0List = {};
end

f = double(f(:).');
m = numel(f);
items = cell(1, m);
hasBands = ~isempty(bands);
if hasBands
    nb = size(bands, 1);
else
    nb = 0;
end
for order = 1:m
    bi = bandOfOrder(order);          % 0 起计
    rec = struct('order', order - 1, 'frequency', f(order), ...
        'band_index', bi, 'order_in_band', order - 1 - offsetOfBand(bi + 1));
    if hasBands && bi >= 0 && bi < nb
        rec.band_lower = double(bands(bi + 1, 1));
        rec.band_upper = double(bands(bi + 1, 2));
    end
    if ~isempty(f0List) && bi >= 0 && bi < numel(f0List) && ~isempty(f0List{bi + 1})
        rec.f0 = double(f0List{bi + 1}(:).');
    end
    items{order} = rec;
end
end
