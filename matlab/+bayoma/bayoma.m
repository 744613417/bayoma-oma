function out = bayoma(IN)
%BAYOMA 自动贝叶斯模态识别总入口，输入输出字段与参考 pcode 的契约一致。
%
%   OUT = bayoma(IN) 执行自动贝叶斯模态识别。IN 为结构体，字段：
%
%   必填字段
%       tdata   nt x nch 的环境振动数据，默认按加速度计（单位 g）
%       fs      采样频率（Hz）
%       f1f2    nb x 2 的频带上下界矩阵，每行一个频带（Hz）
%       f0      nb x 1 的 cell，每项为该频带内固有频率初值的一维数组
%
%   可选字段（基本）
%       mute    1 表示不输出中间信息，0 表示输出，缺省 0
%       ichan   参与计算的通道索引（1 基），缺省全部
%       iband   参与计算的频带索引（1 基），缺省全部
%
%   可选字段（高级）
%       maxiter MPV 迭代上限，缺省 500
%       f1f2d   需从似然函数排除的频带，nd x 2，缺省空
%       p       原始数据各通道乘以 (i*2*pi*f)^p 的指数，标量或长度 nch
%       q       模态频响函数除以 (i*2*pi*fk)^q 的指数，缺省 0
%
%   可选字段（结果输出）
%       geometry    通道几何，给出时在 recognition.spatial 中给出振型图
%       figure_dir  振型图保存目录，缺省不画
%       spectra     是否输出功率谱与奇异值谱，缺省 false
%
%   输出字段
%       f        各阶频率（Hz）
%       z        各阶阻尼比
%       Sii      各阶模态力自谱（单边）
%       Se       各阶通道噪声谱（单边）
%       phi      振型，每列一阶，欧氏范数为 1
%       rms      模态加速度均方根
%       S        各频带的模态力谱矩阵（cell）
%       coefv    各参数的后验变异系数
%       cmat     各频带的后验协方差矩阵（cell）
%       sn       模态信噪比
%       time_req 各频带求 MPV 与后验协方差的耗时（s），2 x nb
%       recognition 识别结果分组：频带对应、密集模态区分度、联合后验协方差、
%                 逐阶振型不确定性，以及可选的功率谱与空间振型图
%
%   模型与单位约定见 docs/IO_SPEC.md。识别频带未给出时按奇异值谱自动选带，
%   此时以平均谱矩阵判定模态数。
%
%   联合后验协方差按频带分块对角拼接：不同频带的频点互不重叠，似然可加而
%   海森矩阵分块对角；同一频带内各阶由联合似然同时确定，其相关性保留在块内。

if nargin < 1 || ~isstruct(IN)
    error('bayoma:badInput', '输入必须是结构体，字段见 bayoma 的帮助文本');
end

IN = fill_defaults(IN, struct( ...
    'mute', 0, 'ichan', [], 'iband', [], 'maxiter', 500, ...
    'f1f2d', [], 'p', 0, 'q', 0, 'f1f2', [], 'f0', {{}}, ...
    'geometry', [], 'figure_dir', [], 'spectra', false));

tdata = double(IN.tdata);
if ndims(tdata) ~= 2
    error('bayoma:badData', 'tdata 必须是二维矩阵，行对应时刻、列对应通道');
end
if ~isempty(IN.ichan)
    tdata = tdata(:, IN.ichan);
    channelIndex = double(IN.ichan(:).') - 1;
else
    channelIndex = 0:size(tdata, 2) - 1;
end
nch = size(tdata, 2);

if isempty(IN.f1f2)
    [wfreqs, wpsd] = bayoma.welch_psd(tdata, IN.fs);
    sel = bayoma.select_bands(wfreqs, wpsd);
    bands = sel.bands;
    f0cell = num2cell(sel.peaks(:));
    peaks = sel.peaks(:);
    if ~IN.mute
        fprintf('自动选带：%d 个频带\n', size(bands, 1));
    end
else
    bands = IN.f1f2;
    peaks = nan(size(bands, 1), 1);
    if isempty(IN.f0)
        f0cell = repmat({[]}, size(bands, 1), 1);
    else
        f0cell = IN.f0;
        if numel(f0cell) ~= size(bands, 1)
            error('bayoma:badF0', 'f0 的长度需与频带数一致');
        end
    end
end

if ~isempty(IN.iband)
    bands = bands(IN.iband, :);
    f0cell = f0cell(IN.iband);
    peaks = peaks(IN.iband);
end
% 初值有两种合法写法：{1.0}（一项一个数值向量）与 {{1.0}}（一层多余的嵌套）。
% 后者常见于 JSON 解出后手写描述文件的情形，统一剥掉外层元胞，避免后续按
% 数值向量使用时报「无法从 cell 转换为 double」。
for i = 1:numel(f0cell)
    f0cell{i} = unwrap_f0(f0cell{i});
end

nb = size(bands, 1);
% 各频带是否含被排除的子区间：排除段只从似然里去掉，频带本身仍在结果里
excludedFlags = false(1, nb);
for i = 1:nb
    if isempty(IN.f1f2d)
        break;
    end
    for k = 1:size(IN.f1f2d, 1)
        if IN.f1f2d(k, 2) > bands(i, 1) && IN.f1f2d(k, 1) < bands(i, 2)
            excludedFlags(i) = true;
        end
    end
end

fAll = zeros(0, 1); zAll = zeros(0, 1);
SiiAll = zeros(0, 1); SeAll = zeros(0, 1); phiAll = zeros(nch, 0);
Slist = cell(nb, 1); cmatList = cell(nb, 1);
cvF = zeros(0, 1); cvZ = zeros(0, 1); cvS = zeros(0, 1);
cvSe = zeros(0, 1); cvPhi = zeros(0, 1);
timeReq = zeros(2, nb);
nModsPerBand = zeros(1, nb);
% 每频带的参与量：协方差块、逐阶振型协方差、参数排布，供结果分组装配
blocks = cell(1, nb);

for i = 1:nb
    t0 = tic;
    band = bands(i, :);
    S = bayoma.scaled_fft(tdata, IN.fs, band, IN.p);
    [freqs, fft] = exclude_bands(S.freqs, S.fft, IN.f1f2d);
    [D, dTotal] = bayoma.psd_matrices(struct('fft', fft, 'nch', nch, ...
        'nf', numel(freqs), 'freqs', freqs));

    f0i = f0cell{i};
    if isempty(f0i)
        sv = bayoma.sv_curves(D);
        [~, idx] = max(sv(1, :));
        f0i = freqs(idx);
    end
    f0i = f0i(:).';

    opts = struct('f0', f0i, 'q', IN.q, 'maxiter', IN.maxiter);
    if numel(f0i) > 1
        res = bayoma.multi_mode(D, fft, freqs, numel(f0i), opts);
    else
        res = bayoma.single_mode(D, dTotal, freqs, opts);
    end

    nm = numel(res.f);
    offset = numel(fAll);
    fAll = [fAll; res.f(:)];                        %#ok<AGROW>
    zAll = [zAll; res.z(:)];                        %#ok<AGROW>
    SiiAll = [SiiAll; real(diag(res.S))];           %#ok<AGROW>
    SeAll = [SeAll; expand_to(res.Se(:), nm)];      %#ok<AGROW>
    phiAll = [phiAll, res.phi];                     %#ok<AGROW>
    Slist{i} = res.S;
    cmatList{i} = res.cmat;
    nModsPerBand(i) = nm;

    cvF = [cvF; expand_to(res.coefv.f(:), nm)];         %#ok<AGROW>
    cvZ = [cvZ; expand_to(res.coefv.z(:), nm)];         %#ok<AGROW>
    cvS = [cvS; expand_to(res.coefv.Sii(:), nm)];       %#ok<AGROW>
    cvSe = [cvSe; expand_to(res.coefv.Se(:), nm)];      %#ok<AGROW>
    cvPhi = [cvPhi; expand_to(res.coefv.phi(:), nm)];   %#ok<AGROW>
    timeReq(1, i) = toc(t0);

    blk = struct();
    % 联合协方差需要该频带全部参数（含各通道振型分量）的协方差；单模态与
    % 多模态都在识别结果里另行给出全矩阵 full，cmat 仍按原约定保存谱参数块
    if isfield(res, 'full') && ~isempty(res.full)
        blk.cov = res.full;
    else
        blk.cov = res.cmat;
    end
    blk.cov_phi = res.cov_phi;
    blk.m = nm;
    blk.n_off = res.n_off;
    blk.sizes = res.sizes;
    blk.band_index = i - 1;
    blk.offset = offset;
    blk.complex = res.complex;
    blk.nch = nch;
    blk.channel_index = channelIndex;
    % 单模态的参数排列与「谱参数 + 切空间坐标」的通用排布不同：振型张满整个
    % 通道空间，其分量直接作为参数给出，故由识别结果显式带上索引
    if isfield(res, 'index') && ~isempty(res.index)
        blk.index = res.index;
    end
    blocks{i} = blk;

    if ~IN.mute
        fprintf('频带 %.4f-%.4f Hz：%d 阶，f=%s，z=%s\n', band(1), band(2), ...
            nm, mat2str(res.f(:).', 5), mat2str(res.z(:).', 5));
    end
end

withZ = zAll > 0;
rms = nan(size(fAll));
rms(withZ) = sqrt(pi * SiiAll(withZ) .* fAll(withZ) ./ (4 * zAll(withZ)));
sn = nan(size(fAll));
sn(withZ) = SiiAll(withZ) ./ (4 * SeAll(withZ) .* zAll(withZ).^2);

out.f = fAll.';
out.z = zAll.';
out.Sii = SiiAll.';
out.Se = SeAll.';
out.phi = phiAll;
out.rms = rms.';
out.S = Slist;
out.coefv = struct('f', cvF.', 'z', cvZ.', 'Sii', cvS.', 'Se', cvSe.', ...
    'phi', cvPhi.', 'rms', 0.5 * sqrt(cvF.^2 + cvZ.^2 + cvS.^2).');
out.cmat = cmatList;
out.sn = sn.';
out.time_req = timeReq;
out.bands = bands;
out.nch = nch;
out.fs = IN.fs;

out.recognition = bayoma.result_recognition(out.f, out.z, out.phi, bands, ...
    f0cell, nModsPerBand, blocks, ...
    'channel_index', channelIndex, ...
    'data', tdata, ...
    'fs', IN.fs, ...
    'spectra', logical(IN.spectra), ...
    'geometry', IN.geometry, ...
    'figure_dir', IN.figure_dir, ...
    'peaks', peaks, ...
    'excluded', excludedFlags);
end

function mask = excluded_mask(freqs, f1f2d)
%EXCLUDED_MASK 标记需从似然函数排除的频率点。
mask = false(size(freqs));
if isempty(f1f2d)
    return;
end
for i = 1:size(f1f2d, 1)
    mask = mask | (freqs >= f1f2d(i, 1) & freqs <= f1f2d(i, 2));
end
end

function [freqs, fft] = exclude_bands(freqs, fft, f1f2d)
%EXCLUDE_BANDS 剔除指定频带内的频率点。
if isempty(f1f2d)
    return;
end
keep = ~excluded_mask(freqs, f1f2d);
if ~any(keep)
    error('bayoma:emptyAfterExclude', '排除频带后没有剩余频率点');
end
freqs = freqs(keep);
fft = fft(:, keep);
end

function s = fill_defaults(s, d)
%FILL_DEFAULTS 用缺省值补齐结构体字段。
names = fieldnames(d);
for i = 1:numel(names)
    if ~isfield(s, names{i})
        s.(names{i}) = d.(names{i});
    end
end
end

function v = expand_to(raw, nm)
%EXPAND_TO 把标量或短向量展开为长度 nm 的列向量。
%   多模态分支的噪声谱与部分变异系数是标量，单模态分支是向量，统一展开后
%   才能按频带拼接。
v = double(raw(:));
if numel(v) == 1
    v = repmat(v, nm, 1);
elseif numel(v) ~= nm
    error('bayoma:badShape', '结果长度与模态数不一致：%d 对 %d', numel(v), nm);
end
end

function v = unwrap_f0(raw)
%UNWRAP_F0 把嵌套的初值元胞剥到数值向量。
%   一层元胞里再套一层元胞时递归剥开，最终要求得到数值向量；其余情形按
%   数值向量原样返回，让上层报出更具体的形状错误。
if iscell(raw)
    if isempty(raw)
        v = [];
    else
        v = unwrap_f0(raw{1});
    end
    return;
end
if isnumeric(raw) || islogical(raw)
    v = double(raw(:).');
    return;
end
v = raw;
end
