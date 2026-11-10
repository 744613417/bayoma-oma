function items = result_index(kind, varargin)
%RESULT_INDEX 协方差矩阵各参数的排列描述（与 Python 侧 results.py 同构）。
%
%   ITEMS = bayoma.result_index(KIND, ...) 返回一个元胞数组，每项是一个标量
%   结构体，描述协方差矩阵中某个对角元对应的参数。字段 name、kind 必给，
%   另按参数类别给出 mode、coord、symbol、unit、channel、group、group_pair
%   等定位信息。数组下标的起算方式与结果文件一致：模态阶次、通道号、测组号
%   均自 0 起计。
%
%   KIND 的取值与对应参数：
%       'spectral'      (M, NOFF)                    多阶谱参数的排列
%       'shape'         (M, SIZES)                   各阶振型切空间坐标
%       'single'        (NCH)                        单模态频带（含振型分量）
%       'async_single'  (NCH, NG)                    异步单阶（含相干矩阵）
%       'async_multi'   (M, NG, NOFF, COMPLEXS, ...) 异步多阶（含相干模与相位）
%       'layout'        (M, NG, NCH, COVSIZE, NPAIRS) 由协方差维数反推自由参数排布
%
%   谱参数的排列与后验协方差的打包顺序一致：
%       [f (M), z (M), diag(S) (M), Re(offdiag S) (NOFF), Im(offdiag S) (NOFF), Se (1)]
%   其中非对角元按上三角行优先顺序排列。

switch lower(kind)
    case 'spectral'
        items = spectral_index(varargin{:});
    case 'shape'
        items = shape_index(varargin{:});
    case 'single'
        items = single_mode_index(varargin{:});
    case 'async_single'
        items = async_single_index(varargin{:});
    case 'async_multi'
        items = async_multi_index(varargin{:});
    case 'layout'
        items = async_layout(varargin{:});
    otherwise
        error('bayoma:badIndexKind', '未知的参数描述类型：%s', kind);
end
end

% ----------------------------------------------------------------------
function items = entry(name, kind)
%ENTRY 建立一条只有名称与类别的参数记录。
items = struct('name', name, 'kind', kind);
end

% ----------------------------------------------------------------------
function p = triu_pairs(m)
%TRIU_PAIRS 上三角（不含对角）的行优先下标对，与 numpy.triu_indices 一致。
p = zeros(m * (m - 1) / 2, 2);
k = 0;
for i = 1:m - 1
    for j = i + 1:m
        k = k + 1;
        p(k, :) = [i, j];
    end
end
end

% ----------------------------------------------------------------------
function items = spectral_index(m, nOff)
%SPECTRAL_INDEX 多阶谱参数的位置描述。
items = cell(1, 0);
for i = 1:m
    s = entry(sprintf('f[%d]', i - 1), 'f');
    s.mode = i - 1; s.symbol = 'f'; s.unit = 'Hz';
    items{end + 1} = s;
end
for i = 1:m
    s = entry(sprintf('z[%d]', i - 1), 'z');
    s.mode = i - 1; s.symbol = 'zeta'; s.unit = '-';
    items{end + 1} = s;
end
for i = 1:m
    s = entry(sprintf('S[%d][%d]', i - 1, i - 1), 'S_diag');
    s.mode = i - 1; s.symbol = 'S'; s.unit = '-';
    items{end + 1} = s;
end
pr = triu_pairs(m);
for k = 1:nOff
    s = entry(sprintf('ReS[%d][%d]', pr(k, 1) - 1, pr(k, 2) - 1), 'S_off_re');
    s.modes = [pr(k, 1) - 1, pr(k, 2) - 1];
    s.symbol = 'Re S'; s.unit = '-';
    items{end + 1} = s;
end
for k = 1:nOff
    s = entry(sprintf('ImS[%d][%d]', pr(k, 1) - 1, pr(k, 2) - 1), 'S_off_im');
    s.modes = [pr(k, 1) - 1, pr(k, 2) - 1];
    s.symbol = 'Im S'; s.unit = '-';
    items{end + 1} = s;
end
s = entry('Se', 'Se');
s.symbol = 'S_e'; s.unit = '-';
items{end + 1} = s;
end

% ----------------------------------------------------------------------
function items = shape_index(m, sizes)
%SHAPE_INDEX 各阶振型切空间坐标的位置描述。
items = cell(1, sum(sizes));
k = 0;
for i = 1:m
    for c = 1:sizes(i)
        k = k + 1;
        s = entry(sprintf('phi[%d].theta[%d]', i - 1, c - 1), 'phi_tangent');
        s.mode = i - 1; s.coord = c - 1; s.symbol = 'theta'; s.unit = '-';
        items{k} = s;
    end
end
end

% ----------------------------------------------------------------------
function items = single_mode_index(nch)
%SINGLE_MODE_INDEX 单模态频带的参数排列，与 single_mode_cov 的输出一致。
%   排列为 [f, z, S, Se, phi_1 ... phi_n]，共 n+4 项。振型分量只在
%   (I - phi phi') 子空间内有意义，此处仍按通道逐项列出，便于追溯。
items = cell(1, nch + 4);
items{1} = entry('f[0]', 'f');
items{1}.mode = 0; items{1}.symbol = 'f'; items{1}.unit = 'Hz';
items{2} = entry('z[0]', 'z');
items{2}.mode = 0; items{2}.symbol = 'zeta'; items{2}.unit = '-';
items{3} = entry('S[0][0]', 'S_diag');
items{3}.mode = 0; items{3}.symbol = 'S'; items{3}.unit = '-';
items{4} = entry('Se', 'Se');
items{4}.symbol = 'S_e'; items{4}.unit = '-';
for r = 1:nch
    s = entry(sprintf('phi[0].c[%d]', r - 1), 'phi_component');
    s.mode = 0; s.coord = r - 1; s.symbol = 'phi'; s.unit = '-'; s.channel = r - 1;
    items{4 + r} = s;
end
end

% ----------------------------------------------------------------------
function items = async_single_index(nch, ng)
%ASYNC_SINGLE_INDEX 异步单阶的参数排列。
%   [f, z, S, Se_1..Se_ng, theta (nch-1 维), {Re v_ij, Im v_ij}]。
items = cell(1, 0);
s = entry('f[0]', 'f');
s.mode = 0; s.symbol = 'f'; s.unit = 'Hz';
items{end + 1} = s;
s = entry('z[0]', 'z');
s.mode = 0; s.symbol = 'zeta'; s.unit = '-';
items{end + 1} = s;
s = entry('S[0][0]', 'S_diag');
s.mode = 0; s.symbol = 'S'; s.unit = '-';
items{end + 1} = s;
% 各测组各有一条预测误差谱密度，故 Se 按测组编号
for g = 1:ng
    s = entry(sprintf('Se[%d]', g - 1), 'Se');
    s.group = g - 1; s.symbol = 'S_e'; s.unit = '-';
    items{end + 1} = s;
end
% 振型张满通道空间，切空间维数为 nch-1
for c = 1:nch - 1
    s = entry(sprintf('phi[0].theta[%d]', c - 1), 'phi_tangent');
    s.mode = 0; s.coord = c - 1; s.symbol = 'theta'; s.unit = '-';
    items{end + 1} = s;
end
% 组间相干矩阵的实部与虚部，按测组对逐对给出
for g = 1:ng - 1
    for h = g + 1:ng
        s = entry(sprintf('Rev[%d][%d]', g - 1, h - 1), 'v_re');
        s.group = g - 1; s.group_pair = h - 1; s.symbol = 'Re v'; s.unit = '-';
        items{end + 1} = s;
        s = entry(sprintf('Imv[%d][%d]', g - 1, h - 1), 'v_im');
        s.group = g - 1; s.group_pair = h - 1; s.symbol = 'Im v'; s.unit = '-';
        items{end + 1} = s;
    end
end
end

% ----------------------------------------------------------------------
function items = async_multi_index(m, ng, nOff, complexS, nTan, pairs)
%ASYNC_MULTI_INDEX 异步多阶的参数排列。
%   谱参数段为 [f, z, diag(S), S 的非对角元, Se]；S 的非对角元在单测组多阶
%   （S 取复 Hermitian 矩阵）时按实部、虚部分别列出，其余情形只列实部。
%   其后是各阶振型切空间坐标，最后是各阶对的相干矩阵（先模后相位）。
ntri = m * (m - 1) / 2;
pr = triu_pairs(m);
items = cell(1, 0);
for i = 1:m
    s = entry(sprintf('f[%d]', i - 1), 'f');
    s.mode = i - 1; s.symbol = 'f'; s.unit = 'Hz';
    items{end + 1} = s;
end
for i = 1:m
    s = entry(sprintf('z[%d]', i - 1), 'z');
    s.mode = i - 1; s.symbol = 'zeta'; s.unit = '-';
    items{end + 1} = s;
end
for i = 1:m
    s = entry(sprintf('S[%d][%d]', i - 1, i - 1), 'S_diag');
    s.mode = i - 1; s.symbol = 'S'; s.unit = '-';
    items{end + 1} = s;
end
if complexS
    for k = 1:ntri
        s = entry(sprintf('ReS[%d][%d]', pr(k, 1) - 1, pr(k, 2) - 1), 'S_off_re');
        s.modes = [pr(k, 1) - 1, pr(k, 2) - 1];
        s.symbol = 'Re S'; s.unit = '-';
        items{end + 1} = s;
    end
    for k = 1:ntri
        s = entry(sprintf('ImS[%d][%d]', pr(k, 1) - 1, pr(k, 2) - 1), 'S_off_im');
        s.modes = [pr(k, 1) - 1, pr(k, 2) - 1];
        s.symbol = 'Im S'; s.unit = '-';
        items{end + 1} = s;
    end
else
    for k = 1:nOff
        s = entry(sprintf('S[%d][%d]', pr(k, 1) - 1, pr(k, 2) - 1), 'S_off');
        s.modes = [pr(k, 1) - 1, pr(k, 2) - 1];
        s.symbol = 'S'; s.unit = '-';
        items{end + 1} = s;
    end
end
for g = 1:ng
    s = entry(sprintf('Se[%d]', g - 1), 'Se');
    s.group = g - 1; s.symbol = 'S_e'; s.unit = '-';
    items{end + 1} = s;
end
for i = 1:m
    for c = 1:nTan(i)
        s = entry(sprintf('phi[%d].theta[%d]', i - 1, c - 1), 'phi_tangent');
        s.mode = i - 1; s.coord = c - 1; s.symbol = 'theta'; s.unit = '-';
        items{end + 1} = s;
    end
end
% 相干矩阵按阶对逐对给出：先各测组对的模，再各测组对的相位，与后验协方差
% 中「先模后相位」的打包顺序一致
npair = ng * (ng - 1) / 2;
for q = 1:size(pairs, 1)
    i = pairs(q, 1);
    j = pairs(q, 2);
    for g = 1:ng - 1
        for h = g + 1:ng
            s = entry(sprintf('|v|[%d][%d][%d][%d]', i - 1, j - 1, g - 1, h - 1), 'v_mod');
            s.modes = [i - 1, j - 1];
            s.group = g - 1; s.group_pair = h - 1; s.symbol = '|v|'; s.unit = '-';
            items{end + 1} = s;
        end
    end
    for g = 1:ng - 1
        for h = g + 1:ng
            s = entry(sprintf('phase v[%d][%d][%d][%d]', i - 1, j - 1, g - 1, h - 1), 'v_phase');
            s.modes = [i - 1, j - 1];
            s.group = g - 1; s.group_pair = h - 1; s.symbol = 'arg v'; s.unit = 'rad';
            items{end + 1} = s;
        end
    end
    if npair ~= ng * (ng - 1) / 2
        error('bayoma:badLayout', '测组对数不一致');
    end
end
end

% ----------------------------------------------------------------------
function layout = async_layout(m, ng, nch, covSize, nPairs)
%ASYNC_LAYOUT 由协方差维数反推异步多阶的自由参数排布。
%   n_spec = 3m + n_off + ng，其中 n_off = 2*m(m-1)/2（单测组且多阶时 S 取复
%   Hermitian 矩阵）或 m(m-1)/2（其余情形）；形状块总宽为
%   covSize - n_spec - 2*npair*nPairs，单测组多阶时每阶 2*m0-2、其余每阶
%   nch-1，由此定出每阶宽度与子空间维数 m0。
ntri = m * (m - 1) / 2;
complexS = (ng == 1) && (m > 1);
if complexS
    nOff = 2 * ntri;
else
    nOff = ntri;
end
nSpec = 3 * m + nOff + ng;
npair = ng * (ng - 1) / 2;
shapeTotal = covSize - nSpec - 2 * npair * nPairs;
if shapeTotal < 0
    error('bayoma:badLayout', '协方差维数 %d 小于谱参数与相干参数所占的 %d', ...
        covSize, nSpec + 2 * npair * nPairs);
end
if complexS
    if m == 0 || mod(shapeTotal, 2 * m) ~= 0
        error('bayoma:badLayout', '单测组多阶的形状块宽度 %d 不能被 2m = %d 整除', ...
            shapeTotal, 2 * m);
    end
    m0 = shapeTotal / (2 * m) + 1;
    widths = repmat(2 * m0 - 2, 1, m);
    subspaceDim = m0;
else
    if m == 0 || mod(shapeTotal, m) ~= 0
        error('bayoma:badLayout', '形状块宽度 %d 不能被阶数 %d 整除', shapeTotal, m);
    end
    widths = repmat(shapeTotal / m, 1, m);
    subspaceDim = widths(1) + 1;
end
layout = struct('m', m, 'ng', ng, 'nch', nch, 'n_off', nOff, ...
    'complex', complexS, 'n_spec', nSpec, 'npair', npair, ...
    'n_tan', widths, 'subspace_dim', subspaceDim, 'shape_start', nSpec);
end
