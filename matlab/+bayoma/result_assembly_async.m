function asm = result_assembly_async(out, spec)
%RESULT_ASSEMBLY_ASYNC 异步数据识别结果的组装分组。
%
%   ASM = bayoma.result_assembly_async(OUT, SPEC) 把异步数据识别的输出组装成
%   结果文件里的 assembly 分组。OUT 为 bayoma.io_json 内部 run_asynchronous
%   的返回值，SPEC 为描述结构体，用于取几何段与振型图目录。
%
%   单阶情形各测组共享一个节点振型，参数排列为
%   ``[f, z, S, Se_1..Se_ng, theta (nch-1 维), {Re v_ij, Im v_ij}]``；多阶情形
%   参数排列为谱参数段、各阶振型切空间坐标与各阶对的相干模与相位。两种排布
%   都由 bayoma.result_index 给出，多阶的切空间块宽度还须由协方差维数反推
%   （bayoma.result_index 的 'layout' 分支给出 subspace_dim 与 shape_start）。
%
%   切空间基与后验协方差的参数化保持一致：单测组多阶时振型为复向量，
%   先在测组张成的子空间内取坐标 alpha，再在 alpha 的单位球切空间内取基；
%   其余情形振型本身就是通道空间向量，直接在其单位球切空间内取基。由切空间
%   协方差块到通道空间协方差的映射经 bayoma.channel_map 完成。
%
%   返回结构体见 bayoma.result_assembly，另在 shape_uncertainty_per_mode 中给出
%   各阶振型变异系数的简表。

if nargin < 2
    spec = struct();
end

nch = double(out.nch);
m = numel(out.f);
ng = numel(out.Se);
isMulti = isfield(out, 'vij');

geometry = [];
if isfield(spec, 'geometry') && ~isempty(spec.geometry)
    geometry = spec.geometry;
end
figureDir = [];
if isfield(spec, 'figures') && ~isempty(spec.figures) && ...
        isfield(spec.figures, 'dir') && ~isempty(spec.figures.dir)
    figureDir = spec.figures.dir;
end

% 各测组的记录：组号自 0 起计，通道号为该组的通道下标（自 0 起计）
setups = cell(1, ng);
for r = 1:ng
    item = struct();
    item.group_index = r - 1;
    item.channels = double(out.groups{r}(:).') - 1;
    setups{r} = item;
end

cov = double(out.cov);
shapes = double(out.phi);

if isMulti
    pairs = sorted_pairs(out.vij);
    layout = bayoma.result_index('layout', m, ng, nch, size(cov, 1), size(pairs, 1));
    index = bayoma.result_index('async_multi', m, ng, layout.n_off, ...
        layout.complex, layout.n_tan, pairs);
    covPhi = cell(1, m);
    if layout.complex
        % 单测组多阶：先在测组张成的子空间内取坐标，再在单位球切空间内取基
        G = assemble_G(shapes, out.groups, ng);
        [~, B] = column_basis(G);
        alpha = B.' * shapes;
        pos = layout.shape_start;
        for i = 1:m
            base = bayoma.tangent_basis(alpha(:, i), true);
            width = size(base, 2);
            blk = cov(pos + 1:pos + width, pos + 1:pos + width);
            mat = bayoma.channel_map(base, B);
            covPhi{i} = mat * blk * mat.';
            pos = pos + width;
        end
    else
        pos = layout.shape_start;
        for i = 1:m
            base = bayoma.tangent_basis(shapes(:, i));
            width = size(base, 2);
            blk = cov(pos + 1:pos + width, pos + 1:pos + width);
            mat = bayoma.channel_map(base, []);
            covPhi{i} = mat * blk * mat.';
            pos = pos + width;
        end
    end
    asm = bayoma.result_assembly(shapes, cov, index, ...
        'channel_index', 0:nch - 1, ...
        'geometry', geometry, ...
        'figure_dir', figureDir, ...
        'setups', setups, ...
        'shape_covariances', covPhi);
else
    index = bayoma.result_index('async_single', nch, ng);
    % 单阶情形的振型为通道空间的实向量，切空间维数 nch-1
    base = bayoma.tangent_basis(real(shapes(:)));
    pos = index_positions(index, 'phi_tangent');
    blk = cov(pos, pos);
    mat = bayoma.channel_map(base, []);
    covPhi = {mat * blk * mat.'};
    asm = bayoma.result_assembly(shapes(:), cov, index, ...
        'channel_index', 0:nch - 1, ...
        'geometry', geometry, ...
        'figure_dir', figureDir, ...
        'setups', setups, ...
        'shape_covariances', covPhi);
end

% 相干矩阵：单阶只有一组振型，相干矩阵即整块 ng x ng 矩阵，直接放在组装分组里。
% 多阶时相干矩阵按阶对分成 v12、v13 等若干块，键名无法用矩阵表达；同名的整块
% 信息已在结果顶层以 v12/v13 为键给出，组装分组里不再重复，与 Python 侧一致。
if ~isMulti
    asm.coherence = complex_split(complex(out.v));
end

% 各阶振型变异系数的简表
perMode = cell(1, numel(covPhi));
for i = 1:numel(covPhi)
    rec = struct();
    rec.order = i - 1;
    rec.coefv = sqrt(max(trace(double(covPhi{i})), 0));
    perMode{i} = rec;
end
asm.shape_uncertainty_per_mode = perMode;
end

% ----------------------------------------------------------------------
function pos = index_positions(index, kind)
%INDEX_POSITIONS 取指定类别的参数在协方差矩阵中的位置（自 1 起计）。
pos = zeros(1, 0);
for i = 1:numel(index)
    if strcmp(index{i}.kind, kind)
        pos(end + 1) = i; %#ok<AGROW>
    end
end
end

% ----------------------------------------------------------------------
function G = assemble_G(phi, groups, ng)
%ASSEMBLE_G 把各测组的振型拼成块矩阵（第 (i-1)*ng+r 列只在第 r 组的行块上取振型 i）。
nch = size(phi, 1);
m = size(phi, 2);
if isreal(phi)
    G = zeros(nch, m * ng);
else
    G = complex(zeros(nch, m * ng));
end
for i = 1:m
    for r = 1:ng
        idx = groups{r};
        G(idx, (i - 1) * ng + r) = phi(idx, i);
    end
end
end

% ----------------------------------------------------------------------
function [m0, B] = column_basis(G)
%COLUMN_BASIS 矩阵 G 列空间的正交基（复值矩阵按复内积正交归一）。
sv = svd(G);
if isempty(sv) || sv(1) <= 0
    m0 = 0;
    B = zeros(size(G, 1), 0);
    return;
end
m0 = sum(sv > 1e-10 * sv(1));
[Ub, ~] = svd(G, 'econ');
B = Ub(:, 1:m0);
end

% ----------------------------------------------------------------------
function pairs = sorted_pairs(vij)
%SORTED_PAIRS 相干矩阵字段名排序后的阶次对。
%   字段名为 v12、v13 等，先按字符串排序（与 Python 侧的排序一致），再拆出阶次。
names = sort(fieldnames(vij));
pairs = zeros(numel(names), 2);
for i = 1:numel(names)
    tok = regexp(names{i}, '^v(\d+?)(\d+)$', 'tokens', 'once');
    if isempty(tok)
        error('bayoma:badIndex', '相干矩阵字段名无法解析：%s', names{i});
    end
    pairs(i, :) = [str2double(tok{1}), str2double(tok{2})];
end
end

% ----------------------------------------------------------------------
function value = complex_split(mat)
%COMPLEX_SPLIT 复数矩阵转成 re/im 两个字段的实矩阵。
mat = complex(mat);
value = struct();
value.re = real(mat);
value.im = imag(mat);
end
