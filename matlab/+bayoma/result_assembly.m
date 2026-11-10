function out = result_assembly(phi, cov, index, varargin)
%RESULT_ASSEMBLY 组装结果分组：全局振型、识别与组装不确定性的合成、后验协方差与索引。
%
%   OUT = bayoma.result_assembly(PHI, COV, INDEX) 中 PHI 为组装后的全局振型矩阵
%   （每列一阶），COV 为全部参数的后验协方差矩阵，INDEX 为参数描述元胞（每项
%   含 name 与 kind，来自 bayoma.result_index）。参数描述的项数须等于 COV 的维数。
%
%   可选参数以名值对给出：
%       'channel_index'          通道编号，缺省 0:nch-1
%       'geometry'               几何段，给出时另绘空间振型图
%       'figure_dir'             振型图保存目录
%       'setups'                 各测组的记录（元胞）
%       'shape_covariances'      各阶振型在通道空间的协方差（元胞）
%       'shape_bases'            各阶切空间基（元胞），给出时由 COV 的振型块映射
%       'shape_channel_basis'    各阶的子空间基（元胞），可与 shape_bases 配合
%
%   返回结构体字段：phi、channel_index、shape_uncertainty、uncertainty、
%   covariance_total、covariance_from_assembly、covariance_recognition、
%   posterior_covariance、index、setups、spatial。
%
%   全局振型的边际协方差取 C_uu；其中可由其余参数线性解释的部分记为组装阶段
%   引入的不确定性，余下为识别本身的贡献，两者按迹相加等于总方差。这是高斯
%   近似下的条件方差分解。

opt = struct('channel_index', [], 'geometry', [], 'figure_dir', [], ...
    'setups', [], 'shape_covariances', [], 'shape_bases', [], ...
    'shape_channel_basis', []);
if mod(numel(varargin), 2) ~= 0
    error('bayoma:badOption', '可选参数须以名值对给出');
end
for k = 1:2:numel(varargin)
    name = varargin{k};
    if ~isfield(opt, name)
        error('bayoma:badOption', '未知的可选参数：%s', name);
    end
    opt.(name) = varargin{k + 1};
end

nch = size(phi, 1);
channelIndex = opt.channel_index;
if isempty(channelIndex)
    channelIndex = 0:nch - 1;
end
channelIndex = double(channelIndex(:).');

cov = double(cov);
if numel(index) ~= size(cov, 1)
    error('bayoma:badIndex', '参数描述 %d 项与协方差维数 %d 不符', ...
        numel(index), size(cov, 1));
end

% 按参数类别分组：振型切空间坐标为 u，其余参数为 theta
shapeIdx = [];
thetaIdx = [];
for i = 1:numel(index)
    if strcmp(index{i}.kind, 'phi_tangent')
        shapeIdx(end + 1) = i; %#ok<AGROW>
    else
        thetaIdx(end + 1) = i; %#ok<AGROW>
    end
end
dec = conditional_decomposition(cov, shapeIdx, thetaIdx);

entries = cell(1, numel(index));
for pos = 1:numel(index)
    rec = index{pos};
    rec.index = pos - 1;
    entries{pos} = rec;
end

totalTrace = max(dec.trace_total, 0);

% 逐阶给出通道空间的振型协方差：按各阶切空间宽度切块，再用子空间基映射回
% 通道空间。切空间基的列数即该阶的坐标宽度。
shapeCovs = [];
if ~isempty(opt.shape_bases)
    pos = 0;
    shapeCovs = cell(1, numel(opt.shape_bases));
    for i = 1:numel(opt.shape_bases)
        base = opt.shape_bases{i};
        cb = [];
        if ~isempty(opt.shape_channel_basis)
            cb = opt.shape_channel_basis{i};
        end
        width = size(base, 2);
        blk = dec.total(pos + 1:pos + width, pos + 1:pos + width);
        mat = bayoma.channel_map(base, cb);
        shapeCovs{i} = mat * blk * mat.';
        pos = pos + width;
    end
elseif ~isempty(opt.shape_covariances)
    shapeCovs = cell(1, numel(opt.shape_covariances));
    for i = 1:numel(opt.shape_covariances)
        shapeCovs{i} = double(opt.shape_covariances{i});
    end
end

perMode = {};
if ~isempty(shapeCovs)
    for i = 1:numel(shapeCovs)
        mat = double(shapeCovs{i});
        if size(mat, 1) == 2 * nch
            % 复形状：协方差按「实部在前、虚部在后」排布，逐通道取两块对角之和
            varCh = max(diag(mat(1:nch, 1:nch)) + diag(mat(nch + 1:end, nch + 1:end)), 0);
        else
            varCh = max(diag(mat), 0);
        end
        stdCh = sqrt(varCh);
        rec = struct();
        rec.order = i - 1;
        rec.complex = logical(size(mat, 1) == 2 * nch);
        rec.coefv = sqrt(max(trace(mat), 0));
        rec.channel_index = channelIndex;
        rec.std_per_channel = stdCh(:).';
        rec.ci95_halfwidth_per_channel = 1.96 * stdCh(:).';
        rec.shape = bayoma.json_null();
        rec.cov = mat;
        perMode{end + 1} = rec; %#ok<AGROW>
    end
else
    rec = struct();
    rec.order = 0;
    rec.complex = false;
    rec.coefv = sqrt(totalTrace);
    rec.channel_index = channelIndex;
    rec.std_per_channel = zeros(1, nch);
    rec.ci95_halfwidth_per_channel = zeros(1, nch);
    rec.shape = bayoma.json_null();
    rec.cov = dec.total;
    perMode{1} = rec;
end

% 顶层逐通道标准差按各阶方差的算术平均给出：单阶即该阶的标准差，多阶为各阶的
% 平均不确定性，便于调用方在不区分阶次时也有一组通道级数字。
chanVar = zeros(1, nch);
for i = 1:numel(perMode)
    vals = perMode{i}.std_per_channel;
    if numel(vals) == nch
        chanVar = chanVar + vals .^ 2;
    end
end
if ~isempty(perMode)
    chanVar = chanVar / numel(perMode);
end
topStd = sqrt(max(chanVar, 0));

shapeUnc = struct();
shapeUnc.coefv = sqrt(totalTrace);
shapeUnc.std_per_channel = topStd;
shapeUnc.ci95_halfwidth_per_channel = 1.96 * topStd;
shapeUnc.channel_index = channelIndex;
shapeUnc.per_mode = perMode;

unc = struct();
unc.method = '高斯近似下的条件方差分解：C_uu = C_uθ C_θθ^+ C_θu + 余项';
unc.total_coefv = sqrt(totalTrace);
unc.assembly_coefv = sqrt(max(dec.trace_from_assembly, 0));
unc.recognition_coefv = sqrt(max(dec.trace_residual, 0));
unc.trace_total = dec.trace_total;
unc.trace_assembly = dec.trace_from_assembly;
unc.trace_recognition = dec.trace_residual;
unc.note = ['组装项为可由其余参数（测组噪声谱、相干矩阵等）线性解释的振型方差；' ...
    '识别项为余下方差。两项按迹相加等于总方差。'];

out = struct();
% 复振型按 re/im 分开给出，实振型直接给矩阵
if isreal(phi)
    out.phi = double(phi);
else
    out.phi = struct('re', real(double(phi)), 'im', imag(double(phi)));
end
out.channel_index = channelIndex;
out.shape_uncertainty = shapeUnc;
out.uncertainty = unc;
out.covariance_total = dec.total;
out.covariance_from_assembly = dec.from_assembly;
out.covariance_recognition = dec.residual;
out.posterior_covariance = cov;
out.index = entries;
if isempty(opt.setups)
    out.setups = bayoma.json_null();
else
    out.setups = opt.setups;
end
out.spatial = bayoma.json_null();

if ~isempty(opt.geometry)
    % 单阶传一维、多阶传矩阵，统一成列矩阵后再画
    if isvector(phi)
        shapes = phi(:);
    else
        shapes = phi;
    end
    out.spatial = bayoma.result_spatial(shapes, opt.geometry, opt.figure_dir, 'assembly');
end

if isempty(opt.setups)
    out.setups = {};
end
end

% ----------------------------------------------------------------------
function out = conditional_decomposition(cov, shapeIdx, thetaIdx)
%CONDITIONAL_DECOMPOSITION 把振型协方差按「由其他参数解释」与「其余」作分解。
%
%   C_uu 为振型的边际协方差，C_uθ C_θθ^+ C_θu 为可由 θ 线性解释的部分，余下为
%   偏协方差。两项都半正定，在高斯近似下分别对应组装阶段引入的不确定性
%   与识别本身的其余不确定性。
su = shapeIdx(:);
st = thetaIdx(:);
uu = cov(su, su);
if isempty(st)
    out = struct('total', uu, 'from_assembly', zeros(size(uu)), ...
        'residual', uu, 'trace_total', trace(uu), ...
        'trace_from_assembly', 0, 'trace_residual', trace(uu));
    return;
end
tt = cov(st, st);
ut = cov(su, st);
% 用伪逆而非逆：θ 段可能含近似共线的参数（例如相邻阶的模态力谱）
ttInv = pinv(0.5 * (tt + tt.'), 1e-12);
explained = ut * ttInv * ut.';
explained = 0.5 * (explained + explained.');
residual = 0.5 * ((uu - explained) + (uu - explained).');
out = struct('total', uu, 'from_assembly', explained, ...
    'residual', residual, 'trace_total', trace(uu), ...
    'trace_from_assembly', trace(explained), 'trace_residual', trace(residual));
end
