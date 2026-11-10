function result = io_json(specPath, outPath)
%IO_JSON 机器可读接口层：按 JSON 描述执行识别，把结果写成 JSON。
%
%   RESULT = bayoma.io_json(SPECPATH) 读取描述文件 SPECPATH，执行识别，
%   按描述中的 output.path 写出结果文件，并返回结果结构体。
%
%   RESULT = bayoma.io_json(SPECPATH, OUTPATH) 用 OUTPATH 覆盖描述中的
%   输出路径。
%
%   描述文件与结果文件的字段与 Python 实现 bayoma.io_json 完全一致，
%   两套实现可读同一份描述、产出同一套字段，便于交叉比对。
%
%   描述文件字段
%       data       数据来源结构体：path、format（csv/npy/mat）、fs、key、
%                  delimiter、skip_header
%       fs         采样频率（Hz），data.fs 优先
%       bands      识别频带，形如 [[0.9,1.1],[1.3,1.5]]；缺省时自动选带
%       f0         各频带内固有频率初值，形如 [[1.0],[1.4]]
%       options    q、p、maxiter、mute、ichan、iband、f1f2d
%       multi_setup 多测组描述：n_dof、setups（每项含 data、dofs、bands、f0）
%       asynchronous 异步数据描述：data、groups、bands、f0、q，给出时按第 10.4 节
%                  的异步模型处理（组内同步、组间用复相干矩阵）；
%                  三类描述互斥，优先级 asynchronous、multi_setup、单测组
%       output     结果文件路径：path
%
%   结果文件字段与 bayoma.bayoma 的输出一致，另加 input（记录长度等）、
%   n_modes（模态总数）、elapsed（总耗时）。复数矩阵按 {"re":…,"im":…}
%   形式给出。
%
%   参见 bayoma.bayoma、bayoma.multi_setup。

if nargin < 1 || isempty(specPath)
    error('bayoma:badSpec', '必须给出描述文件路径');
end
if nargin < 2
    outPath = '';
end

t0 = tic;
spec = jsondecode(fileread_utf8(specPath));
if ~isempty(outPath)
    spec.output = struct('path', outPath);
end

result = run_from_dict(spec);
result.elapsed = toc(t0);

target = output_path(spec);
if ~isempty(target)
    write_json(target, result);
end
end

% ------------------------------------------------------------------
function result = run_from_dict(spec)
%RUN_FROM_DICT 按描述结构体执行识别。
%   多测组描述的采样频率可写在顶层 fs，也可写在任一个测组的数据来源里，
%   因此多测组分支先于顶层 data 的读取。
if isfield(spec, 'asynchronous') && ~isempty(spec.asynchronous)
    out = run_asynchronous(spec);
    result = async_result_to_dict(out);
    arr = load_data(spec.asynchronous.data);
    fsA = 0.0;
    if isfield(spec.asynchronous.data, 'fs') && ~isempty(spec.asynchronous.data.fs)
        fsA = double(spec.asynchronous.data.fs);
    elseif isfield(spec, 'fs') && ~isempty(spec.fs)
        fsA = double(spec.fs);
    end
    if fsA <= 0.0
        error('bayoma:badFs', '异步数据识别必须给出正的采样频率 fs');
    end
    result.input = struct('fs', fsA, 'nch', size(arr, 2), ...
        'nt', size(arr, 1), 'duration', size(arr, 1) / fsA, ...
        'n_group', numel(spec.asynchronous.groups));
    % 组装分组与识别分组：异步数据在单频带内一次给出全部参数，后验协方差为
    % 整体矩阵，故识别分组的联合协方差按 structure = 'whole' 给出
    result.assembly = bayoma.result_assembly_async(out, spec);
    result.recognition = async_recognition(result, out, spec);
    return;
end

if isfield(spec, 'multi_setup') && ~isempty(spec.multi_setup)
    fsMs = 0.0;
    if isfield(spec, 'fs') && ~isempty(spec.fs)
        fsMs = double(spec.fs);
    end
    if fsMs <= 0.0
        for i = 1:numel(spec.multi_setup.setups)
            cand = spec.multi_setup.setups(i).data;
            if isfield(cand, 'fs') && ~isempty(cand.fs)
                fsMs = double(cand.fs);
                break;
            end
        end
    end
    if fsMs <= 0.0
        error('bayoma:badFs', '多测组识别必须给出正的采样频率 fs');
    end
    out = run_multi_setup(spec, fsMs);
    result = result_to_dict(out);
    nt = 0;
    for i = 1:numel(spec.multi_setup.setups)
        arr = load_data(spec.multi_setup.setups(i).data);
        nt = max(nt, size(arr, 1));
    end
    result.input = struct('fs', fsMs, 'nch', double(out.nch), ...
        'nt', nt, 'duration', nt / fsMs, ...
        'n_setup', numel(spec.multi_setup.setups));
    return;
end

dataSpec = spec.data;
data = load_data(dataSpec);

fs = 0.0;
if isfield(dataSpec, 'fs') && ~isempty(dataSpec.fs)
    fs = double(dataSpec.fs);
elseif isfield(spec, 'fs') && ~isempty(spec.fs)
    fs = double(spec.fs);
end
if fs <= 0.0
    error('bayoma:badFs', '必须给出正的采样频率 fs');
end

IN = struct('tdata', data, 'fs', fs);
if isfield(spec, 'bands') && ~isempty(spec.bands)
    IN.f1f2 = to_band_matrix(spec.bands);
end
if isfield(spec, 'f0') && ~isempty(spec.f0)
    IN.f0 = to_f0_cell(spec.f0);
end
if isfield(spec, 'options') && ~isempty(spec.options)
    IN = merge_options(IN, spec.options);
end
% 几何、出图目录与谱表：传递给总入口，用于补充识别分组中的空间振型图与
% PSD/SV 谱表；缺省时识别分组里对应字段写 null
if isfield(spec, 'geometry') && ~isempty(spec.geometry)
    IN.geometry = spec.geometry;
end
if isfield(spec, 'figures') && ~isempty(spec.figures) && ...
        isfield(spec.figures, 'dir') && ~isempty(spec.figures.dir)
    IN.figure_dir = spec.figures.dir;
end
if isfield(spec, 'spectra') && ~isempty(spec.spectra)
    IN.spectra = logical(spec.spectra);
end
out = bayoma.bayoma(IN);

result = result_to_dict(out);
result.input = struct('fs', fs, 'nch', size(data, 2), ...
    'nt', size(data, 1), 'duration', size(data, 1) / fs);
end

% ------------------------------------------------------------------
function out = run_asynchronous(spec)
%RUN_ASYNCHRONOUS 按描述执行异步数据识别。
raw = spec.asynchronous;
arr = load_data(raw.data);

fsA = 0.0;
if isfield(raw.data, 'fs') && ~isempty(raw.data.fs)
    fsA = double(raw.data.fs);
elseif isfield(spec, 'fs') && ~isempty(spec.fs)
    fsA = double(spec.fs);
end
if fsA <= 0.0
    error('bayoma:badFs', '异步数据识别必须给出正的采样频率 fs');
end
if ~isfield(raw, 'bands') || isempty(raw.bands)
    error('bayoma:badSpec', '异步数据识别需要给出识别频带 bands');
end

band = to_band_matrix(raw.bands);
band = band(1, :);
Sspec = bayoma.scaled_fft(arr, fsA, band, option_value(spec, 'p', 0));

% JSON 描述里的分组编号与 Python 实现一致，从 0 起计；MATLAB 内部从 1 起计
groups = to_groups_cell(raw.groups);
for i = 1:numel(groups)
    groups{i} = groups{i} + 1;
end

opts = struct('q', 0);
if isfield(raw, 'q') && ~isempty(raw.q)
    opts.q = double(raw.q);
else
    opts.q = option_value(spec, 'q', 0);
end
if isfield(raw, 'f0') && ~isempty(raw.f0)
    f0cell = to_f0_cell(raw.f0);
    opts.f0 = f0cell{1};
end

% 频带内的模态阶数，缺省 1；大于 1 时走第 10.4 节的多阶模型
nMod = 1;
if isfield(raw, 'n_modes') && ~isempty(raw.n_modes)
    nMod = double(raw.n_modes);
elseif isfield(raw, 'f0') && ~isempty(raw.f0)
    f0probe = to_f0_cell(raw.f0);
    nMod = numel(f0probe{1});
end
if nMod < 1 || nMod ~= round(nMod)
    error('bayoma:badSpec', '异步数据识别的模态阶数 n_modes 必须为正整数');
end

if nMod > 1
    out = bayoma.asynchronous_multi_mode(Sspec.freqs, Sspec.fft, groups, nMod, opts);
else
    out = bayoma.asynchronous_single_mode(Sspec.freqs, Sspec.fft, groups, opts);
end
out.bands = band;
out.nch = size(arr, 2);
out.fs = fsA;
out.n_modes = nMod;
end

% ------------------------------------------------------------------
function result = async_result_to_dict(out)
%ASYNC_RESULT_TO_DICT 把异步识别结果转成与 Python 侧同构的结构体。
%   字段约定：coefv 用 S 而非 Sii，另给 groups 与 coherence。单阶实现给出
%   f、z、S、Se、phi、v，多阶实现另给 vij 结构体；其余汇总量在这里补算。
%   形状展开同样交给 to_json_tree。
isMulti = isfield(out, 'vij');
result = struct();
result.f = double(out.f(:).');
result.z = double(out.z(:).');
Smat = complex(out.S);
% 多阶时 S 为 m x m 矩阵，对角线即各阶模态力谱密度
result.Sii = real(diag(Smat)).';
result.Se = double(out.Se(:).');
result.rms = sqrt(pi * real(diag(Smat)).' .* out.f(:).' ./ (4 * out.z(:).'));
result.sn = real(diag(Smat)).' ./ (4 * mean(out.Se) * out.z(:).'.^2);
% 单测组多阶时振型为复向量，按 {"re", "im"} 给出，与结果契约的复数写法一致
if isreal(out.phi)
    result.phi = double(out.phi);
else
    result.phi = complex_split(double(out.phi));
end

result.S = {complex_split(Smat)};
result.cmat = {double(out.cov)};

result.coefv = struct();
result.coefv.f = double(out.coefv.f(:).');
result.coefv.z = double(out.coefv.z(:).');
if isMulti
    % 多阶时 coefv.S 为 m x m 实矩阵，对角线是各阶对角元的变异系数，
    % 非对角元是实部与虚部合成后的变异系数（见异步多阶实现文档）
    sd = real(diag(complex(out.coefv.S))).';
    result.coefv.S = sd;
    nOff = numel(out.f) * (numel(out.f) - 1) / 2;
    if nOff > 0
        so = complex(out.coefv.S);
        offv = zeros(1, nOff);
        k = 0;
        for i = 1:numel(out.f) - 1
            for j = i + 1:numel(out.f)
                k = k + 1;
                offv(k) = real(so(i, j));
            end
        end
        result.coefv.S_off = offv;
    end
else
    result.coefv.S = double(out.coefv.S(:).');
end
result.coefv.Se = double(out.coefv.Se(:).');
result.coefv.phi = double(out.coefv.phi(:).');
if isMulti
    % 多阶时各阶对的相干矩阵存放在 vij 结构体里，coefv.v 的同名字段给出该对
    % 模与相位合成后的变异系数；取各对的平均作为整体汇总量
    names = fieldnames(out.coefv.v);
    vals = zeros(1, numel(names));
    for i = 1:numel(names)
        item = out.coefv.v.(names{i});
        if isstruct(item) && isfield(item, 'mod')
            vals(i) = max(double(item.mod(:)));
        else
            vals(i) = max(double(item(:)));
        end
    end
    if isempty(vals)
        result.coefv.v = 0.0;
    else
        result.coefv.v = mean(vals);
    end
else
    result.coefv.v = double(out.coefv.v);
end
result.coefv.rms = 0.5 * sqrt( ...
    out.coefv.f(:).'.^2 + out.coefv.z(:).'.^2 + real(diag(Smat)).'.^2);

result.bands = double(out.bands);
result.time_req = zeros(2, 1);
result.nch = double(out.nch);
result.fs = double(out.fs);
if isfield(out, 'n_modes')
    result.n_modes = double(out.n_modes);
else
    result.n_modes = 1;
end
% 输出时还原成 0 起计，与描述文件和 Python 侧一致
gOut = cell(1, numel(out.groups));
for i = 1:numel(out.groups)
    gOut{i} = double(out.groups{i}(:)) - 1;
end
result.groups = gOut;
if isMulti
    % 各阶对的相干矩阵按三维数组存放，最后一维为阶对数；输出时拆成结构体，
    % 每个字段是该阶对的 {"re", "im"} 矩阵，键为 v12、v13 等，
    % 与 Python 侧的字典一致。
    pairs = fieldnames(out.vij);
    coh = struct();
    for i = 1:numel(pairs)
        coh.(pairs{i}) = complex_split(complex(out.vij.(pairs{i})));
    end
    result.coherence = coh;
else
    result.coherence = complex_split(complex(out.v));
end
end

% ------------------------------------------------------------------
function out = run_multi_setup(spec, fs)
%RUN_MULTI_SETUP 按描述执行多测组识别。
ms = spec.multi_setup;
nDof = double(ms.n_dof);

setups = repmat(struct('freqs', [], 'fft', [], 'dofs', []), 1, numel(ms.setups));
for i = 1:numel(ms.setups)
    item = ms.setups(i);
    arr = load_data(item.data);
    if ~isfield(item, 'bands') || isempty(item.bands)
        error('bayoma:badSpec', '多测组的每个测组都需要给出识别频带');
    end
    band = double(item.bands);
    Sspec = bayoma.scaled_fft(arr, fs, band(1, :), option_value(spec, 'p', 0));
    setups(i).freqs = Sspec.freqs;
    setups(i).fft = Sspec.fft;
    % JSON 描述里的自由度编号与 Python 实现一致，从 0 起计；MATLAB 内部从 1 起计
    setups(i).dofs = double(item.dofs(:).') + 1;
end

opts = struct('q', option_value(spec, 'q', 0), ...
    'maxiter', option_value(spec, 'maxiter', 200));
out = bayoma.multi_setup(setups, nDof, opts);

% 多测组结果转成与单测组同构的结构体，并接入后验协方差
ns = numel(out.f);
out.Sii = real(out.S(:).');
out.rms = sqrt(pi * real(out.S(:).') .* out.f(:).' ./ (4 * out.z(:).'));
out.sn = real(out.S(:).') ./ (4 * out.Se(:).' .* out.z(:).'.^2);
cov = bayoma.multi_setup_cov(out, setups, opts);
out.coefv = struct('f', cov.coefv.f(:).', 'z', cov.coefv.z(:).', ...
    'Sii', cov.coefv.S(:).', 'Se', cov.coefv.Se(:).', ...
    'phi', cov.coefv.phi, ...
    'rms', 0.5 * sqrt(cov.coefv.f(:).'.^2 + cov.coefv.z(:).'.^2 + cov.coefv.S(:).'.^2));
out.cmat = {cov.matrix};
out.S = num2cell(out.S(:).');
out.time_req = zeros(2, 1);
out.bands = double(ms.setups(1).bands);
out.nch = nDof;
out.fs = fs;
out.n_setup = ns;
end

% ------------------------------------------------------------------
function data = load_data(spec)
%LOAD_DATA 按描述读取数据矩阵。
if ~isfield(spec, 'path') || isempty(spec.path)
    error('bayoma:badSpec', '数据描述缺少 path 字段');
end
path = spec.path;
if exist(path, 'file') ~= 2
    error('bayoma:noData', '数据文件不存在：%s', path);
end

fmt = '';
if isfield(spec, 'format') && ~isempty(spec.format)
    fmt = lower(spec.format);
end
if isempty(fmt)
    [~, ~, ext] = fileparts(path);
    fmt = lower(ext(2:end));
end
if any(strcmp(fmt, {'txt', 'dat'}))
    fmt = 'csv';
end

switch fmt
    case 'csv'
        delimiter = ',';
        if isfield(spec, 'delimiter') && ~isempty(spec.delimiter)
            delimiter = spec.delimiter;
        end
        skip = 0;
        if isfield(spec, 'skip_header') && ~isempty(spec.skip_header)
            skip = double(spec.skip_header);
        end
        data = dlmread(path, delimiter, skip, 0);
    case 'mat'
        key = 'tdata';
        if isfield(spec, 'key') && ~isempty(spec.key)
            key = spec.key;
        end
        content = load(path);
        if ~isfield(content, key)
            error('bayoma:noData', 'mat 文件 %s 中没有变量 %s', path, key);
        end
        data = content.(key);
    otherwise
        error('bayoma:badFormat', '不支持的数据格式：%s', fmt);
end

data = double(data);
if isrow(data)
    data = data(:);
end
if ndims(data) ~= 2
    error('bayoma:badData', '数据必须是二维矩阵，行对应时刻、列对应通道');
end
end

% ------------------------------------------------------------------
function result = result_to_dict(out)
%RESULT_TO_DICT 把总入口输出转成与 Python 侧同构的结构体。
%   这里只做字段搬运与改名，数值保持 MATLAB 原生数组；写文件时由
%   to_json_tree 统一展开形状（长度 1 的数组写成 [x]，矩阵写成 [[...]]），
%   避免编码规则散落在两处。
result = struct();
for name = {'f', 'z', 'Sii', 'Se', 'rms', 'sn'}
    result.(name{1}) = double(out.(name{1})(:).');
end
result.phi = double(out.phi);

Slist = cell(1, numel(out.S));
for i = 1:numel(out.S)
    Slist{i} = complex_split(out.S{i});
end
result.S = Slist;

cmat = cell(1, numel(out.cmat));
for i = 1:numel(out.cmat)
    cmat{i} = double(out.cmat{i});
end
result.cmat = cmat;

cv = out.coefv;
result.coefv = struct();
for name = {'f', 'z', 'phi', 'rms'}
    result.coefv.(name{1}) = double(cv.(name{1})(:).');
end
result.coefv.Sii = double(cv.Sii(:).');
result.coefv.Se = double(cv.Se(:).');

result.bands = double(out.bands);
result.time_req = double(out.time_req);
result.nch = double(out.nch);
result.fs = double(out.fs);
result.n_modes = numel(out.f);
% 识别结果分组（频带、阶次映射、区分度、联合协方差、逐阶振型不确定性、谱与
% 空间可视化）；同步识别一定给出，故直接搬运
if isfield(out, 'recognition') && ~isempty(out.recognition)
    result.recognition = out.recognition;
end
end

% ------------------------------------------------------------------
function value = complex_split(mat)
%COMPLEX_SPLIT 复数矩阵转成 re/im 两个字段的实矩阵。
%   Python 侧对应 {"re": [[...]], "im": [[...]]}，两部分都按行给出。
mat = complex(mat);
value = struct();
value.re = real(mat);
value.im = imag(mat);
end

% ------------------------------------------------------------------
function cellGroups = to_groups_cell(raw)
%TO_GROUPS_CELL 把 JSON 解出的分组数组整理成元胞，每项一个通道编号行向量。
%   jsondecode 把 [[0,1],[2,3]] 解成 2x2 数值矩阵，把 [[0,1,2],[3]]
%   解成元胞。两种都要接住，否则会误用花括号索引。
if iscell(raw)
    cellGroups = cell(1, numel(raw));
    for i = 1:numel(raw)
        cellGroups{i} = double(raw{i}(:)');
    end
    return;
end
mat = double(raw);
if isvector(mat)
    cellGroups = {mat(:)'};
    return;
end
cellGroups = cell(1, size(mat, 1));
for i = 1:size(mat, 1)
    row = mat(i, :);
    cellGroups{i} = row(~isnan(row));
end
end

% ------------------------------------------------------------------
function value = complex_to_json(mat)
%COMPLEX_TO_JSON 复数矩阵转成 re/im 两部分的实矩阵。
value = complex_split(mat);
end

% ------------------------------------------------------------------
function mat = to_band_matrix(raw)
%TO_BAND_MATRIX 把 JSON 解出的频带数组整理成 nb x 2 矩阵。
%   jsondecode 会把 [[0.9,1.1]] 解成 2x1 列向量，需要按「每行一个频带」
%   还原，否则频带数会被误判。
mat = double(raw);
if isvector(mat)
    if numel(mat) ~= 2
        error('bayoma:badBands', '频带必须是每行两个元素（下界、上界）');
    end
    mat = reshape(mat, 1, 2);
    return;
end
if size(mat, 2) ~= 2
    if size(mat, 1) == 2
        mat = mat.';
    else
        error('bayoma:badBands', '频带矩阵必须是 nb x 2');
    end
end
end

% ------------------------------------------------------------------
function cellF0 = to_f0_cell(raw)
%TO_F0_CELL 把 JSON 解出的初值数组转成 cell，每项一个频带的初值行向量。
%   jsondecode 会把 [1.0, 1.4] 解成 2x1 列向量，若按行数当作频带数，单个频带的
%   两阶初值会被拆成两个频带、每带一个初值。故先判是否为向量：向量一律视为
%   「一个频带的多个初值」。
if iscell(raw)
    cellF0 = raw;
    return;
end
if isstruct(raw)
    error('bayoma:badF0', 'f0 的格式无法识别');
end
if isvector(raw)
    cellF0 = {double(raw(:).')};
    return;
end
cellF0 = cell(size(raw, 1), 1);
for i = 1:size(raw, 1)
    cellF0{i} = double(raw(i, :));
    cellF0{i} = cellF0{i}(~isnan(cellF0{i}));
end
end

% ------------------------------------------------------------------
function IN = merge_options(IN, opts)
%MERGE_OPTIONS 把 options 结构体并入总入口的输入结构体。
names = fieldnames(opts);
for i = 1:numel(names)
    name = names{i};
    if strcmp(name, 'f1f2d')
        IN.f1f2d = double(opts.(name));
    else
        IN.(name) = opts.(name);
    end
end
end

% ------------------------------------------------------------------
function value = option_value(spec, name, fallback)
%OPTION_VALUE 取 options 中的字段，缺省时返回 fallback。
value = fallback;
if isfield(spec, 'options') && ~isempty(spec.options) && isfield(spec.options, name) ...
        && ~isempty(spec.options.(name))
    value = spec.options.(name);
end
end

% ------------------------------------------------------------------
function path = output_path(spec)
%OUTPUT_PATH 取描述中的输出路径，缺省返回空串。
path = '';
if isfield(spec, 'output') && ~isempty(spec.output) && isfield(spec.output, 'path')
    path = spec.output.path;
end
end

% ------------------------------------------------------------------
function rec = async_recognition(result, out, spec)
%ASYNC_RECOGNITION 异步数据识别结果的识别分组。
%
%   异步数据的全部参数在同一条频带内由联合似然一次定出，故联合协方差即组装
%   分组的后验协方差整体，不分块；频带表与阶次映射按描述里给出的频带与阶数
%   填写，区分度表只在一个频带内比较，故频带号恒为 0。
%
%   RESULT 为已填好基础字段的结果结构体（取 f、z、phi 与 n_modes），OUT 为
%   run_asynchronous 的返回值，SPEC 为描述结构体。

nMod = double(result.n_modes);
fArr = double(result.f(:).');
zArr = double(result.z(:).');
phiMat = double(out.phi);

% 描述里的频带与初值：频带取第一条（异步识别只用单条频带），阶数取模态总数
bands = double(out.bands);
if size(bands, 1) > 1
    bands = bands(1, :);
end
nBands = size(bands, 1);
f0List = cell(1, nBands);
if isfield(spec, 'asynchronous') && ~isempty(spec.asynchronous) && ...
        isfield(spec.asynchronous, 'f0') && ~isempty(spec.asynchronous.f0)
    cand = to_f0_cell(spec.asynchronous.f0);
elseif isfield(spec, 'f0') && ~isempty(spec.f0)
    cand = to_f0_cell(spec.f0);
else
    cand = {};
end
for i = 1:min(nBands, numel(cand))
    f0List{i} = cand{i};
end
nModsPerBand = repmat(nMod, 1, nBands);

bandTable = bayoma.result_band_table(bands, f0List, nModsPerBand);

% 全部模态落在第 0 条频带内，频带内次序即全局次序
modes = cell(1, nMod);
for i = 1:nMod
    modes{i} = struct('order', i - 1, 'band_index', 0, 'order_in_band', i - 1);
end

% 异步数据的协方差为整体矩阵，区分度表不传联合协方差，故不给频率相关系数
separation = bayoma.result_separation(fArr, zArr, phiMat, zeros(1, nMod));

asm = result.assembly;
joint = struct();
joint.index = asm.index;
joint.size = size(asm.posterior_covariance, 1);
joint.structure = 'whole';
joint.note = '异步数据在单频带内一次给出全部参数，协方差为整体矩阵，不分块。';

rec = struct();
rec.bands = bandTable;
rec.modes = modes;
rec.separation = separation;
rec.joint_covariance = joint;
rec.joint_covariance_matrix = asm.posterior_covariance;
rec.shape_uncertainty = asm.shape_uncertainty.per_mode;
end

% ------------------------------------------------------------------
function write_json(path, result)
%WRITE_JSON 以 UTF-8 写出 JSON 结果文件。
%   用 encode_json 而不是 jsonencode：后者会把 1x1 数组写成标量、把行向量
%   写成 [a,b] 而非 [[a,b]]，与 Python 侧契约不一致。
folder = fileparts(path);
if ~isempty(folder) && exist(folder, 'dir') ~= 7
    mkdir(folder);
end
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid < 0
    error('bayoma:writeFail', '无法写入结果文件：%s', path);
end
fprintf(fid, '%s', bayoma.encode_json(bayoma.to_json_tree(result)));
fclose(fid);
end

% ------------------------------------------------------------------
function text = fileread_utf8(path)
%FILEREAD_UTF8 以 UTF-8 读取文本文件。
fid = fopen(path, 'r', 'n', 'UTF-8');
if fid < 0
    error('bayoma:noSpec', '无法读取描述文件：%s', path);
end
text = fread(fid, [1, inf], '*char');
fclose(fid);
end
