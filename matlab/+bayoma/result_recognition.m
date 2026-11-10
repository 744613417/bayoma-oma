function out = result_recognition(f, z, phi, bands, f0List, nModsPerBand, blocks, varargin)
%RESULT_RECOGNITION 识别结果分组：频带对应、密集模态区分度、联合后验协方差、
%   逐阶振型不确定性，以及可选的功率谱与空间振型图。
%
%   OUT = bayoma.result_recognition(F, Z, PHI, BANDS, F0LIST, NMODES, BLOCKS)
%   中 F、Z 为各阶固有频率与阻尼比，PHI 为振型矩阵（每列一阶），BANDS 为
%   nb x 2 的频带矩阵，F0LIST 为各频带频率初值的元胞，NMODES 为各频带阶数，
%   BLOCKS 为元胞，每项一个频带的参与量：
%       cov           该频带的后验协方差矩阵
%       cov_phi       各阶振型在通道空间的协方差（元胞）
%       m             该频带的模态阶数
%       n_off         模态力谱非对角元个数 m(m-1)/2
%       sizes         各阶振型切空间坐标个数
%       band_index    频带序号（自 0 起计）
%       offset        该频带首阶在全局的序号
%       complex       振型是否为复数
%       index         可选，该频带的参数描述
%       channel_index 可选，通道编号
%
%   可选参数以名值对给出：
%       'channel_index'  通道编号，缺省 0:nch-1
%       'data'           时程数据，给出且 spectra 为真时另算功率谱
%       'fs'             抽样频率，与 data 同时给出
%       'spectra'        是否输出功率谱，缺省 false
%       'geometry'       几何段，给出时另绘空间振型图
%       'figure_dir'     振型图与相关图的保存目录
%       'peaks'          各频带的谱峰频率
%       'excluded'       各频带是否含被剔除的子区间
%
%   返回结构体字段：bands、modes、separation、joint_covariance（不含矩阵本身）、
%   joint_covariance_matrix、shape_uncertainty，以及 spectra 与 spatial（未计算
%   时为 null）。
%
%   联合协方差把各频带的块拼成分块对角矩阵：不同频带的频点互不重叠，似然可加
%   而海森矩阵分块对角；同一频带内各阶由联合似然同时确定，相关性保留在块内。
%   密度判据必须在同一频带内比较，故区分度表只对同频带的阶对给出。

opt = struct('channel_index', [], 'data', [], 'fs', [], 'spectra', false, ...
    'geometry', [], 'figure_dir', [], 'peaks', [], 'excluded', []);
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

f = double(f(:).');
z = double(z(:).');
nOrder = numel(f);
nch = size(phi, 1);
nb = size(bands, 1);

channelIndex = opt.channel_index;
if isempty(channelIndex)
    channelIndex = 0:nch - 1;
end
channelIndex = double(channelIndex(:).');

% 逐阶的频带号与各频带首阶的全局序号
bandOfOrder = zeros(1, nOrder);
offsetOfBand = zeros(1, nb);
covPhi = cell(1, nOrder);
for b = 1:numel(blocks)
    blk = blocks{b};
    bi = double(blk.band_index);
    off = 0;
    if isfield(blk, 'offset') && ~isempty(blk.offset)
        off = double(blk.offset);
    end
    if bi >= 0 && bi < nb
        offsetOfBand(bi + 1) = off;
    end
    if isfield(blk, 'cov_phi') && ~isempty(blk.cov_phi)
        for k = 1:numel(blk.cov_phi)
            order = off + k;
            if order >= 1 && order <= nOrder
                covPhi{order} = double(blk.cov_phi{k});
            end
        end
    end
    for k = 1:double(blk.m)
        order = off + k;
        if order >= 1 && order <= nOrder
            bandOfOrder(order) = bi;
        end
    end
end

bandTable = bayoma.result_band_table(bands, f0List, nModsPerBand, ...
    opt.peaks, opt.excluded);
modeMap = bayoma.result_mode_map(f, bandOfOrder, offsetOfBand, bands, f0List);

joint = bayoma.result_joint_cov(blocks);
% 频率参数在联合协方差中的位置，用于给出频率估计之间的相关系数。
% joint.index{i}.index 自 0 起计（与结果文件一致），这里加 1 换成 MATLAB 下标。
fPositions = [];
for i = 1:numel(joint.index)
    if strcmp(joint.index{i}.kind, 'f')
        fPositions(end + 1) = joint.index{i}.index + 1; %#ok<AGROW>
    end
end

separation = bayoma.result_separation(f, z, phi, bandOfOrder, ...
    joint.joint, fPositions);
shapeUnc = bayoma.result_shape_uncertainty(covPhi, phi, channelIndex);

jrec = struct();
jrec.index = joint.index;
jrec.size = joint.size;
jrec.structure = joint.structure;
jrec.note = joint.note;

out = struct();
out.bands = bandTable;
out.modes = modeMap;
out.separation = separation;
out.joint_covariance = jrec;
out.joint_covariance_matrix = joint.joint;
out.shape_uncertainty = shapeUnc;
out.spectra = bayoma.json_null();
out.spatial = bayoma.json_null();

if opt.spectra && ~isempty(opt.data) && ~isempty(opt.fs)
    out.spectra = bayoma.result_spectra(opt.data, opt.fs, bands, f0List);
end

if ~isempty(opt.geometry)
    out.spatial = bayoma.result_spatial(phi, opt.geometry, opt.figure_dir, 'mode');
end
end
