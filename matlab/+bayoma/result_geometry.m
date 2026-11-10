function geo = result_geometry(geometry, nch, channelIndex)
%RESULT_GEOMETRY 校验并规范化通道几何：位置、方向、连接。
%
%   GEO = bayoma.result_geometry(GEOMETRY, NCH, CHANNELINDEX) 检查 GEOMETRY 的通道
%   数与数据通道数一致，把每个通道的位置与方向整理成 nch x 3 的矩阵，方向归一化，
%   并核对连接的通道下标是否越界。
%
%   GEOMETRY 为描述文件里的几何段，字段 channels 与 connections；每个通道含
%   position（长度 3，单位与结构尺寸一致）、direction（长度 3，测点方向）、
%   name（可选）。connections 为通道对，用于把测点连成结构简图。
%
%   返回结构体字段：channels（各通道的记录元胞）、connections（通道对元胞）、
%   positions（nch x 3）、directions（nch x 3）。
%
%   方向向量的模不为 1 时按模归一，因此描述文件可以直接给形心方向而不必先归一。

if nargin < 3 || isempty(channelIndex)
    channelIndex = 0:nch - 1;
end

chans = get_channels(geometry);
if isempty(chans)
    error('bayoma:badGeometry', 'geometry.channels 不能为空');
end
if numel(chans) ~= nch
    error('bayoma:badGeometry', 'geometry.channels 的长度应为 %d，收到 %d', ...
        nch, numel(chans));
end

positions = zeros(nch, 3);
directions = zeros(nch, 3);
names = cell(1, nch);
for i = 1:nch
    ch = chans{i};
    pos = get_vec3(ch, 'position', [0, 0, 0]);
    dirv = get_vec3(ch, 'direction', [0, 0, 1]);
    nrm = norm(dirv);
    if nrm <= 0
        error('bayoma:badGeometry', '通道 %d 的 direction 不能为零向量', i - 1);
    end
    positions(i, :) = pos;
    directions(i, :) = dirv / nrm;
    if isfield(ch, 'name') && ~isempty(ch.name)
        names{i} = char(ch.name);
    else
        names{i} = sprintf('ch%d', i - 1);
    end
end

rawConns = get_connections(geometry);
conns = cell(1, numel(rawConns));
for k = 1:numel(rawConns)
    pair = double(rawConns{k}(:).');
    if numel(pair) ~= 2
        error('bayoma:badGeometry', 'connections 的每项必须是两个通道下标');
    end
    if any(pair < 0) || any(pair >= nch)
        error('bayoma:badGeometry', '连接 (%d, %d) 的通道下标越界', pair(1), pair(2));
    end
    conns{k} = [pair(1), pair(2)];
end

recs = cell(1, nch);
for i = 1:nch
    r = struct();
    r.index = double(channelIndex(i));
    r.local_index = i - 1;
    r.name = names{i};
    r.position = positions(i, :);
    r.direction = directions(i, :);
    recs{i} = r;
end

geo = struct();
geo.channels = recs;
geo.connections = conns;
geo.positions = positions;
geo.directions = directions;
end

% ----------------------------------------------------------------------
function chans = get_channels(geometry)
%GET_CHANNELS 取出通道记录，兼容结构体数组与元胞两种写法。
chans = {};
if ~isstruct(geometry) || ~isfield(geometry, 'channels') || isempty(geometry.channels)
    return;
end
raw = geometry.channels;
if iscell(raw)
    chans = raw(:).';
elseif isstruct(raw)
    chans = cell(1, numel(raw));
    for i = 1:numel(raw)
        chans{i} = raw(i);
    end
end
end

% ----------------------------------------------------------------------
function conns = get_connections(geometry)
%GET_CONNECTIONS 取出通道对，兼容数值矩阵、元胞与结构体数组。
conns = {};
if ~isstruct(geometry) || ~isfield(geometry, 'connections') || isempty(geometry.connections)
    return;
end
raw = geometry.connections;
if iscell(raw)
    conns = raw(:).';
elseif isstruct(raw)
    conns = cell(1, numel(raw));
    for i = 1:numel(raw)
        r = raw(i);
        if isfield(r, 'channels')
            conns{i} = double(r.channels);
        elseif isfield(r, 'pair')
            conns{i} = double(r.pair);
        else
            error('bayoma:badGeometry', 'connections 的每项需给出通道对');
        end
    end
else
    mat = double(raw);
    if isvector(mat)
        conns = {mat(:).'};
    else
        conns = cell(1, size(mat, 1));
        for i = 1:size(mat, 1)
            conns{i} = mat(i, :);
        end
    end
end
end

% ----------------------------------------------------------------------
function v = get_vec3(s, name, default)
%GET_VEC3 取长度 3 的向量字段，缺省用 default，长度不符时报错。
if isfield(s, name) && ~isempty(s.(name))
    v = double(s.(name)(:).');
else
    v = double(default);
end
if numel(v) ~= 3
    error('bayoma:badGeometry', '%s 必须是长度 3 的数组', name);
end
end
