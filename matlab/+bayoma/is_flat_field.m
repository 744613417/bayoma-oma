function tf = is_flat_field(name)
%IS_FLAT_FIELD 判断字段是否按 JSON 契约写成一维数组。
%   `name` 可以是完整路径（如 ``coefv.f``），判定规则：
%
%   - ``coefv`` 下的各项在 Python 侧一律是一维 ndarray，全部按一维输出；
%   - 顶层的量值向量见 FLAT 集合（f、z、Sii、Se、rms、sn、S_off、v）；
%   - 结果分组里的一维字段见 FLAT2 集合：振型分量、参数下标、通道下标、
%     频带内的初值与频率、谱的频率轴、几何连接、密模态对等；
%   - ``groups`` 与 ``setups[].channels`` 的每一项是通道编号的一维数组；
%   - 复振型的实部、虚部（``shape.re``、``shape.im``）按一维输出，与
%     numpy 中一维复数数组的 tolist() 结果一致。
%
%   二维字段（振型矩阵 phi、频带 bands、协方差矩阵 cmat 与 joint_covariance_matrix、
%   模态力谱 S、时间统计 time_req、相干矩阵、谱的 sv 与 psd_diag、振型协方差 cov）
%   不在此列，仍按矩阵展开成嵌套数组。

if startsWith(name, 'coefv.')
    tf = true;
    return;
end
% 复振型的实部与虚部是通道方向的一维数组，需与协方差矩阵的 re/im 区分开
if endsWith(name, 'shape.re') || endsWith(name, 'shape.im')
    tf = true;
    return;
end
last = name;
dot = find(name == '.', 1, 'last');
if ~isempty(dot)
    last = name(dot + 1:end);
end
flat = {'f', 'z', 'Sii', 'Se', 'rms', 'sn', 'S_off', 'v', 'groups'};
% 结果分组中的一维字段
flat2 = {'modes', 'orders', 'n_tan', 'half_power_bw', 'dense_pairs', ...
    'channel_index', 'std_per_channel', 'ci95_halfwidth_per_channel', ...
    'freqs', 'connections', 'channels', 'shape', 'f0'};
tf = any(strcmp(flat, last)) || any(strcmp(flat2, last));
end
