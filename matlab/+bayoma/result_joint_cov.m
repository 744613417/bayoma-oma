function out = result_joint_cov(blocks)
%RESULT_JOINT_COV 把各频带的后验协方差块拼成联合协方差矩阵，并给出参数索引。
%
%   OUT = bayoma.result_joint_cov(BLOCKS) 中 BLOCKS 为元胞数组，每项一个频带，
%   字段为：
%       cov          该频带的参数协方差矩阵
%       band_index   频带序号（自 0 起计）
%       offset       该频带首阶在全局的序号
%       m            该频带的模态阶数
%       n_off        模态力谱非对角元的个数 m(m-1)/2
%       sizes        各阶振型切空间坐标的个数
%       index        可选，显式给出该块的参数描述（元胞，每项含 name 与 kind）
%       channel_index 可选，通道编号
%
%   返回结构体字段：joint（联合协方差矩阵）、index（参数描述元胞，每项另带
%   全局下标 index、频带号 band_index、阶次 order 或阶次对 orders、通道号
%   channel）、size（矩阵维数）、structure（'block_diagonal'）、note。
%
%   分块对角的依据：不同频带使用互不重叠的频点，似然可加，海森矩阵关于频带
%   分块对角；同一频带内的模态由联合似然同时确定，其相关性保留在块内。

nb = numel(blocks);
total = 0;
for b = 1:nb
    total = total + size(blocks{b}.cov, 1);
end

joint = zeros(total);
index = {};
pos = 0;
for b = 1:nb
    blk = blocks{b};
    cov = double(blk.cov);
    k = size(cov, 1);
    joint(pos + 1:pos + k, pos + 1:pos + k) = 0.5 * (cov + cov.');
    off = 0;
    if isfield(blk, 'offset') && ~isempty(blk.offset)
        off = double(blk.offset);
    end
    entries = prepare_entries(blk);
    if numel(entries) ~= k
        error('bayoma:badIndex', '频带 %d 的参数描述 %d 项与协方差维数 %d 不符', ...
            blk.band_index, numel(entries), k);
    end
    for local = 1:k
        rec = entries{local};
        rec.index = pos + local - 1;
        if isfield(blk, 'band_index') && ~isempty(blk.band_index)
            rec.band_index = double(blk.band_index);
        else
            rec.band_index = 0;
        end
        % 模态阶次换算到全局：频带内第 k 阶对应全局第 offset+k 阶
        if isfield(rec, 'mode') && ~isempty(rec.mode) && ~isnan(rec.mode)
            rec.order = off + double(rec.mode);
        else
            rec.order = bayoma.json_null();
        end
        if isfield(rec, 'modes') && ~isempty(rec.modes)
            rec.orders = off + double(rec.modes);
        end
        % 振型分量的通道号按通道编号映射，通道编号未必等于局部下标
        if isfield(rec, 'channel') && ~isempty(rec.channel) && ...
                isfield(blk, 'channel_index') && ~isempty(blk.channel_index)
            rec.channel = double(blk.channel_index(double(rec.channel) + 1));
        end
        index{end + 1} = rec; %#ok<AGROW>
    end
    pos = pos + k;
end

out = struct();
out.joint = joint;
out.index = index;
out.size = total;
out.structure = 'block_diagonal';
out.note = ['不同频带使用互不重叠的频点，似然可加，Hessian 关于频带分块对角；' ...
    '同一频带内的模态由联合似然同时确定，其相关性保留在块内。'];
end

% ----------------------------------------------------------------------
function entries = prepare_entries(blk)
%PREPARE_ENTRIES 取一个协方差块的参数描述，优先使用块自带的显式索引。
if isfield(blk, 'index') && ~isempty(blk.index)
    entries = blk.index;
    return;
end
m = double(blk.m);
nOff = double(blk.n_off);
sizes = double(blk.sizes);
entries = [bayoma.result_index('spectral', m, nOff), ...
    bayoma.result_index('shape', m, sizes)];
end
