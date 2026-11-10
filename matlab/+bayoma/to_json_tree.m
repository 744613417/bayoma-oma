function value = to_json_tree(x, field)
%TO_JSON_TREE 把结果结构体转成 encode_json 需要的「形状显式」树。
%   MATLAB 的数值数组无法区分标量与长度 1 的数组，若直接交给 jsonencode，
%   长度 1 的数组会被写成 JSON 数字，与 Python 侧的契约（同一字段一律数组）
%   不一致。本函数只在写文件时展开形状，`bayoma.io_json` 的返回值仍是普通
%   数值数组，调用方不受影响。
%
%   形状规则与 numpy 的 ``ndarray.tolist()`` 对齐，按字段分四类：
%
%   - 标量字段（is_scalar_field）：必须为标量，写成 JSON 数字，如 nch、fs；
%   - 一维字段（is_flat_field）：不论行向量还是列向量都写成 [a, b]，
%     如 f、z、Sii、Se、rms、sn、coefv 的各项、groups 的每一项；
%   - 长度 1 且不属于上述两类的数值：写成 JSON 数字。结果契约里所有需要
%     保持数组形状的长度 1 字段都已列入一维字段，因此这一条不会破坏契约，
%     同时使 index、order、band_index 等整数按数字写出；
%   - 其余数值按二维处理：m x n 矩阵写成 [[a, b], [c, d]]，因此 1x1 写成
%     [[x]]、列向量写成 [[a], [b]]，与 numpy 中形状 (n, 1) 的 tolist() 一致。
%
%   结构体数组（如频带表、参数索引）用元胞存放，编码为对象数组；允许为
%   null 的字段用 bayoma.json_null 占位，编码为 JSON null。
%
%   `field` 是当前值所属的字段名，逐层向下传递，用于上述判定。

if nargin < 2
    field = '';
end

if isstruct(x)
    if ~isscalar(x)
        error('bayoma:badJson', '结构体必须是标量结构体');
    end
    if isfield(x, 'nul')
        value = x;
        return;
    end
    names = fieldnames(x);
    value = struct();
    for i = 1:numel(names)
        % 逐字段赋值，避免 struct(...) 把元胞字段解开一层花括号；
        % 字段名按路径拼接，使 is_flat_field 能识别 coefv.f 这类嵌套字段
        if isempty(field)
            path = names{i};
        else
            path = [field '.' names{i}];
        end
        value.(names{i}) = bayoma.to_json_tree(x.(names{i}), path);
    end
elseif iscell(x)
    value = cell(size(x));
    for i = 1:numel(x)
        value{i} = bayoma.to_json_tree(x{i}, field);
    end
elseif ischar(x) || isstring(x)
    value = char(x);
elseif islogical(x)
    if isscalar(x)
        value = x;
    else
        value = bayoma.to_json_tree(double(x), field);
    end
elseif isnumeric(x)
    if ~isreal(x)
        error('bayoma:badJson', '复数必须先转成 re/im 结构再编码');
    end
    if bayoma.is_scalar_field(field)
        if ~isscalar(x)
            error('bayoma:badJson', '标量字段 %s 不是标量', field);
        end
        value = double(x);
        return;
    end
    if isempty(x)
        value = {};
    elseif bayoma.is_flat_field(field)
        value = num2cell(double(x(:).'));
    elseif bayoma.is_matrix_field(field)
        % 二维字段即使只有一个元素也要写成数组的数组，与 numpy 的 tolist() 一致
        rows = cell(1, size(x, 1));
        for i = 1:size(x, 1)
            rows{i} = num2cell(double(x(i, :)));
        end
        value = rows;
    elseif isscalar(x)
        value = double(x);
    elseif isvector(x)
        % 未列入一维字段的向量（例如 time_req 的每一列）按行展开
        value = {num2cell(double(x(:).'))};
    else
        rows = cell(1, size(x, 1));
        for i = 1:size(x, 1)
            rows{i} = num2cell(double(x(i, :)));
        end
        value = rows;
    end
else
    error('bayoma:badJson', '无法编码的类型：%s', class(x));
end
end
