function text = encode_json(value)
%ENCODE_JSON 把结果结构体编成 JSON 文本。
%   不使用 jsonencode，因为它会把 1x1 数组写成标量、把行向量与矩阵的形状
%   混同，导致结果文件与 Python 侧的契约不一致（例如 bands 写成 [0.9,1.1]
%   而不是 [[0.9,1.1]]，Sii 写成标量而不是长度 1 的数组）。这里的编码规则
%   与 Python 的 json.dumps 一一对应：
%
%   - struct        -> 对象，字段名保持原样；
%   - cell 数组     -> 数组，元素按顺序编码；
%   - 空数组        -> []；
%   - 标量          -> 数值（结果契约里所有标量字段都存成 1x1 数组，
%                      因此这里只有在确实要给标量时才用标量）；
%   - 行/列向量     -> 一维数组；
%   - 二维矩阵      -> 数组的数组，每个内层数组是一行；
%   - 字符/逻辑     -> 字符串/布尔。
%
%   复数会报错，调用方须先转成 {re, im} 结构，避免静默丢虚部。

if isstruct(value)
    if isscalar(value) && isfield(value, 'nul')
        % bayoma.json_null 的占位值：契约里允许为 null 的字段
        text = 'null';
        return;
    end
    text = encode_struct(value);
elseif iscell(value)
    text = encode_cell(value);
elseif ischar(value)
    text = encode_string(value);
elseif isstring(value)
    text = encode_string(char(value));
elseif islogical(value)
    if isscalar(value)
        if value
            text = 'true';
        else
            text = 'false';
        end
    else
        text = encode_numeric(double(value));
    end
elseif isnumeric(value)
    if ~isreal(value)
        error('bayoma:badJson', '复数必须先转成 re/im 结构再编码');
    end
    text = encode_numeric(value);
else
    error('bayoma:badJson', '无法编码的类型：%s', class(value));
end
end

% ------------------------------------------------------------------
function text = encode_struct(value)
if ~isscalar(value)
    error('bayoma:badJson', '结果结构体必须是标量结构体');
end
names = fieldnames(value);
parts = cell(1, numel(names));
for i = 1:numel(names)
    parts{i} = sprintf('%s:%s', encode_string(names{i}), bayoma.encode_json(value.(names{i})));
end
text = ['{', strjoin(parts, ','), '}'];
end

% ------------------------------------------------------------------
function text = encode_cell(value)
parts = cell(1, numel(value));
for i = 1:numel(value)
    parts{i} = bayoma.encode_json(value{i});
end
text = ['[', strjoin(parts, ','), ']'];
end

% ------------------------------------------------------------------
function text = encode_numeric(value)
if isempty(value)
    text = '[]';
    return;
end
if isscalar(value)
    text = encode_number(double(value));
    return;
end
if isvector(value)
    items = arrayfun(@(x) encode_number(double(x)), value(:).', ...
        'UniformOutput', false);
    text = ['[', strjoin(items, ','), ']'];
    return;
end
rows = cell(1, size(value, 1));
for i = 1:size(value, 1)
    rows{i} = encode_numeric(value(i, :));
end
text = ['[', strjoin(rows, ','), ']'];
end

% ------------------------------------------------------------------
function text = encode_number(x)
if ~isfinite(x)
    error('bayoma:badJson', '结果中出现非有限数值：%g', x);
end
if x == round(x) && abs(x) < 1e15
    text = sprintf('%d', x);
else
    text = sprintf('%.17g', x);
end
end

% ------------------------------------------------------------------
function text = encode_string(value)
escaped = strrep(value, '\', '\\');
escaped = strrep(escaped, '"', '\"');
escaped = strrep(escaped, sprintf('\n'), '\n');
escaped = strrep(escaped, sprintf('\r'), '\r');
escaped = strrep(escaped, sprintf('\t'), '\t');
text = ['"', escaped, '"'];
end
