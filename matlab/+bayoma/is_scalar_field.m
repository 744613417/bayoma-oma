function tf = is_scalar_field(name)
%IS_SCALAR_FIELD 判断字段是否按 JSON 契约写成数字而非数组。
%   `name` 可以是完整路径（如 ``input.duration``），只有最后一段参与判定。
%   nch、fs、n_modes 以及 input 段的数量字段在 schema/result.schema.json 里
%   都是数字。

last = name;
dot = find(name == '.', 1, 'last');
if ~isempty(dot)
    last = name(dot + 1:end);
end
scalars = {'nch', 'fs', 'n_modes', 'nt', 'duration', 'n_setup', 'n_group', ...
    'elapsed', 'n_iter', 'converged'};
tf = any(strcmp(scalars, last));
end
