function value = json_null()
%JSON_NULL 结果树中的 JSON null 占位值。
%
%   结果文件里有一批字段允许为 null（例如参数描述中不适用于该参数的 mode、
%   未生成的空间可视化 spatial、未计算的谱 spectra）。MATLAB 的空数组在
%   编码后是 []，与 null 不等价，故用一个带标记的标量结构体表示：编码时写
%   成 null，其余场合不应参与数值运算。
%
%   标记字段名取 `nul`，不能取带下划线的名字：MATLAB 的 struct 字段名必须
%   是合法的标识符，用 `__null__` 会在创建结构体时直接报错。
%
%   参见 bayoma.to_json_tree、bayoma.encode_json。

value = struct('nul', true);
end
