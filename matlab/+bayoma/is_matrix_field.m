function tf = is_matrix_field(name)
%IS_MATRIX_FIELD 判断字段是否必须按二维数组写成嵌套数组。
%
%   判据只看字段名，用途是纠正「1x1 矩阵被写成裸数字」这一种情形：复数矩阵
%   按 {"re": [...], "im": [...]} 拆分后，若矩阵只有一个元素，MATLAB 的标量
%   分支会把它写成裸数字，与 JSON 契约里「二维字段写成数组的数组」不符。
%   Python 侧用 numpy 的 tolist() 写同一份契约，1x1 矩阵得到的是 [[x]]。
%
%   复振型的实部与虚部（shape.re、shape.im）不在此列：它们按通道方向的一维
%   数组输出，由 is_flat_field 先行判定。
%
%   参见 bayoma.to_json_tree、bayoma.is_flat_field、bayoma.complex_split。

tf = endsWith(name, '.re') || endsWith(name, '.im');
end
