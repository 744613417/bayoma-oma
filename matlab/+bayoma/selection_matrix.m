function L = selection_matrix(dofs, nDof)
%SELECTION_MATRIX 构造测组的选择矩阵 L_r（式 14.24 之后的说明）。
%
%   L = bayoma.selection_matrix(DOFS, NDOF) 返回 n_r x NDOF 的矩阵，第 i 行
%   在第 DOFS(i) 个自由度为 1、其余为 0。该矩阵把全局振型 u 映射为测组的
%   局部振型 t_r = L_r u。
%
%   DOFS 为正整数向量，取值 1..NDOF。

dofs = dofs(:);
if any(dofs < 1) || any(dofs > nDof) || any(dofs ~= round(dofs))
    error('bayoma:selection_matrix:badDofs', '自由度编号必须是 1..%d 内的整数', nDof);
end
if numel(unique(dofs)) ~= numel(dofs)
    error('bayoma:selection_matrix:duplicateDofs', '同一测组内的自由度编号不得重复');
end

n = numel(dofs);
L = zeros(n, nDof);
for i = 1:n
    L(i, dofs(i)) = 1;
end
end
