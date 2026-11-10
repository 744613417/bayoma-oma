function [Lc, info] = condensation_matrix(dofsPerSetup, nDof)
%CONDENSATION_MATRIX 参考自由度凝聚矩阵（课本式 14.17）
%   [Lc, info] = bayoma.condensation_matrix(dofsPerSetup, nDof)
%
%   输入
%     dofsPerSetup : 元胞数组，每个测组的测点自由度编号（1 起计）
%     nDof         : 结构全部自由度个数
%   输出
%     Lc    : nDof x nCond，满足 Lc' * Lc = I
%     info  : 结构体，字段 reference（参考自由度）、nonReference（非参考自由度
%             按测组顺序拼接）、nCondensed（凝聚参数个数）
%
%   全局振型按 u = [uR; u01; ...; u0q] 排列：uR 为被两个以上通道测到的
%   自由度分量，u0r 为第 r 个测组独有的自由度分量。凝聚参数
%   uc = [uR; a1; ...; aq]，其中 ar 复制第 r 个测组的非参考分量，
%   从而 u = Lc * uc（式 14.17）。

nDof = double(nDof);
nSetup = numel(dofsPerSetup);
dofs = cell(1, nSetup);
counts = zeros(1, nDof);
for r = 1:nSetup
    dofs{r} = double(dofsPerSetup{r}(:).');
    if any(dofs{r} < 1) || any(dofs{r} > nDof) || any(dofs{r} ~= round(dofs{r}))
        error('bayoma:badInput', '测点自由度编号须为 1 到 %d 的整数', nDof);
    end
    for k = 1:numel(dofs{r})
        counts(dofs{r}(k)) = counts(dofs{r}(k)) + 1;
    end
end
reference = find(counts > 1);
isRef = false(1, nDof);
isRef(reference) = true;

columns = zeros(nDof, 0);
for k = 1:numel(reference)
    col = zeros(nDof, 1);
    col(reference(k)) = 1.0;
    columns = [columns, col]; %#ok<AGROW>
end

nonReference = zeros(1, 0);
for r = 1:nSetup
    extra = dofs{r}(~isRef(dofs{r}));
    nonReference = [nonReference, extra]; %#ok<AGROW>
    for k = 1:numel(extra)
        col = zeros(nDof, 1);
        col(extra(k)) = 1.0;
        columns = [columns, col]; %#ok<AGROW>
    end
end

Lc = columns;
info = struct('reference', reference, 'nonReference', nonReference, ...
              'nCondensed', size(Lc, 2));
end
