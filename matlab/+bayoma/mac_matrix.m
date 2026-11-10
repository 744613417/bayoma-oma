function mac = mac_matrix(phiA, phiB)
%MAC_MATRIX 计算两组振型之间的模态置信度矩阵。
%
%   MAC = bayoma.mac_matrix(PHIA, PHIB) 返回 MAC 矩阵，元素 (i,j) 为
%   PHIA 第 i 列与 PHIB 第 j 列的模态置信度
%
%       MAC(i,j) = |phiA(:,i)' * phiB(:,j)|^2 /
%                  ( (phiA(:,i)' * phiA(:,i)) * (phiB(:,j)' * phiB(:,j)) )
%
%   PHIA 与 PHIB 的列数可以不同，行数（测点数）必须相同。MAC 为 1 表示两列
%   完全相关（同一阶模态），为 0 表示正交。
%
%   参见 bayoma.bayoma、bayoma.single_mode。

a = double(phiA);
b = double(phiB);
if size(a, 1) ~= size(b, 1)
    error('bayoma:badShape', '两组振型的测点数必须相同');
end

na = size(a, 2);
nb = size(b, 2);
mac = zeros(na, nb);

na2 = sum(conj(a) .* a, 1).';        % na x 1
nb2 = sum(conj(b) .* b, 1);          % 1 x nb
cross = abs(a' * b) .^ 2;            % na x nb
denom = na2 * nb2;
withDenom = denom > 0;
mac(withDenom) = cross(withDenom) ./ denom(withDenom);
end
