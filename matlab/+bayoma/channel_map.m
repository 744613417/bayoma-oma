function Y = channel_map(base, basis)
%CHANNEL_MAP 把切空间坐标的线性映射矩阵映回通道空间。
%
%   Y = bayoma.channel_map(BASE, BASIS) 返回形状为 (nOut x nCol) 的实矩阵，
%   用于把切空间坐标 theta 的协方差块映射为通道空间振型的协方差矩阵。BASE
%   为切空间正交基：实振型时为 (n x n-1)，复振型时为 (m0 x 2*m0-2)；BASIS
%   为把子空间坐标映射到通道空间的正交基 B0，即 U = B0 * alpha，形状
%   (n x m0)。
%
%   BASIS 为空表示振型向量本身即通道空间向量，此时 BASE 直接作为映射矩阵
%   返回（实情形），或按实部、虚部展开后返回（复情形）。
%
%   复振型的切空间坐标按 [Re(dalpha); Im(dalpha)] 排列，经块对角基
%   blk = [[B0, 0]; [0, B0]] 映射到 [Re(du); Im(du)]。B0 为实基时，实部与
%   虚部的映射相互独立，故该块对角形式与课本第 13 章的子空间表示一致。

if isempty(basis)
    if isreal(base)
        Y = double(base);
        return;
    end
    % 无子空间基但形状为复向量：通道空间维数即形状长度
    n = size(base, 1);
    blk = eye(2 * n);
    Y = blk * [real(base); imag(base)];
    return;
end

b = double(basis);
if isreal(base) && isreal(b)
    % 实振型：切空间坐标在子空间内的扰动经 B0 线性映射到通道空间
    Y = b * double(base);
    return;
end

% 复振型：实部、虚部分别经 B0 映射，坐标按 [Re; Im] 堆叠
n = size(b, 1);
blk = [b, zeros(n, size(b, 2)); zeros(n, size(b, 2)), b];
Y = blk * [real(base); imag(base)];
end
