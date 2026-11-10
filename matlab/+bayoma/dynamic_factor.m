function Dk = dynamic_factor(f, z, fk, q)
%DYNAMIC_FACTOR 计算式 (12.4) 的动态放大因子 D_k。
%
%   DK = bayoma.dynamic_factor(F, Z, FK, Q) 返回与频率坐标 FK 同形的动态因子
%       D_k = (2*pi*fk)^(-2q) / [ (1 - beta^2)^2 + (2*z*beta)^2 ]
%   其中 beta = F/FK。Q = 0 对应加速度，1 为速度，2 为位移。
%
%   输入 F、Z 为标量模态参数，FK 为频率向量。

if nargin < 4 || isempty(q)
    q = 0;
end

beta = f ./ fk;
den = (1 - beta.^2).^2 + (2 * z * beta).^2;
Dk = (2 * pi * fk).^(-2 * q) ./ den;
end
