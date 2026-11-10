function [Dk, der] = dynamic_factor_derivs(f, z, fk, q)
%DYNAMIC_FACTOR_DERIVS 动态因子 D_k 对 f 与 z 的一阶、二阶导数（表 12.2）。
%
%   [DK, DER] = bayoma.dynamic_factor_derivs(F, Z, FK, Q) 返回 D_k 及其导数。
%   DER 为结构体，字段 D、f、z、ff、fz、zz 均为与 FK 同形的向量，其中 D 为
%   D_k 本身，其余为对 F、Z 的一阶与二阶偏导。
%
%   表 12.2 给出的是 D_k^{-1} 的导数形式；此处直接对
%       D_k = om / re,  om = (2*pi*fk)^(-2q),  re = (1-beta^2)^2 + (2*z*beta)^2
%   求导，两式数学等价，实现后以有限差分交叉校核。该形式避免了对 D_k 直接
%   求导时的除零，且对一般参数值成立。

if nargin < 4 || isempty(q)
    q = 0;
end

beta = f ./ fk;
b2 = beta.^2;
om = (2 * pi * fk).^(-2 * q);
re = (1 - b2).^2 + (2 * z * beta).^2;
Dk = om ./ re;

% re 对 f、z 的一阶导数
re_f = 4 * beta .* (b2 - 1 + 2 * z^2) / f;
re_z = 8 * z * b2;

% re 对 f、z 的二阶导数
re_ff = 4 * (3 * b2 - 1 + 2 * z^2) / (f * f);
re_fz = 16 * z * beta / f;
re_zz = 8 * b2;

der.D = Dk;
der.f = -om .* re_f ./ re.^2;
der.z = -om .* re_z ./ re.^2;
der.ff = 2 * om .* re_f.^2 ./ re.^3 - om .* re_ff ./ re.^2;
der.fz = 2 * om .* re_f .* re_z ./ re.^3 - om .* re_fz ./ re.^2;
der.zz = 2 * om .* re_z.^2 ./ re.^3 - om .* re_zz ./ re.^2;
end
