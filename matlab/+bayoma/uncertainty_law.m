function out = uncertainty_law(f, z, S, Se, kappa, duration, nch, q)
%UNCERTAINTY_LAW 用第 15 章的不确定度定律预测各参数后验变异系数。
%
%   OUT = bayoma.uncertainty_law(F, Z, S, SE, KAPPA, DURATION, NCH, Q)
%   返回结构体，字段 f、z、S、Se、phi 为各参数变异系数的预测值。
%
%   该预测只需模态参数与记录长度，无需重新计算 Hessian，因此可用于比较不同
%   选带方案与试验时长的预期精度。
%
%   输入
%       f        固有频率（Hz）
%       z        阻尼比
%       S        模态力谱密度
%       Se       预测误差谱密度
%       kappa    带宽因子，频带取 f*(1 ± kappa*z)
%       duration 记录时长（s）
%       nch      测点数
%       q        数据类型指数，0 加速度、1 速度、2 位移
%
%   公式依据式 (15.2)–(15.4) 与表 15.1。频率的零阶定律分子为阻尼比，不是频率。
%
%   参见 bayoma.select_bands、bayoma.bayoma。

if nargin < 8
    q = 0;
end

f = double(f);
z = double(z);
nf = 2.0 * kappa * z * f * double(duration);
nc = nf / (2.0 * kappa * z);
if nf <= 0.0 || nc <= 0.0
    out = struct('f', inf, 'z', inf, 'S', inf, 'Se', inf, 'phi', inf);
    return;
end

mu = double(Se) / max(double(S), 1.0e-300) * (2.0 * pi * f) ^ (2.0 * q);
if mu > 0.0
    c = 1.0 / (4.0 * mu * z * z);
else
    c = inf;
end

at = atan(kappa);
par = at + kappa / (kappa * kappa + 1.0);

bracket_f = at - kappa / (kappa * kappa + 1.0);
b_f = 2.0 / pi * bracket_f;
a_f = 4.0 * (kappa - at) / bracket_f;

b_z = 2.0 / pi * (at + kappa / (kappa * kappa + 1.0) - 2.0 * at * at / kappa);
a_z = 4.0 * (kappa * kappa + 1.0) * (3.0 * at - 3.0 * kappa + kappa * kappa * at) * at ...
    / (3.0 * ((kappa * kappa + 1.0) * (kappa - 2.0 * at) * at + kappa * kappa));

b_s = 1.0 - 2.0 / kappa * at * at / par;
a_s = 2.0 + (2.0 / 3.0) * kappa * kappa ...
    - at * at / (2.0 * at * at - par * kappa) ...
    * (8.0 * at / par - 8.0 * kappa / at + (4.0 / 3.0) * kappa * kappa + 4.0);

b_u = 2.0 / pi * at;

out = struct();
out.f = sqrt(max(scale_law(z / (2.0 * pi * nc * b_f), a_f, c), 0.0));
out.z = sqrt(max(scale_law(1.0 / (2.0 * pi * z * nc * b_z), a_z, c), 0.0));
out.S = sqrt(max(scale_law(1.0 / (nf * b_s), a_s, c), 0.0));
out.Se = sqrt(max(1.0 / ((nch - 1) * nf), 0.0));
out.phi = sqrt(max(2.0 * (nch - 1) * mu * z / (pi * nc * b_u), 0.0));
end

% ------------------------------------------------------------------
function value = scale_law(zeroth, alpha, c)
%SCALE_LAW 施加一阶修正 ε² ≈ ε²₀*(1 + α/c)。
if ~isfinite(c) || c <= 0.0
    value = zeroth;
else
    value = zeroth * (1.0 + alpha / c);
end
end
