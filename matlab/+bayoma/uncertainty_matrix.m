function out = uncertainty_matrix(f, z, S, Se, kappa, duration, nch, q)
%UNCERTAINTY_MATRIX 第 16 章表 16.2 的闭式 Fisher 信息矩阵与参数间相关性。
%
%   OUT = bayoma.uncertainty_matrix(F, Z, S, SE, KAPPA, DURATION, NCH, Q)
%
%   第 15 章表 15.1 只给出各参数的边际变异系数，且已把阻尼比与模态力谱之间的
%   相关性解析地吸收进 B_zeta。表 16.2 直接给出长数据渐近下 NLLF 二阶导矩阵
%   （即 Fisher 信息矩阵）的闭式，其逆同时给出边际方差与参数间相关系数。
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
%   输出结构体
%       names    参数名元胞 {'f','z','S','Se'}
%       fisher   4×4 Fisher 信息矩阵，顺序 (f, z, S, Se)
%       cov      4×4 后验协方差矩阵（物理单位）
%       corr     4×4 相关矩阵
%       coefv    4×1 变异系数，顺序同 names
%       phi      振型变异系数 sigma_phi = sqrt(trace(C_phi))
%       q_zeta_S 阻尼比与模态力谱的敏感度相关系数
%
%   公式依据式 (16.7)、表 16.1 与表 16.2；符号与表 16.4 一致。
%
%   参见 bayoma.uncertainty_law、bayoma.select_bands。

if nargin < 8
    q = 0;
end

f = double(f);
z = double(z);
S = double(S);
Se = double(Se);
kappa = double(kappa);

nf = 2.0 * kappa * z * f * double(duration);
nc = nf / (2.0 * kappa * z);
at = atan(kappa);
k = kappa;
k2 = k * k;

mat = zeros(4, 4);
mat(1, 1) = 4.0 * nc / (f * f * z) * (at - k / (k2 + 1.0));
mat(1, 2) = 4.0 * nc / f * (at + (2.0 * k ^ 3 + k) / (k2 + 1.0) ^ 2);
mat(1, 3) = -2.0 * nc * z / (f * S) * (2.0 * at + k ^ 3 / (k2 + 1.0));
mat(1, 4) = -8.0 * nc * z ^ 3 / (3.0 * f * S) * (7.0 * k ^ 3 + 6.0 * k);
mat(2, 2) = 4.0 * nc / z * (at + k / (k2 + 1.0));
mat(2, 3) = -4.0 * nc / S * at;
mat(2, 4) = -16.0 * nc * z * z * k / S;
mat(3, 3) = nf / (S * S);
mat(3, 4) = 8.0 * nc * z ^ 3 / (3.0 * S * S) * (k ^ 3 + 3.0 * k);
mat(4, 4) = (nch - 1) * nf / (Se * Se);
info = mat + triu(mat, 1).';

cov = inv(info);
sd = sqrt(diag(cov));
corr = cov ./ (sd * sd.');
base = [f; z; S; Se];

l_zs = info(2, 3);
q_zeta_S = l_zs / sqrt(info(2, 2) * info(3, 3));

nu = (2.0 * pi * f) ^ (2.0 * q) * Se / S;
sigma_phi = sqrt(nu * z * (nch - 1) / (nc * at));

out = struct();
out.names = {'f', 'z', 'S', 'Se'};
out.fisher = info;
out.cov = cov;
out.corr = corr;
out.coefv = sd ./ base;
out.phi = sigma_phi;
out.q_zeta_S = q_zeta_S;
end
