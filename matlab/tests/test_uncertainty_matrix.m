% 校核第 16 章表 16.2 的闭式 Fisher 信息矩阵与表 15.1 的边际定律一致。
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..')));

f = 1.0; z = 0.01; S = 1.0; Se = 1.0; kappa = 10.0; dur = 2400.0; nch = 3;

mat = bayoma.uncertainty_matrix(f, z, S, Se, kappa, dur, nch);
law = bayoma.uncertainty_law(f, z, S, Se, kappa, dur, nch);

fprintf('Fisher 矩阵特征值：%s\n', mat2str(sort(eig(mat.fisher)).', 6));
fprintf('对称性检查 max|J-J^T| = %.3e\n', max(max(abs(mat.fisher - mat.fisher.'))));
fprintf('相关矩阵：\n');
disp(mat.corr);

names = {'f', 'z', 'S', 'Se'};
fprintf('表 16.2 求逆 vs 表 15.1 零阶：\n');
for i = 1:4
    a = mat.coefv(i);
    b = law.(names{i});
    fprintf('  %-3s %.6e vs %.6e  比值 %.6f\n', names{i}, a, b, a / b);
end
fprintf('  phi %.6e vs %.6e  比值 %.6f\n', mat.phi, law.phi, mat.phi / law.phi);

% 表 16.4 的阻尼比与模态力谱解析式
at = atan(kappa); b_par = at + kappa / (kappa * kappa + 1.0);
nc = f * dur;
nf = 2.0 * kappa * z * f * dur;
var_z_table = z / (4.0 * nc * (b_par - 2.0 * at * at / kappa));
var_z_exact = 1.0 / (mat.fisher(2, 2) * (1.0 - mat.q_zeta_S ^ 2));
fprintf('阻尼比方差：精确 %.6e  表 16.4 %.6e  相对差 %.3e\n', ...
    var_z_exact, var_z_table, abs(var_z_exact - var_z_table) / var_z_table);

var_s_exact = 1.0 / (mat.fisher(3, 3) * (1.0 - mat.q_zeta_S ^ 2));
var_s_table = S * S / nf / (1.0 - 2.0 * at * at / (kappa * b_par));
fprintf('模态力谱方差：精确 %.6e  表 16.4 %.6e  相对差 %.3e\n', ...
    var_s_exact, var_s_table, abs(var_s_exact - var_s_table) / var_s_table);

% 不同 kappa 与 q
fprintf('\nk 依赖（T=2400 s）：\n');
for k = [3.0 5.0 10.0 20.0]
    m = bayoma.uncertainty_matrix(f, z, S, Se, k, dur, nch);
    fprintf('  k=%5.1f  f %.6e  z %.6e  S %.6e  Se %.6e\n', ...
        k, m.coefv(1), m.coefv(2), m.coefv(3), m.coefv(4));
end

fprintf('\nq 依赖（k=10）：\n');
for qq = [0 1 2]
    m = bayoma.uncertainty_matrix(f, z, S, Se, kappa, dur, nch, qq);
    fprintf('  q=%d  f %.6e  z %.6e  S %.6e  Se %.6e  phi %.6e\n', qq, ...
        m.coefv(1), m.coefv(2), m.coefv(3), m.coefv(4), m.phi);
end
