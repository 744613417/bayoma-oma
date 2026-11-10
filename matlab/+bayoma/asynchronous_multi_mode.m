function out = asynchronous_multi_mode(freqs, fft, groups, nModes, opts)
%ASYNCHRONOUS_MULTI_MODE 异步数据的多阶模态贝叶斯识别（课本 10.4 节，式 (10.44)–(10.49)）。
%
%   OUT = bayoma.asynchronous_multi_mode(FREQS, FFT, GROUPS, NMODES)
%   OUT = bayoma.asynchronous_multi_mode(FREQS, FFT, GROUPS, NMODES, OPTS)
%
% 输入
%   FREQS   频带内频率坐标（Hz，Nf x 1）
%   FFT     缩放 FFT，n x Nf，第 k 列为 F_k，约定 E[F_k F_k'] = E_k
%   GROUPS  测组划分的元胞数组，GROUPS{r} 为第 r 组各通道的索引。各组互不重叠
%           且覆盖全部通道；组内通道同步采集，组间采集时钟不同步
%   NMODES  频带内模态阶数 m
%   OPTS    结构体，字段：
%       qexp     模态频响函数的指数，缺省 0（0 加速度 / 1 速度 / 2 位移）
%       f0       各阶固有频率初值（Hz），长度 m，缺省取奇异值谱峰
%       maxiter  交替极小化的外层迭代上限，缺省 60
%       tol      收敛容差，缺省 1e-7
%       silent   是否静默，缺省 false
%
% 返回结构体字段
%   f         1 x m  各阶固有频率（Hz）
%   z         1 x m  各阶阻尼比
%   S         m x m  模态力谱密度矩阵；ng >= 2 时为实对称（非对角元允许为
%                     负实数），ng == 1 且 m > 1 时为 Hermitian 复矩阵（非对角
%                     元的实部与虚部都是自由参数）
%   Se        1 x q  各测组的预测误差谱密度
%   phi       n x m  各阶模态形状（各列欧氏范数为 1）；ng >= 2 时为实向量，
%                     ng == 1 且 m > 1 时为复向量，每列绝对值最大的分量为正
%                     实数（相位规范），其余分量的相位相对它给出
%   vij       结构体 各阶之间的相干矩阵，字段名 v12、v13、…，各为 q x q 的
%                     Hermitian 矩阵，对角元为 1
%   nllf      负对数似然函数值，式 (10.42)
%   niter     外层迭代次数
%   converged 是否满足收敛判据
%   cov       无量纲自由参数的后验协方差矩阵（对角线即各参数变异系数的平方）
%   coefv     各参数的后验变异系数，字段 f、z、S、Se、phi、v
%   groups    测组划分（原样返回）
%   m0        振型算子 G 的列空间维数，即降维后问题的维数
%
% 模型。第 r 组的各通道与本组时钟同步，组间的模态响应各自按本组时钟给出
% （式 (10.44)、(10.46)）：
%
%   F_k = sum_i U_i n_ik + e_k,   U_i = blockOp(u_i, groups) 属于 R^{n x q}
%
% 其中 U_i 的第 r 列只在第 r 组的行块上等于 u_i 的对应分量、其余为零，n_ik 为
% 第 i 阶模态响应按各组时钟的缩放 FFT。组间不完全相干由相干因子表征
% （式 (10.45)）：
%
%   E[n_ik^(r) conj(n_jk^(s))] = h_ik conj(h_jk) S_ij v_ijrs
%
% 由此得到谱矩阵（式 (10.47)、(10.48)）：
%
%   E_k = sum_i sum_j h_ik conj(h_jk) S_ij (U_i V_ij U_j') + D0,
%   D0  = diag{Se_r I_{n_r}},  V_ii = I_q
%
% 其中 h_ik = (2*pi*1i*f_k)^(-qexp) / (1 - beta^2 - 2i*z_i*beta)，beta = f_i/f_k。
% 取 G = [U_1, ..., U_m] 属于 R^{n x (m q)}、S_v 的 (i,j) 分块等于 S_ij V_ij、
% W_k = kron(diag(h_k), I_q)，则紧凑形式为
%
%   E_k = G W_k S_v W_k' G' + D0                                      (10.48)
%
% 参数化。模态力谱矩阵的规范（gauge）按测组数分两支。测组数 ng >= 2 时，式
% (10.45) 中 S_ij 的相位与 V_ij 的整体相位只以乘积出现，二者是同一规范自由度，
% 取 S 为实对称即固定该规范；若改用复 Hermitian，Hessian 沿该规范方向的曲率为
% 零，后验方差发散。测组数 ng == 1 时 V 不存在（v_entries 对 ng == 1 返回空），
% 式 (10.45) 的组间相干不进入模型，S_ij 的相位不再有与之对偶的自由度，是由数
% 据辨识的参数，故取 Hermitian 复矩阵；m == 1 时 S 只有对角元，两支一致。相干
% 矩阵的非对角元用「模 rho 属于 [0, 0.99]、相位 theta 属于 (-pi, pi]」参数化，
% Hermitian 性由构造保证（对角元恒为 1）。
%
% 模态形状的模型类同样按测组数分两支。ng >= 2 时振型取实向量；ng == 1 且 m > 1
% 时取复向量，使单测组情形与第 13 章的同步多阶问题（phi = B0*alpha，B0 为实正
% 交基、alpha 为复坐标）模型类一致。强制实形状时，式 (10.48) 的交叉项为
% S_ij*h_i*conj(h_j)*(u_i*u_j')，而同步多阶为 (B0*alpha_i)*h_i 与
% (B0*alpha_j)*conj(h_j) 的外积，两者不相等——复坐标给每个自由度一个附加相位，
% 该相位在 ng == 1 时没有 V_ij 可以吸收，只能由 u_i 自身承担，故实形状类严格
% 小于复形状类，同一份数据上最低负对数似然更高、阻尼比最可能值系统性偏移。
% 复形状在单位球面的切空间内参数化（见 tangent_basis）：u_i(theta) =
% (u_i0 + Z_i*theta)/||u_i0 + Z_i*theta||，Z_i 为 C^n 单位球切空间的正交基，
% 并经 Gram-Schmidt 剔除整体相位方向 1i*u_i0——该方向可与 S 的第 i 行、列相位
% 联动，属规范自由度，不剔除会使 Hessian 在该方向曲率为零。实坐标维数为
% 2n - 2（实向量情形为 n - 1）。单位范数由归一化保证。
%
% 相位规范。优化结束后把每阶形状归一化为「绝对值最大的分量为正实数」，并把该
% 相位差并回模态力谱矩阵：u_i -> u_i*exp(-1i*theta_i)、
% S_ij -> S_ij*exp(1i*(theta_i - theta_j))，式 (10.48) 的乘积不变。该规范同时
% 固定了 S 的相位基准，使后验 Hessian 满秩（ng == 1 时 S 的相位是可辨识参数，
% 但其基准依赖形状的整体相位）。
%
% 算法。交替极小化：先固定振型与相干矩阵更新谱参数与 S，再固定谱参数更新各阶
% 振型切空间坐标与各 V_ij 的模与相位；收敛后把全部自由参数拼成一个向量作一次
% 联合精化。输出前按频率升序重排各阶并固定符号规范。
%
% 校核入口（供校核脚本调用内部子函数，不属于常规调用形式）
%   OUT = bayoma.asynchronous_multi_mode('theoreticalPsd', PAR)
%   OUT = bayoma.asynchronous_multi_mode('nllf', PAR)
% PAR 为结构体，字段 f、z、S、Se、phi、groups、freqs、qexp，以及相干矩阵
% （vij 结构体，与输出同形；或 Vc 元胞，Vc{i,j} 为 q x q 矩阵）。第二种形式还需
% 字段 fft。前者返回 n x n x Nf 的谱矩阵序列，后者返回降维负对数似然函数值。
%
% 参见 bayoma.asynchronous_single_mode、bayoma.multi_mode、bayoma.uncertainty_law。

if ischar(freqs) && nargin >= 2 && isstruct(fft)
    out = internal_entry(lower(freqs), fft);
    return;
end

if nargin < 5 || isempty(opts)
    opts = struct();
end
opts = fill_defaults(opts, struct('qexp', 0, 'f0', [], 'maxiter', 60, ...
    'tol', 1e-7, 'silent', false));
% 与包内其它函数（single_mode、multi_mode、multi_setup、asynchronous_single_mode）
% 统一用 q 命名；qexp 保留为旧名，两者同时给出时以 q 为准。
if isfield(opts, 'q') && ~isempty(opts.q)
    opts.qexp = opts.q;
end

qexp = double(opts.qexp);
freqs = double(freqs(:));
fft = double(fft);
[nch, nf] = size(fft);
if numel(freqs) ~= nf
    error('bayoma:asynchronous:badSize', '频率点数必须等于 fft 的列数');
end
groups = validate_groups(groups, nch);
ng = numel(groups);
nrv = cellfun(@numel, groups);
nrv = nrv(:);

m = round(double(nModes));
if ~isscalar(m) || m < 1 || m > nch
    error('bayoma:asynchronous:badModes', '模态阶数必须是 1..%d 内的整数', nch);
end
nOff = m * (m - 1) / 2;
% 规范（gauge）规则：单测组且多阶时 S 取 Hermitian 复矩阵，其余情形取实对称。
% 判据见上文「参数化」一段与式 (10.44)–(10.45)。
complexS = (ng == 1) && (m > 1);
% 模型类规则：同一条件下模态形状取复向量，使单测组情形与第 13 章同步多阶
% 问题（phi = B0*alpha）的模型类一致。理由见上文「参数化」一段：实形状类
% 无法表达 alpha 给出的逐自由度附加相位，严格小于复形状类。
complexPhi = complexS;
vent = v_entries(m, ng);
if isempty(opts.f0)
    f0in = [];
else
    f0in = double(opts.f0(:).');
    if numel(f0in) ~= m
        error('bayoma:asynchronous:badF0', '频率初值的个数必须等于模态阶数');
    end
end

% 谱密度的上下界按数据量纲取：缩放 FFT 的平均谱能量给出 S 与 Se 的自然尺度，
% 相差八个数量级足以覆盖实际识别范围。用与数据无关的宽定界（如 1e-30~1e30）时，
% 有界拟牛顿法的线搜索在初值处即报告相对下降收敛，谱参数子问题不再前进。
dscale = mean(abs(fft(:)).^2);
if ~isfinite(dscale) || dscale <= 0
    dscale = 1;
end
smin = max(dscale * 1e-8, 1e-30);
smax = min(dscale * 1e8, 1e30);

% ---------------------------------------------------------------- 初值
[fv, zv, Sm, sev, phi, Vc] = initial_guess(freqs, fft, groups, m, qexp, f0in);

x = pack_spectrum(fv, zv, Sm, sev, complexS);
lb = [log(max(freqs(1) * 0.5, 1e-12)) * ones(m, 1);
      log(1e-6) * ones(m, 1);
      log(smin) * ones(m, 1);
      -smax * ones(nOff, 1);
      -smax * ones(complexS * nOff, 1);
      log(smin) * ones(ng, 1)];
ub = [log(freqs(end) * 2.0) * ones(m, 1);
      log(0.5) * ones(m, 1);
      log(smax) * ones(m, 1);
      smax * ones(nOff, 1);
      smax * ones(complexS * nOff, 1);
      log(smax) * ones(ng, 1)];

optSpec = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'MaxIterations', 100, 'MaxFunctionEvaluations', 1500, ...
    'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-12);
optShape = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
    'MaxIterations', 100, 'MaxFunctionEvaluations', 1500, ...
    'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-12);

cur = nllf(fv, zv, Sm, sev, phi, groups, Vc, freqs, fft, qexp);
if ~isfinite(cur)
    error('bayoma:asynchronous:badStart', '初值处的负对数似然函数非有限');
end
if ~opts.silent
    fprintf('  初值：f = [%s]，NLLF = %.6f\n', sprintf(' %.4f', fv), cur);
end

step_tol = max(sqrt(opts.tol), 1e-6);
converged = false;
niter = 0;

for niter = 1:opts.maxiter
    old = cur;
    xOld = x;
    phiOld = phi;
    VcOld = Vc;

    % ---- 谱参数与模态力谱矩阵子问题：对数空间的有界拟牛顿法 ----
    objS = @(xx) guard(spectrum_objective(xx, phi, Vc, freqs, fft, groups, ...
        qexp, m, nOff, ng, complexS));
    xn = fmincon(objS, x, [], [], [], [], lb, ub, [], optSpec);
    if all(isfinite(xn))
        x = xn(:);
    end
    [fv, zv, Sm, sev] = unpack_spectrum(x, m, nOff, ng, complexS);

    % ---- 振型切空间坐标与相干矩阵子问题 ----
    Zt = cell(m, 1);
    nth = zeros(m, 1);
    for i = 1:m
        % 复形状分支下切空间始终按 2n 维实坐标构造（即便当前 phi 恰为实向量），
        % 否则切空间只含 n-1 个实方向，第一步就退化为实形状类。
        Zt{i} = tangent_basis(phi(:, i), complexPhi);
        nth(i) = size(Zt{i}, 2);
    end
    y0 = [zeros(sum(nth), 1); pack_v(Vc, vent)];
    lby = [-ones(sum(nth), 1); repmat([0; -pi], size(vent, 1), 1)];
    uby = [ones(sum(nth), 1); repmat([0.99; pi], size(vent, 1), 1)];

    objA = @(yy) guard(shape_objective(yy, phi, Zt, nth, vent, m, ng, ...
        fv, zv, Sm, sev, freqs, fft, groups, qexp));
    yn = fmincon(objA, y0, [], [], [], [], lby, uby, [], optShape);
    if all(isfinite(yn))
        yv = yn(:);
    else
        yv = y0;
    end

    phiNew = phi;
    k = 0;
    for i = 1:m
        u = phi(:, i) + Zt{i} * yv(k + 1:k + nth(i));
        k = k + nth(i);
        nu = norm(u);
        if nu > 1e-12
            phiNew(:, i) = u / nu;
        end
    end
    VcNew = assemble_V(yv(k + 1:end), vent, m, ng);

    phi = phiNew;
    Vc = VcNew;
    cur = nllf(fv, zv, Sm, sev, phi, groups, Vc, freqs, fft, qexp);
    if ~isfinite(cur)
        phi = phiOld;
        Vc = VcOld;
        x = xOld;
        [fv, zv, Sm, sev] = unpack_spectrum(x, m, nOff, ng, complexS);
        cur = old;
        break;
    end

    rel = abs(old - cur) / max(abs(old), 1.0);
    step = max(abs([x - xOld; phi(:) - phiOld(:); ...
        pack_v(Vc, vent) - pack_v(VcOld, vent)]));
    if ~opts.silent
        fprintf('  迭代 %3d：NLLF = %.6f，相对变化 %.2e，参数步长 %.2e，f = [%s]\n', ...
            niter, cur, rel, step, sprintf(' %.4f', fv));
    end
    if rel < opts.tol && step < step_tol
        converged = true;
        break;
    end
end

% ---------------------------------------------------------------- 联合精化
% 谱参数、模态力谱矩阵非对角元、各测组预测误差谱密度、各阶振型切空间坐标与
% 各相干矩阵的模与相位拼成一个向量再极小化一次，把交替极小化停在的驻点收紧，
% 使后验协方差在似然驻点处计算。
base = build_base(fv, zv, Sm, sev, phi, Vc, m, ng, nOff, vent, complexS, complexPhi);
bnd = base_bounds(base);
objJ = @(yy) guard(base_objective(yy, base, freqs, fft, groups, qexp));
try
    yj = fmincon(objJ, base.y0, [], [], [], [], bnd.lb, bnd.ub, [], optShape);
    if all(isfinite(yj))
        val = objJ(yj);
        if isfinite(val) && val <= cur + 1e-8 * max(abs(cur), 1)
            base.y0 = yj(:);
            [fv, zv, Sm, sev, phi, Vc] = unpack_y(base.y0, base);
            cur = val;
            converged = true;
        end
    end
catch
end

% ---------------------------------------------------------------- 排序与符号
[fv, zv, Sm, sev, phi, Vc] = sort_and_normalise(fv, zv, Sm, sev, phi, Vc, m, ng);

G = assemble_G(phi, groups, ng);
[m0, ~] = column_basis(G);

out.f = fv(:).';
out.z = zv(:).';
% 取 Hermitian 部分：complexS 为真时保留非对角元的虚部，(S + S')/2 对实对称
% 矩阵不改变任何元素；(S + S.')/2 会取实部，不能用于复 Hermitian 情形。
out.S = (Sm + Sm') / 2;
out.Se = sev(:).';
out.phi = phi;
out.groups = groups;
out.vij = Vc_to_struct(Vc, m);
out.nllf = cur;
out.niter = niter;
out.converged = converged;
out.m0 = m0;

[cov, coefv] = covariance_estimate(out, freqs, fft, qexp, vent);
out.cov = cov;
out.coefv = coefv;
end

% ======================================================================
% 降维似然函数
% ======================================================================
function val = nllf(f, z, S, Se, phi, groups, Vc, freqs, fft, qexp)
%NLLF 式 (10.42) 的负对数似然函数（降维实现，全部运算在 m0 x m0 上进行）。
%
%   降维恒等式。取 G = [U_1, ..., U_m] 的列空间正交基 B（n x m0，m0 = rank G），
%   记 A = B'G、Bt = D0^(-1/2) B、H = Bt'Bt，E0_k = A W_k S_v W_k' A'，则
%
%       ln|E_k| = sum_r n_r ln Se_r + ln det(I_m0 + E0_k H)
%       F_k' inv(E_k) F_k = |D0^(-1/2) F_k|^2 - G_k' inv(inv(E0_k) + H) G_k
%
%   其中 G_k = Bt' D0^(-1/2) F_k。第一个式子由 det(D0 + Bt^(1/2)…) 的 Sylvester
%   定理 ded(I_n + Bt E0 Bt') = det(I_m0 + E0 Bt'Bt) 得到；第二个式子由 Woodbury
%   恒等式 inv(inv(E0)+H) = E0 - E0 W (I + W E0 W)^(-1) W E0 得到，其中
%   W = H^(1/2)。两式都化为同一个 m0 x m0 的 Hermitian 正定矩阵
%   Q_k = I_m0 + W E0_k W 的 Cholesky 分解，单次评估为 O(Nf m0^3)。
[nch, nf] = size(fft);
ng = numel(groups);
m = numel(f);
val = inf;

if any(~isfinite(f)) || any(f <= 0) || any(~isfinite(z)) || any(z <= 0) || any(z >= 1)
    return;
end
Se = Se(:);
if numel(Se) ~= ng || any(~isfinite(Se)) || any(Se <= 0)
    return;
end
% 式 (10.47) 中的 S 只以 Hermitian 形式进入似然：取 (S + S')/2 保留非对角元的
% 虚部（单测组多阶时 S 为复 Hermitian），对实对称矩阵不改变任何元素。若写作
% real(S)，ng == 1 且 m > 1 时 S 的虚部被强制为零，似然被限制在实对称子流形上，
% 与全维似然不再相等。
S = (S + S') / 2;
% 模态形状允许为复向量（ng == 1 且 m > 1 分支，见主函数「参数化」一段），故
% 此处不作 real(phi)。取实部会把复形状的虚部强制为零，似然被限制在实形状子
% 流形上，与同步多阶的复坐标模型不再等价。
if any(~isfinite(S(:))) || any(~isfinite(phi(:)))
    return;
end

h = bayoma.transfer_matrix(f(:), z(:), freqs, qexp);
if any(~isfinite(h(:)))
    return;
end

% 振型为复向量时 U_i 为复矩阵，式 (10.47) 的外积为 U_i V_ij U_j^H（共轭转置）。
% 实振型下两者相同，故统一写作共轭转置，使两支共用同一条代码路径。
G = assemble_G(phi, groups, ng);
[m0, B] = column_basis(G);
if m0 == 0
    return;
end
A = B' * G;

d0 = noise_diagonal(Se, groups, nch);
r0 = sqrt(d0);
Bt = B ./ r0;
H = Bt' * Bt;
H = (H + H') / 2;
[Wv, Dw] = eig(H);
W = Wv * diag(sqrt(max(real(diag(Dw)), 0))) * Wv';
W = (W + W') / 2;

% 各阶分块 P_ij = A_i V_ij A_j^H，E0_k = sum_ij h_i conj(h_j) S_ij P_ij。
% 复振型下 A_j 为复矩阵，必须用共轭转置 A_j'；写作 A_j.' 得到的是
% U_i V_ij U_j^T，E0_k 随之非 Hermitian，batched_chol 的 Cholesky 分解失败。
P = cell(m, m);
for i = 1:m
    Ai = A(:, (i - 1) * ng + (1:ng));
    for j = 1:m
        Aj = A(:, (j - 1) * ng + (1:ng));
        P{i, j} = Ai * Vc{i, j} * Aj';
    end
end

Y = Bt' * (fft ./ r0);
Q3 = complex(zeros(m0, m0, nf));
for i = 1:m
    for j = 1:m
        if S(i, j) == 0
            continue;
        end
        cf = S(i, j) * h(i, :) .* conj(h(j, :));
        Q3 = Q3 + reshape(cf, 1, 1, nf) .* (W * P{i, j} * W);
    end
end
Q3 = Q3 + eye(m0);

% 二次型按“数据投影”形式计算，不能先展开成 |y|^2 - Y'(inv(E0)+H)^(-1)Y 再
% 拆成 y'y、Y'E0Y 与三角回代项相减。谱值比真实值大若干量级时，Y'E0Y 与回代
% 项都达 1e+13 而两者之差只剩 1，直接相减丢掉全部有效位（实测负对数似然出现
% -5.5e+11 的伪极小值，优化器随即停在参数边界）。改用等价形式
%       q_k = |y_k|^2 - |g_k|^2 + g_k' inv(Q_k) g_k,  g_k = W^(-1) Y_k,
% 其中 |y_k|^2 - |g_k|^2 与频率无关只算一次，g_k' inv(Q_k) g_k 由 Cholesky 回代
% 得到；两项都随谱值增大单调趋向有限值，不发生大数相消。
Winv = Wv * diag(1 ./ sqrt(max(real(diag(Dw)), 0))) * Wv';
g = Winv * Y;
[ok, logdetTot, quadTot] = batched_chol(Q3, g);
if ~ok
    val = inf;
    return;
end

fullq = sum(sum(abs(fft).^2 ./ d0));
gsq = sum(sum(abs(g).^2));
val = nch * nf * log(pi) + nf * sum(nrv_of(groups) .* log(Se)) ...
    + logdetTot + (fullq - gsq) + quadTot;
if ~isfinite(val)
    val = inf;
end
end

% ======================================================================
% 谱矩阵组装（式 (10.47) 的双重求和）
% ======================================================================
function Ek = theoreticalPsd(f, z, S, Se, phi, groups, Vc, freqs, qexp)
%THEORETICALPSD 按式 (10.47) 组装 n x n x Nf 的谱矩阵序列。
%
%   E_k = sum_i sum_j h_ik conj(h_jk) S_ij (U_i V_ij U_j^H) + D0
%
% D0 = diag{Se_r I_{n_r}}。该式与式 (10.48) 的紧凑形式（G、S_v、W_k）逐元素
% 相等，降维似然即由此式出发消去 n 维算子得到。振型为复向量时（ng == 1 且
% m > 1 分支）U_i 为复矩阵，外积取共轭转置 U_j^H；实振型下与 U_j^T 相同。
nch = size(phi, 1);
nf = numel(freqs);
ng = numel(groups);
m = numel(f);
% 与 nllf 一致取 Hermitian 部分：单测组多阶时 S 为复 Hermitian，其虚部进入式
% (10.47) 的交叉项；实对称情形 (S + S')/2 不改变任何元素。
S = (S + S') / 2;
h = bayoma.transfer_matrix(f(:), z(:), freqs, qexp);
d0 = noise_diagonal(Se, groups, nch);
U = cell(m, 1);
for i = 1:m
    U{i} = block_op(phi(:, i), groups, ng);
end
Ek = complex(zeros(nch, nch, nf));
Dm = diag(d0);
for k = 1:nf
    Ek(:, :, k) = Dm;
end
for i = 1:m
    for j = 1:m
        cf = S(i, j) * h(i, :) .* conj(h(j, :));
        base = U{i} * Vc{i, j} * U{j}';
        Ek = Ek + reshape(cf, 1, 1, nf) .* base;
    end
end
end

% ======================================================================
% 校核入口
% ======================================================================
function out = internal_entry(name, par)
%INTERNAL_ENTRY 供校核脚本调用子函数的入口，不属于常规调用形式。
%
%   相干矩阵可用两种形式给出：vij 结构体（字段 v12、v13、…，与 out.vij 同形）
%   或 Vc 元胞（{i,j} 为 q x q 矩阵）。两者都给出时以 Vc 元胞为准；只给出
%   vij 而它不是结构体（例如直接给元胞）时同样按元胞处理，不能静默改用单位阵。
if ~isfield(par, 'qexp') || isempty(par.qexp)
    par.qexp = 0;
end
if isfield(par, 'Vc') && iscell(par.Vc) && ~isempty(par.Vc)
    Vc = par.Vc;
elseif isfield(par, 'vij') && isstruct(par.vij)
    Vc = struct_to_Vc(par.vij, numel(par.f), par.groups);
elseif isfield(par, 'vij') && iscell(par.vij)
    Vc = par.vij;
else
    error('bayoma:asynchronous:badEntry', ...
        '校核入口需要 Vc 元胞或 vij 结构体，以确定相干矩阵');
end
switch name
    case 'theoreticalpsd'
        out = theoreticalPsd(par.f, par.z, par.S, par.Se, par.phi, par.groups, ...
            Vc, par.freqs(:), par.qexp);
    case 'nllf'
        out = nllf(par.f, par.z, par.S, par.Se, par.phi, par.groups, ...
            Vc, par.freqs(:), par.fft, par.qexp);
    otherwise
        error('bayoma:asynchronous:badEntry', '未知的校核入口 %s', name);
end
end

% ======================================================================
% 谱参数打包与解包
% ======================================================================
function x = pack_spectrum(f, z, S, Se, complexS)
%PACK_SPECTRUM 谱参数打包为无约束向量。
%
% 排列为 [ln f; ln z; ln diag(S); Re S_ij (i<j); Im S_ij (i<j); ln Se]，仅当
% complexS 为真时含虚部一段。规范规则见主函数：complexS 为真表示单测组且多阶，
% 此时 S_ij 的相位按式 (10.45) 可辨识，需把实部与虚部同时作为自由参数；其余
% 情形 S 为实对称，虚部一段长度为零，向量排列与原有实现完全一致。
if nargin < 5 || isempty(complexS)
    complexS = false;
end
m = numel(f);
S = (S + S') / 2;
so = offdiag_vec(S);
if complexS
    soIm = imag(so).';
else
    soIm = zeros(1, 0);
end
x = [log(f(:)); log(z(:)); log(max(real(diag(S)), 1e-300)); ...
    real(so).'; soIm; log(Se(:))];
end

function [f, z, S, Se] = unpack_spectrum(x, m, nOff, ng, complexS)
%UNPACK_SPECTRUM 由无约束向量还原谱参数，排列与 pack_spectrum 互逆。
%
% complexS 为真时非对角元为 S(i,j) = so_re + i*so_im、S(j,i) = conj(S(i,j))，
% 构成对角元为实数的 Hermitian 矩阵；为假时虚部一段长度为零，S 为实对称。
if nargin < 5 || isempty(complexS)
    complexS = false;
end
nIm = complexS * nOff;
f = exp(x(1:m));
z = exp(x(m + 1:2 * m));
sd = exp(x(2 * m + 1:3 * m));
soRe = x(3 * m + 1:3 * m + nOff);
soIm = x(3 * m + nOff + 1:3 * m + nOff + nIm);
Se = exp(x(3 * m + nOff + nIm + 1:3 * m + nOff + nIm + ng));
S = diag(sd);
k = 0;
for i = 1:m - 1
    for j = i + 1:m
        k = k + 1;
        if complexS
            S(i, j) = soRe(k) + 1i * soIm(k);
            S(j, i) = soRe(k) - 1i * soIm(k);
        else
            S(i, j) = soRe(k);
            S(j, i) = soRe(k);
        end
    end
end
f = f(:);
z = z(:);
Se = Se(:);
end

function val = spectrum_objective(x, phi, Vc, freqs, fft, groups, qexp, m, nOff, ng, complexS)
%SPECTRUM_OBJECTIVE 给定振型与相干矩阵时的谱参数子问题目标函数。
if nargin < 11 || isempty(complexS)
    complexS = false;
end
[f, z, S, Se] = unpack_spectrum(x, m, nOff, ng, complexS);
val = nllf(f, z, S, Se, phi, groups, Vc, freqs, fft, qexp);
if ~isfinite(val)
    val = 1e300;
end
end

function val = shape_objective(y, phi, Zt, nth, vent, m, ng, f, z, S, Se, ...
    freqs, fft, groups, qexp)
%SHAPE_OBJECTIVE 给定谱参数时的振型切空间坐标与相干矩阵子问题目标函数。
phiNew = phi;
k = 0;
for i = 1:m
    u = phi(:, i) + Zt{i} * y(k + 1:k + nth(i));
    k = k + nth(i);
    nu = norm(u);
    if nu <= 1e-12
        val = 1e300;
        return;
    end
    phiNew(:, i) = u / nu;
end
Vc = assemble_V(y(k + 1:end), vent, m, ng);
val = nllf(f, z, S, Se, phiNew, groups, Vc, freqs, fft, qexp);
if ~isfinite(val)
    val = 1e300;
end
end

% ======================================================================
% 无量纲自由参数：谱参数按最可能值归一，振型取切空间坐标，相干矩阵取模与相位
% ======================================================================
function base = build_base(f, z, S, Se, phi, Vc, m, ng, nOff, vent, complexS, complexPhi)
%BUILD_BASE 构造无量纲自由参数的基准（最可能值与切空间基）。
%
% 参数向量 y 的排列：[谱参数(3m + (1 + complexS)*nOff + ng)；各阶切空间坐标；
% 各相干元的模与相位交替]。谱参数依次为 f、z、S 的对角元、S 非对角元的实部、
% S 非对角元的虚部（仅 complexS 为真时存在）、Se，各自按最可能值归一。
%
% 振型切空间维数随模型类切换：实形状每阶 n-1 个实坐标；复形状（ng == 1 且
% m > 1）每阶 2n-2 个实坐标，单位球面切空间内剔除整体相位方向，见
% tangent_basis。
if nargin < 11 || isempty(complexS)
    complexS = false;
end
if nargin < 12 || isempty(complexPhi)
    complexPhi = complexS;
end
S = (S + S') / 2;
Sd = max(real(diag(S)), 1e-300);
so = offdiag_vec(S);
soRe = real(so);
soIm = imag(so);
% 非对角元按对应两个对角元的几何平均归一，无量纲坐标即相关系数，量级为 1；
% 若按其自身最可能值归一，最可能值为 0 时无量纲坐标退化为零向量，Hessian 的
% 对应行列为零，后验方差失效。虚部按同一尺度归一：最可能值处虚部为零，无量纲
% 坐标为零，但该方向的曲率由几何平均尺度给出，量级与实部一致。
soScale = zeros(1, nOff);
k = 0;
for i = 1:m - 1
    for j = i + 1:m
        k = k + 1;
        soScale(k) = sqrt(Sd(i) * Sd(j));
    end
end
soScale = soScale(:);
if complexS
    numScale = [soScale; soScale];
    numVal = [soRe(:); soIm(:)];
else
    numScale = soScale;
    numVal = soRe(:);
end
base.m = m;
base.ng = ng;
base.nOff = nOff;
base.complexS = complexS;
base.complexPhi = complexPhi;
base.nsp = 3 * m + (1 + complexS) * nOff + ng;
base.scale = [f(:); z(:); Sd; numScale; Se(:)];
base.phi = phi;
base.Zt = cell(m, 1);
base.nth = zeros(m, 1);
for i = 1:m
    % 复形状分支下按 forceComplex 构造切空间：即便当前振型恰为实向量，也要
    % 保留 2n-2 维复切空间，否则坐标维数少一半，复形状不可达。
    base.Zt{i} = tangent_basis(phi(:, i), complexPhi);
    base.nth(i) = size(base.Zt{i}, 2);
end
base.nthtot = sum(base.nth);
base.vent = vent;
base.npv = size(vent, 1);
% 无量纲坐标取最可能值：谱参数与 Se 恰为 1，S 非对角元的实部为相关系数，
% 虚部为 0（最可能值处 S 的虚部即该坐标的原点）
spec = [f(:); z(:); Sd; numVal; Se(:)];
base.y0 = [spec ./ base.scale; zeros(base.nthtot, 1); pack_v(Vc, vent)];
end

function bnd = base_bounds(base)
%BASE_BOUNDS 联合精化与后验协方差所用的箱式约束（无量纲坐标）。
nsp = base.nsp;
ntot = nsp + base.nthtot + 2 * base.npv;
lb = zeros(ntot, 1);
ub = zeros(ntot, 1);
lb(1:nsp) = 1e-3;
ub(1:nsp) = 1e3;
% 对角元与 Se 为正，非对角元的实部（相关系数）与虚部均可正可负，其界单独给出；
% 虚部与实部取同一界 [-2, 2]，两者无量纲化所用的尺度相同（对角元的几何平均）
lb(3 * base.m + 1:3 * base.m + base.nOff) = -2;
ub(3 * base.m + 1:3 * base.m + base.nOff) = 2;
if base.complexS
    lb(3 * base.m + base.nOff + 1:3 * base.m + 2 * base.nOff) = -2;
    ub(3 * base.m + base.nOff + 1:3 * base.m + 2 * base.nOff) = 2;
end
lb(nsp + 1:nsp + base.nthtot) = -1;
ub(nsp + 1:nsp + base.nthtot) = 1;
k = nsp + base.nthtot;
lb(k + 1:2:ntot) = 0;
ub(k + 1:2:ntot) = 0.99;
lb(k + 2:2:ntot) = -pi;
ub(k + 2:2:ntot) = pi;
bnd.lb = lb;
bnd.ub = ub;
end

function [f, z, S, Se, phi, Vc] = unpack_y(y, base)
%UNPACK_Y 由无量纲参数向量还原全部模态参数。
m = base.m;
ng = base.ng;
nOff = base.nOff;
sp = y(1:base.nsp) .* base.scale;
f = sp(1:m);
z = sp(m + 1:2 * m);
S = diag(sp(2 * m + 1:3 * m));
soRe = sp(3 * m + 1:3 * m + nOff);
soIm = sp(3 * m + nOff + 1:3 * m + 2 * nOff);
Se = sp(3 * m + (1 + base.complexS) * nOff + 1:base.nsp);
k = 0;
for i = 1:m - 1
    for j = i + 1:m
        k = k + 1;
        if base.complexS
            S(i, j) = soRe(k) + 1i * soIm(k);
            S(j, i) = soRe(k) - 1i * soIm(k);
        else
            S(i, j) = soRe(k);
            S(j, i) = soRe(k);
        end
    end
end
phi = base.phi;
k = base.nsp;
for i = 1:m
    nt = base.nth(i);
    u = base.phi(:, i) + base.Zt{i} * y(k + 1:k + nt);
    k = k + nt;
    nu = norm(u);
    if nu > 1e-12
        phi(:, i) = u / nu;
    end
end
Vc = assemble_V(y(k + 1:end), base.vent, m, ng);
end

function val = base_objective(y, base, freqs, fft, groups, qexp)
%BASE_OBJECTIVE 无量纲自由参数向量对应的负对数似然函数。
%
%   返回原始似然值（可能是 inf）。优化器一侧由 guard 转成有限惩罚；后验协方差
%   一侧需要区分"有限但很大"与"模型无效"，故此处不做替换。
[f, z, S, Se, phi, Vc] = unpack_y(y, base);
val = nllf(f, z, S, Se, phi, groups, Vc, freqs, fft, qexp);
end

% ======================================================================
% 后验协方差
% ======================================================================
function [cov, coefv] = covariance_estimate(out, freqs, fft, qexp, vent)
%COVARIANCE_ESTIMATE 后验协方差矩阵（数值 Hessian 的正则化逆）。
%
% 自由参数取无量纲向量 y（与联合精化同形）。谱参数按最可能值归一，故协方差
% 矩阵对角线前 3m+(1+complexS)*nOff+ng 个元素直接是各谱参数变异系数的平方；
% 振型取单位球切空间坐标，变异系数取该列全部切向坐标方差之和的平方根（课本
% 11.3.3 节）；相干矩阵取模与相位，变异系数取两者方差之和的平方根。
%
% S 的非对角元在 complexS 为真时含实部与虚部两个无量纲坐标，二者的合成方差为
% var_re + var_im，故 coefv.S 的非对角元取该和的平方根，coefv.S 保持为实矩阵。
%
% 自由参数个数随模型类变化：谱参数段为 3m+(1+complexS)*nOff+ng，振型段在
% ng == 1 且 m > 1 时为每阶 2n-2 个（复形状）、其余情形每阶 n-1 个（实形状），
% 相干矩阵段为 2*numel(vent)。相位规范已在 sort_and_normalise 中固定（每阶
% 最大分量为正实数），故复形状的切空间不再含零曲率的规范方向，数值 Hessian
% 正定。
m = numel(out.f);
ng = numel(out.Se);
nOff = m * (m - 1) / 2;
complexS = (ng == 1) && (m > 1);
complexPhi = complexS;
Vc = struct_to_Vc(out.vij, m, out.groups);
base = build_base(out.f(:), out.z(:), out.S, out.Se(:), out.phi, Vc, m, ng, nOff, ...
    vent, complexS, complexPhi);
if numel(base.y0) ~= base.nsp + base.nthtot + 2 * base.npv
    error('bayoma:asynchronous:badBase', '无量纲参数向量的长度不一致');
end
nfree = numel(base.y0);
obj = @(yy) base_objective(yy, base, freqs, fft, out.groups, qexp);
H = numerical_hessian(obj, base.y0, 1e-4 * ones(nfree, 1));
cov = regularised_inverse(H);
dg = max(diag(cov), 0);
coefv = struct();
coefv.f = sqrt(dg(1:m)).';
coefv.z = sqrt(dg(m + 1:2 * m)).';
sd = sqrt(dg(2 * m + 1:3 * m));
Sm = zeros(m, m);
Sm(1:m + 1:end) = sd;
soRe = dg(3 * m + 1:3 * m + nOff);
if complexS
    soIm = dg(3 * m + nOff + 1:3 * m + 2 * nOff);
else
    soIm = zeros(nOff, 1);
end
so = sqrt(soRe + soIm);
k = 0;
for i = 1:m - 1
    for j = i + 1:m
        k = k + 1;
        Sm(i, j) = so(k);
        Sm(j, i) = so(k);
    end
end
coefv.S = Sm;
coefv.Se = sqrt(dg(3 * m + (1 + complexS) * nOff + 1:3 * m + (1 + complexS) * nOff + ng)).';
cphi = zeros(1, m);
kp = base.nsp;
for i = 1:m
    idx = kp + (1:base.nth(i));
    kp = kp + base.nth(i);
    cphi(i) = sqrt(max(sum(dg(idx)), 0));
end
coefv.phi = cphi;
vc = struct();
kp = base.nsp + base.nthtot;
for t = 1:base.npv
    nm = sprintf('v%d%d', vent(t, 1), vent(t, 2));
    val = sqrt(max(dg(kp + 2 * t - 1) + dg(kp + 2 * t), 0));
    if isfield(vc, nm)
        vc.(nm) = max(vc.(nm), val);
    else
        vc.(nm) = val;
    end
end
coefv.v = vc;
end

% ======================================================================
% 初值
% ======================================================================
function [f0, z0, S0, Se0, phi0, Vc] = initial_guess(freqs, fft, groups, m, qexp, f0in)
%INITIAL_GUESS 按高信噪比渐近式构造初值。
%
%   频率初值取奇异值谱第 1 条曲线的显著谱峰（单次实现的谱矩阵秩为 1，该曲线
%   即 ||F_k||），密集模态之间按 3*df 拉开；阻尼比取 0.01；振型初值取各阶初始
%   频率处样本 FFT 的主方向，组内按本组最大分量的相位对齐后取实部（组间时钟
%   不同步，故相位基准必须逐组建立），再作 Gram-Schmidt 正交化；S 的对角元由
%   各组谱峰扣除噪声基底后求和估计；Se_r 由组内高阶奇异值估计；V_ij 取单位阵。
[nch, nf] = size(fft);
ng = numel(groups);
df = freqs(2) - freqs(1);
spanf = freqs(end) - freqs(1);
z0 = 0.01 * ones(m, 1);      % 必须为 m 维：标量会在打包时少一个元素

s1 = sqrt(sum(abs(fft).^2, 1));
if isempty(f0in)
    f0 = initial_frequencies(freqs, s1, m, df);
else
    f0 = sort(f0in(:));
end
f0 = f0(:);

% Se_r：组内块局部平滑后的最小特征值谱在带内取 10% 分位数。
% 频带全平均的组内秩估计 (tr A_r - lambda_max)/((n_r - 1) N_f) 只在单模态频带成立，
% 多模态时把相邻阶的模态能量计入噪声，可高估一个数量级以上（与
% python/bayoma/asynchronous_multi.py 的 _noise_floor 同一理由）。
halfw = max(3 * df, 0.01 * spanf);
nw = max(1, round(halfw / df));
Se0 = zeros(ng, 1);
for r = 1:ng
    idx = groups{r};
    nr = numel(idx);
    if nr == 1
        cd = real(reshape(fft(idx, :), 1, nf));
        Se0(r) = max(prctile_local(cd, 10), 1e-30);
        continue;
    end
    spec = zeros(nf, 1);
    for k = 1:nf
        lo = max(1, k - nw);
        hi = min(nf, k + nw);
        blk = real(fft(idx, lo:hi) * fft(idx, lo:hi)');
        blk = 0.5 * (blk / (hi - lo + 1) + (blk / (hi - lo + 1))');
        ev = real(eig(blk));
        spec(k) = min(ev);
    end
    Se0(r) = max(prctile_local(spec, 10), 1e-30);
end

% 振型初值：各阶取「带内平均的组内块主特征向量」按样本模态能量缩放后拼接。
%
%   组内同步使组内样本谱矩阵块在每个频率点近似秩一，带内平均后的主特征向量即
%   该阶局部振型；按 sqrt(lambda_max - Se_r) 缩放后拼接，使组间相对幅值由样本
%   谱的模态能量给出（第 12.3.1 节的高信噪比渐近式）。只取单个频率点会丢失组间
%   相对幅值（各组块最大特征值可相差两个数量级）；对拼接结果作 Gram-Schmidt
%   正交化会压掉密集模态的第二阶（两阶振型夹角小，正交化后第二阶被投影成残差）。
%   式 (10.47) 只约束 u_i'u_i = 1，不要求不同阶正交，故此处只逐列归一化。
half = max(3 * df, 0.01 * spanf);
back = max(1, round(half / df));
phi0 = zeros(nch, m);
lam = zeros(m, ng);
for i = 1:m
    [~, kc] = min(abs(freqs - f0(i)));
    lo = max(1, kc - back);
    hi = min(nf, kc + back);
    u = zeros(nch, 1);
    for r = 1:ng
        idx = groups{r};
        seg = fft(idx, lo:hi);
        % 组内通道同步采集，样本块 seg*seg' 是 Hermitian 的秩一近似：模态项为
        % u_r u_r' |n_k|^2，与组内时钟的相位无关，只有整组的基准相位不可辨识。
        % 必须用共轭转置（seg'），写作 seg.' 得到的是 u_r u_r' n^2 这类复对称
        % 矩阵，其实部特征值正负取决于 n^2 的相位，取最大代数特征值时可能选中
        % 噪声方向，初值振型随之与真值近乎正交。
        blk = (seg * seg') / max(hi - lo + 1, 1);
        blk = 0.5 * (blk + blk');
        [vecs, vals] = eig(blk);
        dv = real(diag(vals));
        [lamMax, jmax] = max(dv);
        tvec = vecs(:, jmax);
        % 组内基准相位由本组幅值最大的分量确定，取实部后即为该组的实振型
        [~, km] = max(abs(tvec));
        tvec = real(tvec * exp(-1i * angle(tvec(km))));
        [~, km] = max(abs(tvec));
        if tvec(km) < 0
            tvec = -tvec;
        end
        lam(i, r) = lamMax;
        g = sqrt(max(lamMax - Se0(r), 1e-3 * max(lamMax, 1e-30)));
        u(idx) = g * tvec;
    end
    phi0(:, i) = u;
end

% 逐列归一化；近共线（夹角余弦大于 0.99）时改用块对角样本谱之和的高阶特征向量
% 补足，保证 assemble_G 的列满秩。异步数据不存在跨组的样本谱矩阵，故补足向量
% 取 A0 = sum_k blockdiag_r D_k[g_r, g_r] 的特征向量。
A0 = zeros(nch, nch);
for r = 1:ng
    idx = groups{r};
    A0(idx, idx) = A0(idx, idx) + real(fft(idx, :) * fft(idx, :)');
end
A0 = 0.5 * (A0 + A0.');
[V0, D0v] = eig(A0);
[~, ord0] = sort(real(diag(D0v)), 'descend');

pool = zeros(nch, 0);
for i = 1:m
    v = phi0(:, i);
    nv = norm(v);
    if nv <= 1e-12
        continue;
    end
    v = v / nv;
    duplicate = false;
    for j = 1:size(pool, 2)
        if abs(v.' * pool(:, j)) > 0.99
            duplicate = true;
            break;
        end
    end
    if ~duplicate
        pool = [pool, v]; %#ok<AGROW>
    end
end
for t = 1:numel(ord0)
    if size(pool, 2) >= m
        break;
    end
    v = real(V0(:, ord0(t)));
    for j = 1:size(pool, 2)
        v = v - pool(:, j) * (pool(:, j).' * v);
    end
    nv = norm(v);
    if nv > 1e-12
        pool = [pool, v / nv]; %#ok<AGROW>
    end
end
if size(pool, 2) < m
    error('bayoma:asynchronous:badShapes', ...
        '无法构造线性无关的初始模态形状，请检查频带内的模态数');
end
phi0 = pool(:, 1:m);

% S 的对角元按高信噪比渐近式反解：E_k 在 f = f_i 处的组内块主特征值近似为
% S_ii / (4 zeta_i^2) + S_er（式 10.47 的传递函数在共振点取 1/(2 zeta)，
% 加速度数据另有 (2 pi f_i)^(-2 qexp) 因子）。
S0 = zeros(m, m);
for i = 1:m
    peak = max(lam(i, :));
    S0(i, i) = max((peak - max(Se0)) * (2 * z0(i))^2 ...
        * (2 * pi * f0(i))^(2 * qexp), 1e-12);
end

Vc = cell(m, m);
for i = 1:m
    for j = 1:m
        Vc{i, j} = eye(ng);
    end
end
end

function f0 = initial_frequencies(freqs, s1, m, df)
%INITIAL_FREQUENCIES 在奇异值谱第 1 条曲线上取 m 个显著谱峰作为频率初值。
%
%   平滑窗宽取频带跨度的 1%，最小间隔取 3 df 与 2% 频带跨度中的较大者（与
%   python/bayoma/asynchronous_multi.py 的 _peak_frequencies 一致）。最小间隔
%   只取 3 df 时，同一共振峰顶附近被组间互谱起伏抬起的相邻频点会同时入选，
%   两阶初值落在同一个峰上，交替极小化随即收敛到该峰的单一模态解。
fr = freqs(:);
nf = numel(fr);
s1 = s1(:).';
span = fr(end) - fr(1);
w = max(3, 2 * round(0.005 * span / max(df, 1e-12)) + 1);
if w > 1 && mod(w, 2) == 1 && w < nf
    sm = movmean(s1, w);
else
    sm = s1;
end
hw = max(5, round(0.05 * nf));
cand = zeros(1, 0);
prom = zeros(1, 0);
for i = 2:nf - 1
    if sm(i) >= sm(i - 1) && sm(i) > sm(i + 1)
        lmin = min(sm(max(1, i - hw):i));
        rmin = min(sm(i:min(nf, i + hw)));
        p = sm(i) - max(lmin, rmin);
        if p > 0
            cand(end + 1) = i; %#ok<AGROW>
            prom(end + 1) = p; %#ok<AGROW>
        end
    end
end
minSep = max(3 * df, 0.02 * span);
chosen = zeros(1, 0);
if ~isempty(cand)
    [~, ord] = sort(prom, 'descend');
    ord = ord(prom(ord) >= 0.05 * max(sm));
    for j = ord
        if numel(chosen) >= m
            break;
        end
        if isempty(chosen) || all(abs(fr(cand(j)) - fr(chosen)) >= minSep)
            chosen(end + 1) = cand(j); %#ok<AGROW>
        end
    end
end
if numel(chosen) < m
    [~, ord] = sort(sm, 'descend');
    for j = ord
        if numel(chosen) >= m
            break;
        end
        if isempty(chosen) || all(abs(fr(j) - fr(chosen)) >= minSep)
            chosen(end + 1) = j; %#ok<AGROW>
        end
    end
end
if numel(chosen) < m
    f0 = linspace(fr(1), fr(end), m).';
    return;
end
f0 = sort(fr(chosen(1:m)));
for i = 2:numel(f0)
    if f0(i) - f0(i - 1) < minSep
        f0(i) = min(f0(i - 1) + minSep, fr(end));
    end
end
f0 = f0(:);
end

% ======================================================================
% 排序与符号规范
% ======================================================================
function [f, z, S, Se, phi, Vc] = sort_and_normalise(f, z, S, Se, phi, Vc, m, ng)
%SORT_AND_NORMALISE 按频率升序重排各阶并固定相位（符号）规范。
%
% 相位规范：各阶模态形状取「绝对值最大的分量为正实数」。该阶整体乘
% exp(-1i*theta)（theta 为该分量的相位）使 U_i -> U_i*exp(-1i*theta)，式
% (10.48) 中 (i,j) 与 (j,i) 的交叉项相位随之改变，(i,i) 项不变。恢复 E_k 的
% 做法是把 S 的第 i 行与第 i 列作同一共轭相位变换（S -> diag(e^{i*theta}) *
% S * diag(e^{-i*theta})）：S_ij -> S_ij*exp(1i*(theta_i - theta_j))，S_ji ->
% conj(S_ij) 随之保持 Hermitian，交叉项随即还原。此处不能改为改变 V_ij：V_ij
% 的对角元被约束为 1，乘相位会破坏该约束并改变 (i,i) 项的组间分块。
%
% 实形状时 theta 取 0 或 pi，上式退化为原来按符号反号的 D*S*D。
[~, ord] = sort(f(:).');
f = f(ord);
z = z(ord);
S = S(ord, ord);
phi = phi(:, ord);
V2 = cell(m, m);
for i = 1:m
    for j = 1:m
        V2{i, j} = Vc{ord(i), ord(j)};
    end
end
Vc = V2;

theta = zeros(1, m);
for i = 1:m
    [~, km] = max(abs(phi(:, i)));
    theta(i) = angle(phi(km, i));
end
theta = real(theta);
phi = phi .* repmat(exp(-1i * theta), size(phi, 1), 1);
ph = exp(1i * theta(:));
S = (ph * ones(1, m)) .* S .* (ones(m, 1) * conj(ph).');
% 相位变换在 S 为 Hermitian 时仍为 Hermitian；此处取 (S + S')/2 只是消除舍入
% 造成的反对称部分，保留非对角元的虚部。若写作 (S + S.')/2（非共轭转置），
% 单测组多阶情形会把 S 的虚部清零，与 nllf 使用的 S 不一致。
S = (S + S') / 2;
% 归一化后最大分量应为正实数；舍入可能留下 1e-17 量级的虚部，逐列再归一化
for i = 1:m
    nu = norm(phi(:, i));
    if nu > 0
        phi(:, i) = phi(:, i) / nu;
    end
end
for i = 1:m
    Vc{i, i}(1:ng + 1:end) = 1;
end
f = f(:).';
z = z(:).';
end

% ======================================================================
% 分块算子与相干矩阵
% ======================================================================
function G = assemble_G(phi, groups, ng)
%ASSEMBLE_G 构造 G = [U_1, ..., U_m]，U_i 的第 r 列只在第 r 组行块上非零。
%
%   模态形状为复向量时（ng == 1 且 m > 1 分支）G 随之取复值；若按实矩阵分配，
%   复振型的虚部会在赋值时被静默丢弃，模型退化为实形状类。形状仍为实值时按
%   实矩阵分配，避免整条似然路径走复数运算带来的无谓开销。
nch = size(phi, 1);
m = size(phi, 2);
if isreal(phi)
    G = zeros(nch, m * ng);
else
    G = complex(zeros(nch, m * ng));
end
for i = 1:m
    for r = 1:ng
        idx = groups{r};
        G(idx, (i - 1) * ng + r) = phi(idx, i);
    end
end
end

function U = block_op(u, groups, ng)
%BLOCK_OP 式 (10.46) 的分块振型算子，第 r 列只在第 r 组行块上等于 u 的对应分量。
%
%   与 assemble_G 同理：复振型按复值分配以保留虚部，实振型按实值分配。
u = u(:);
if isreal(u)
    U = zeros(numel(u), ng);
else
    U = complex(zeros(numel(u), ng));
end
for r = 1:ng
    idx = groups{r};
    U(idx, r) = u(idx);
end
end

function [m0, B] = column_basis(G)
%COLUMN_BASIS 矩阵 G 列空间的正交基（复值矩阵按复内积正交归一）。
%
%   返回 B（n x m0，B'B = I_m0）与数值秩 m0，阈值取最大奇异值的 1e-10 倍。
%   振型为复向量时 G 为复矩阵，正交基必须按复内积给出，故此处统一用 svd 的
%   左奇异向量；写作实 QR 会丢掉列空间中的虚方向。
sv = svd(G);
if isempty(sv) || sv(1) <= 0
    m0 = 0;
    B = zeros(size(G, 1), 0);
    return;
end
m0 = sum(sv > 1e-10 * sv(1));
[Ub, ~] = svd(G, 'econ');
B = Ub(:, 1:m0);
end

function d0 = noise_diagonal(Se, groups, nch)
%NOISE_DIAGONAL 把各测组的预测误差谱密度铺成逐通道的对角元。
Se = Se(:);
if numel(Se) ~= numel(groups)
    error('bayoma:asynchronous:badSe', '预测误差谱密度的个数应等于测组数');
end
d0 = zeros(nch, 1);
for r = 1:numel(groups)
    d0(groups{r}) = Se(r);
end
end

function vent = v_entries(m, ng)
%V_ENTRIES 相干矩阵自由元的编号（阶对 i<j 与组对 r<s 的笛卡尔积）。
vent = zeros(0, 4);
for i = 1:m - 1
    for j = i + 1:m
        for r = 1:ng - 1
            for s = r + 1:ng
                vent(end + 1, :) = [i, j, r, s]; %#ok<AGROW>
            end
        end
    end
end
end

function vec = pack_v(Vc, vent)
%PACK_V 把各相干矩阵的非对角元打包为 [模; 相位] 交替排列的实向量。
npv = size(vent, 1);
vec = zeros(2 * npv, 1);
for t = 1:npv
    z = Vc{vent(t, 1), vent(t, 2)}(vent(t, 3), vent(t, 4));
    vec(2 * t - 1) = abs(z);
    vec(2 * t) = angle(z);
end
end

function Vc = assemble_V(vec, vent, m, ng)
%ASSEMBLE_V 由模与相位参数构造对角元为 1 的 Hermitian 相干矩阵（式 (10.51)）。
Vc = cell(m, m);
for i = 1:m
    for j = 1:m
        Vc{i, j} = eye(ng);
    end
end
npv = size(vent, 1);
for t = 1:npv
    i = vent(t, 1);
    j = vent(t, 2);
    r = vent(t, 3);
    s = vent(t, 4);
    z = vec(2 * t - 1) * exp(1i * vec(2 * t));
    Vc{i, j}(r, s) = z;
    Vc{i, j}(s, r) = conj(z);
end
for i = 1:m - 1
    for j = i + 1:m
        Vc{j, i} = Vc{i, j}';
    end
end
end

function s = Vc_to_struct(Vc, m)
%VC_TO_STRUCT 相干矩阵元胞转结构体，字段名 v12、v13、…
s = struct();
for i = 1:m - 1
    for j = i + 1:m
        s.(sprintf('v%d%d', i, j)) = Vc{i, j};
    end
end
end

function Vc = struct_to_Vc(s, m, groups)
%STRUCT_TO_VC 相干矩阵结构体转元胞，缺省为对角元 1 的单位矩阵。
%
%   字段名 v12、v13、… 给出 i<j 的相干矩阵；字段名 v11、v22、… 给出对角块
%   V_ii。式 (10.45) 的 V_ii 在一般情形下是对角元为 1 的 Hermitian 相关矩阵，
%   式 (10.48) 的紧凑形式允许把它并入 S_v 的对角分块。多阶识别时对角块固定为
%   I_q（规范），故缺省取单位矩阵；给出 vII 时按其组装，供 m = 1 时与式
%   (10.45) 的单阶形式（即异步单模态识别的 v）逐元素对照。
ng = numel(groups);
Vc = cell(m, m);
for i = 1:m
    for j = 1:m
        Vc{i, j} = eye(ng);
    end
end
if isstruct(s)
    f = fieldnames(s);
    for k = 1:numel(f)
        nm = f{k};
        tok = regexp(nm, '^v(\d+?)(\d+)$', 'tokens', 'once');
        if ~isempty(tok)
            i = str2double(tok{1});
            j = str2double(tok{2});
            if isfinite(i) && isfinite(j) && i >= 1 && j <= m && i <= j
                if i == j
                    % 对角块必须为 Hermitian：只取 Hermitian 部分，避免舍入使
                    % 组装出的 E_k 不满足 ishermitian、进而 e ig 不正交
                    Vc{i, i} = (s.(nm) + s.(nm)') / 2;
                else
                    Vc{i, j} = s.(nm);
                end
            end
        end
    end
end
for i = 1:m - 1
    for j = i + 1:m
        % V_ji = V_ij^H（共轭转置），不能写作共轭（V_ij.'），否则 E_k 非 Hermitian
        Vc{j, i} = Vc{i, j}';
    end
end
end

% ======================================================================
% 数值工具
% ======================================================================
function [ok, logdetTot, quadTot] = batched_chol(Q3, B)
%BATCHED_CHOL 沿频率维批量求解 Q_k = L_k L_k' 的 Cholesky 分解与三角回代。
%
% Q3 为 m0 x m0 x Nf 的 Hermitian 正定矩阵序列，B 为 m0 x Nf 的右端项。返回
% sum_k ln det Q_k 与 sum_k |L_k \ B_k|^2。分解与回代都用按频率维向量化的标量
% 公式实现：每个 m0 步只作少量长度为 Nf 的向量运算，避免逐频率调用 chol 的
% 解释开销（逐频率调用时单次似然评估的耗时约为批量实现的十余倍）。
[m0, ~, nf] = size(Q3);
ok = true;
logdetTot = 0;
quadTot = 0;
L = complex(zeros(m0, m0, nf));
for i = 1:m0
    if i == 1
        d = reshape(Q3(1, 1, :), 1, nf);
    else
        Li = L(i, 1:i - 1, :);
        d = reshape(Q3(i, i, :), 1, nf) - reshape(sum(conj(Li) .* Li, 2), 1, nf);
    end
    d = real(d);
    if any(d <= 0) || any(~isfinite(d))
        ok = false;
        return;
    end
    sq = sqrt(d);
    L(i, i, :) = reshape(sq, 1, 1, nf);
    logdetTot = logdetTot + 2 * sum(log(sq));
    for j = i + 1:m0
        if i == 1
            s = reshape(Q3(j, i, :), 1, nf);
        else
            s = reshape(Q3(j, i, :), 1, nf) ...
                - reshape(sum(conj(L(i, 1:i - 1, :)) .* L(j, 1:i - 1, :), 2), 1, nf);
        end
        L(j, i, :) = reshape(s ./ sq, 1, 1, nf);
    end
end

x = complex(zeros(m0, nf));
for i = 1:m0
    acc = B(i, :);
    for k = 1:i - 1
        acc = acc - reshape(L(i, k, :), 1, nf) .* x(k, :);
    end
    x(i, :) = acc ./ reshape(L(i, i, :), 1, nf);
end
quadTot = sum(sum(abs(x).^2));
end

function Z = tangent_basis(a, forceComplex)
%TANGENT_BASIS 单位向量 A 处单位球面的切空间正交基（Householder 构造）。
%
%   返回 Z，使 (a + Z*theta) 归一化后覆盖 a 邻域的单位球面。
%
%   实向量情形：Z 为 n x (n-1)，切空间维数 n-1。
%   复向量情形：把 a 按实部、虚部展开为 R^(2n) 的实单位向量，Householder 给出
%   2n-1 个正交方向，再剔除整体相位方向 [-Im a; Re a]（该方向对应
%   a -> a*exp(1i*phi)，似然不变，可由 S 的第 i 行、列相位吸收，属规范自由度），
%   经 Gram-Schmidt 后得到 2n-2 列，返回 Z（n x (2n-2)）。若不剔除相位方向，
%   后验 Hessian 在该方向的曲率为零，方差发散。
%
%   FORCECOMPLEX 为真时即便 a 当前恰为实向量也按复向量构造（2n-2 维）。识别
%   过程中振型可能先从实初值出发，若按 isreal 判定，切空间只含 n-1 个实方向，
%   第一步就退化为实形状类，复形状永远进不去。
if nargin < 2 || isempty(forceComplex)
    forceComplex = ~isreal(a);
end
a = a(:);
n = numel(a);
if forceComplex
    if any(imag(a) ~= 0)
        a = a / norm(a);
    end
elseif ~isreal(a)
    % 未显式要求复形状却收到复向量：仍取共轭转置意义上的归一化，并按实向量
    % 处理会丢信息，故此时强制走复分支
    forceComplex = true;
end
if forceComplex
    m = 2 * n;
    u = [real(a); imag(a)];
    nu = norm(u);
    if nu <= 1e-14
        Z = zeros(n, 0);
        return;
    end
    u = u / nu;
    if m <= 2
        Z = zeros(n, 0);
        return;
    end
    e1 = zeros(m, 1);
    e1(1) = 1;
    v = e1 - u;
    nv = norm(v);
    if nv <= 1e-14
        e1 = zeros(m, 1);
        e1(min(2, m)) = 1;
        v = e1 - u;
        nv = norm(v);
    end
    v = v / nv;
    Hm = eye(m) - 2 * (v * v.');
    Zr = Hm(:, 2:m);                       % m x (m-1)，各列与 u 正交
    ph = [-imag(a); real(a)];
    nph = norm(ph);
    if nph > 1e-12
        ph = ph / nph;
        Zr = Zr - ph * (ph.' * Zr);
        [Q, ~] = qr(Zr, 0);
        Zr = Q(:, 1:m - 2);
    end
    Z = Zr(1:n, :) + 1i * Zr(n + 1:2 * n, :);
    return;
end
if n <= 1
    Z = zeros(n, 0);
    return;
end
e1 = zeros(n, 1);
e1(1) = 1;
v = e1 - a;
nv = norm(v);
if nv <= 1e-14
    e1 = zeros(n, 1);
    e1(min(2, n)) = 1;
    v = e1 - a;
    nv = norm(v);
end
v = v / nv;
Hm = eye(n) - 2 * (v * v.');
Z = Hm(:, 2:n);
end

function v = offdiag_vec(S)
%OFFDIAG_VEC 提取严格上三角元素（按 i<j 行优先）。
m = size(S, 1);
v = zeros(1, m * (m - 1) / 2);
k = 0;
for i = 1:m - 1
    for j = i + 1:m
        k = k + 1;
        v(k) = S(i, j);
    end
end
end

function nr = nr_of(groups)
%NR_OF 各测组的通道数。
nr = cellfun(@numel, groups);
nr = nr(:);
end

function nr = nrv_of(groups)
%NRV_OF 各测组的通道数（行向量）。
nr = cellfun(@numel, groups);
nr = nr(:);
end

function val = guard(val)
%GUARD 把非有限的目标函数值替换为惩罚值，供优化器使用。
if ~isfinite(val)
    val = 1e300;
end
end

function p = prctile_local(x, pct)
%PRCTILE_LOCAL 线性插值分位数，避免依赖统计工具箱的 prctile。
%
%   分位数在噪声谱初值估计中用于取静默段水平。按升序排序后以 (n-1)*pct/100
%   定位并线性插值，与 numpy.percentile 的缺省插值方式一致。
v = sort(x(:));
n = numel(v);
if n == 0
    p = NaN;
elseif n == 1
    p = v(1);
else
    pos = (n - 1) * pct / 100;
    lo = floor(pos);
    hi = ceil(pos);
    w = pos - lo;
    p = (1 - w) * v(lo + 1) + w * v(hi + 1);
end
end

function H = numerical_hessian(func, y, steps)
%NUMERICAL_HESSIAN 中心差分的 Hessian 矩阵。
%
%   步长逐坐标自适应：若在给定步长下某个方向的单侧扰动越出可行域（似然返回
%   non-finite，例如振型切空间坐标越出单位球、参数越过上下界），就把该坐标的
%   步长折半重试，最多十次。固定步长时这种越界会让差分出现 inf-Inf 的 NaN，
%   进而使 Hessian 无法求逆（实测低噪声算例在 z 接近下界时触发）。
n = numel(y);
H = zeros(n, n);
f0 = func(y);
if ~isfinite(f0)
    H(:) = nan;
    return;
end
h = steps(:);
fp = nan(n, 1);
fm = nan(n, 1);
for i = 1:n
    hi = h(i);
    for t = 1:10
        yp = y; yp(i) = yp(i) + hi;
        ym = y; ym(i) = ym(i) - hi;
        vp = func(yp);
        vm = func(ym);
        if isfinite(vp) && isfinite(vm)
            fp(i) = vp;
            fm(i) = vm;
            break;
        end
        hi = hi / 2;
    end
    if ~isfinite(fp(i)) || ~isfinite(fm(i))
        % 该方向两侧都不可行：用极小的固定步长与有限惩罚，避免 NaN 传播
        hi = max(hi, 1e-12);
        yp = y; yp(i) = yp(i) + hi;
        ym = y; ym(i) = ym(i) - hi;
        vp = func(yp);
        vm = func(ym);
        if ~isfinite(vp), vp = f0 + 1e6; end
        if ~isfinite(vm), vm = f0 + 1e6; end
        fp(i) = vp;
        fm(i) = vm;
    end
    h(i) = hi;
    H(i, i) = (fp(i) - 2 * f0 + fm(i)) / hi^2;
end
for i = 1:n - 1
    for j = i + 1:n
        ypp = y; ypp(i) = ypp(i) + h(i); ypp(j) = ypp(j) + h(j);
        ypm = y; ypm(i) = ypm(i) + h(i); ypm(j) = ypm(j) - h(j);
        ymp = y; ymp(i) = ymp(i) - h(i); ymp(j) = ymp(j) + h(j);
        ymm = y; ymm(i) = ymm(i) - h(i); ymm(j) = ymm(j) - h(j);
        a = func(ypp);
        b = func(ypm);
        c = func(ymp);
        d = func(ymm);
        if ~all(isfinite([a, b, c, d]))
            H(i, j) = 0;
            H(j, i) = 0;
            continue;
        end
        val = (a - b - c + d) / (4 * h(i) * h(j));
        H(i, j) = val;
        H(j, i) = val;
    end
end
if any(~isfinite(H(:)))
    H(~isfinite(H)) = 0;
end
end

function cov = regularised_inverse(H)
%REGULARISED_INVERSE 对称矩阵的正则化逆。
%
%   数值微分的 Hessian 在弱可辨识方向上的特征值接近零，直接求逆会把舍入误差
%   放大成虚假的巨大方差；带截断阈值的伪逆则把这类方向判成零方差，给出参数
%   精确已知的错误结论。这里按最大特征值取相对下限，只把非正与近零特征值抬到
%   该下限，使弱可辨识方向得到有限的、量级合理的大方差。
H = (H + H.') / 2;
[V, D] = eig(H);
w = real(diag(D));
wmax = max(abs(w));
if ~isfinite(wmax) || wmax <= 0
    error('bayoma:asynchronous:badHessian', '负对数似然的 Hessian 非正定');
end
floor_ = wmax * 1e-12;
w = max(w, floor_);
cov = V * diag(1 ./ w) * V.';
cov = (cov + cov.') / 2;
end

function groups = validate_groups(groups, nch)
%VALIDATE_GROUPS 校验测组划分并统一为索引列向量的元胞数组。
if isnumeric(groups)
    groups = {groups(:)};
end
if ~iscell(groups) || isempty(groups)
    error('bayoma:asynchronous:badGroups', '测组划分必须是索引向量的元胞数组');
end
out = cell(numel(groups), 1);
used = false(nch, 1);
for r = 1:numel(groups)
    idx = double(groups{r}(:));
    if isempty(idx)
        error('bayoma:asynchronous:emptyGroup', '第 %d 个测组不含任何通道', r);
    end
    if any(idx < 1) || any(idx > nch) || any(idx ~= round(idx))
        error('bayoma:asynchronous:badIndex', '第 %d 个测组的通道索引越界', r);
    end
    if any(used(idx))
        error('bayoma:asynchronous:overlap', '通道 %d 同时属于多个测组', find(used(idx), 1));
    end
    used(idx) = true;
    out{r} = idx;
end
if ~all(used)
    error('bayoma:asynchronous:incomplete', '测组划分必须覆盖全部通道');
end
groups = out;
end

function s = fill_defaults(s, d)
%FILL_DEFAULTS 用缺省值补齐结构体字段。
names = fieldnames(d);
for i = 1:numel(names)
    if ~isfield(s, names{i}) || isempty(s.(names{i}))
        s.(names{i}) = d.(names{i});
    end
end
end
