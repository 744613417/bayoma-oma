function out = result_separation(f, z, phi, bandIndexOfOrder, jointCov, fPositions)
%RESULT_SEPARATION 同一频带内各阶模态之间的频率区分度。
%
%   OUT = bayoma.result_separation(F, Z, PHI, BANDINDEX) 对同一频带内的模态
%   两两给出：频率间隔 DF、半功率带宽、以半功率带宽计的间隔
%   ``separation_in_bw = df / max(2 z_i f_i, 2 z_j f_j)``、振型 MAC，并据
%   ``separation_in_bw < 2`` 判定密集模态。
%
%   BANDINDEX 为长度与 F 相同的频带序号（自 0 起计）。JOINTCOV 与 FPOSITIONS
%   给出时，另取联合后验协方差中频率参数的子块，给出频率估计的相关系数。
%
%   返回结构体字段：criterion（判据表达式）、threshold（阈值）、pairs（各阶对的
%   记录）、dense_pairs（判为密集的阶对）、has_dense_modes（是否存在密集模态）。
%
%   密集模态判据的含义：两阶频率间隔与半功率带宽同量级时，二者的谱峰互相重叠，
%   单模态模型不足以描述，须用联合似然同时确定两阶参数。

DENSE_MODE_BW_THRESHOLD = 2.0;

f = double(f(:).');
z = double(z(:).');
bandIndexOfOrder = double(bandIndexOfOrder(:).');
m = numel(f);

corrMat = [];
if nargin >= 5 && ~isempty(jointCov) && nargin >= 6 && ~isempty(fPositions)
    cov = double(jointCov);
    idx = double(fPositions(:).');
    sub = cov(idx, idx);
    sd = sqrt(max(diag(sub), 0));
    denom = sd * sd.';
    corrMat = sub ./ denom;
    corrMat(~isfinite(corrMat)) = 0;
end

pairs = {};
for i = 1:m
    for j = i + 1:m
        if bandIndexOfOrder(i) ~= bandIndexOfOrder(j)
            continue;
        end
        df = abs(f(j) - f(i));
        bwI = 2 * abs(z(i)) * f(i);
        bwJ = 2 * abs(z(j)) * f(j);
        bw = max(bwI, bwJ);
        if bw > 0
            sep = df / bw;
        else
            sep = Inf;
        end
        % 判据可能取无穷（半功率带宽为零）或未定义（振型范数为零），结果文件
        % 里按 JSON null 给出，与 Python 侧把非有限值写成 None 的处理一致
        rec = struct('modes', [i - 1, j - 1], ...
            'band_index', bandIndexOfOrder(i), ...
            'df', finite_or_null(df), ...
            'half_power_bw', [bwI, bwJ], ...
            'separation_in_bw', finite_or_null(sep), ...
            'dense', logical(sep < DENSE_MODE_BW_THRESHOLD), ...
            'mac', finite_or_null(mac_value(phi(:, i), phi(:, j))));
        if ~isempty(corrMat)
            rec.f_correlation = corrMat(i, j);
        end
        pairs{end + 1} = rec;
    end
end

densePairs = {};
for k = 1:numel(pairs)
    if pairs{k}.dense
        densePairs{end + 1} = pairs{k}.modes;
    end
end

out = struct();
out.criterion = 'separation_in_bw = df / max(2 z_i f_i, 2 z_j f_j)';
out.threshold = DENSE_MODE_BW_THRESHOLD;
out.pairs = pairs;
out.dense_pairs = densePairs;
out.has_dense_modes = logical(~isempty(densePairs));
end

% ----------------------------------------------------------------------
function value = finite_or_null(x)
%FINITE_OR_NULL 非有限值转成 JSON null 占位，有限值原样返回。
%   契约里这些字段允许为 null；MATLAB 的 NaN/Inf 在编码时会报错。
if isfinite(x)
    value = double(x);
else
    value = bayoma.json_null();
end
end

% ----------------------------------------------------------------------
function val = mac_value(a, b)
%MAC_VALUE 两个振型向量之间的模态置信度判据（MAC）。
%   MAC = |a' b|^2 / (||a|| ||b||)^2，取值 0 至 1；两向量正交时为 0、
%   同向时为 1。范数为零时返回 NaN，表示该判据不适用。
a = a(:);
b = b(:);
na = norm(a);
nb = norm(b);
if na <= 0 || nb <= 0
    val = NaN;
    return;
end
val = abs(a' * b) ^ 2 / (na * nb) ^ 2;
end
