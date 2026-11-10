function [tdata, Ek_all] = simulate_oma_data(f, z, phi, S, Se, fs, dur, opts)
%SIMULATE_OMA_DATA 按式 (10.10) 合成环境振动数据。
%
%   TDATA = bayoma.simulate_oma_data(F, Z, PHI, S, SE, FS, DUR) 生成 DUR 秒、
%   采样频率 FS 的多通道环境振动数据，TDATA 为 nt x nch 矩阵，nt = DUR*FS。
%
%   F、Z 为各阶固有频率（Hz）与阻尼比，PHI 为 nch x m 的振型矩阵（各列欧氏
%   范数应为 1），S 为 m x m 的模态力单边谱密度矩阵（Hermitian），SE 为通道
%   噪声单边谱密度。
%
%   [TDATA, EK] = bayoma.simulate_oma_data(...) 同时返回全频段的谱矩阵序列，
%   EK 为 nch x nch x nf，可用于校核。
%
%   TDATA = bayoma.simulate_oma_data(..., OPTS) 接受结构体 OPTS：
%       q      数据类型指数，0 加速度（缺省）、1 速度、2 位移
%       seed   随机数种子
%
%   合成采用频域法：先由式 (10.10) 构造各频率点的谱矩阵 E_k，再抽取圆对称
%   复高斯样本得到缩放 FFT，最后由共轭对称补全负频率半轴并逆变换回时域。
%   缩放方式与 bayoma.scaled_fft 一致，即 E[F_k F_k'] = E_k。

if nargin < 8
    opts = struct();
end
if ~isfield(opts, 'q') || isempty(opts.q)
    opts.q = 0;
end
if ~isfield(opts, 'seed') || isempty(opts.seed)
    opts.seed = 0;
end

f = f(:);
z = z(:);
nch = size(phi, 1);
nt = round(dur * fs);
if mod(nt, 2) ~= 0
    nt = nt + 1;    % 取偶数点，使单边谱与双边谱的对应关系无歧义
end
dt = 1 / fs;
nf = nt / 2 + 1;
freqs = (0:nf-1).' * fs / nt;

h = bayoma.transfer_matrix(f, z, freqs, opts.q);

rng(opts.seed);
spec = complex(zeros(nf, nch));
Ek_all = complex(zeros(nch, nch, nf));
for k = 2:nf-1
    hk = h(:, k);
    g = phi .* hk.';                 % nch x m
    Ek = g * S * g' + Se * eye(nch); % 共轭转置保证 Hermitian
    Ek = (Ek + Ek') / 2;
    Ek_all(:, :, k) = Ek;
    [Ve, De] = eig(Ek);
    dv = max(real(diag(De)), 0);
    root = Ve * diag(sqrt(dv * nt / dt));
    w = (randn(nch, 1) + 1i * randn(nch, 1)) / sqrt(2);
    spec(k, :) = (root * w).';
end

full = [spec; conj(spec(end-1:-1:2, :))];
tdata = real(ifft(full, [], 1));
end
