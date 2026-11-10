"""异步数据的单模态贝叶斯识别（课本第 10.4 节，式 10.42–10.51）。

同一测组内各通道同步采集、不同测组之间不同步时，组间的模态响应存在未知的时间
漂移，组间互谱谱密度不再是实值，第 10.1 节的同步模型不再适用。第 10.4 节按
测组把模态形状分块，用复相干矩阵表征组间相关性：

    u = [t_1; ...; t_q],   t_r 为第 r 组 n_r 个测点的局部振型              (10.43)

    E[eta_ik^(r) eta_jk^(s)* | theta] = h_ik h_jk* S_ij v_ijrs,           (10.45)

其中 v_ijrs 为复相干因子，|v_ijrs| <= 1，v_ijrr = 1。模态响应按组取各自的
时钟，式 (10.46) 把缩放 FFT 写成按组分块的振型与模态响应的乘积。单一模态
（m = 1）时谱矩阵为

    E_k = S D_k u~ v u~^T + diag{S_er I_{n_r}},  r = 1..q                 (10.50)

u~ 为 n x q 的分块矩阵，第 r 行的行块只有第 r 列为 t_r、其余为零；v 为 q x q
的 Hermitian 相干矩阵，对角元为 1（式 10.51）。负对数似然函数为

    L(theta) = n N_f ln(pi) + sum_k ln|E_k(theta)|
               + sum_k F_k^* E_k(theta)^{-1} F_k.                          (10.42)

参数集为 theta = {f, zeta, S, {S_er}, u (n x 1), v (q x q, 对角元为 1)}，
模态形状受单位范数约束 u^T u = 1。q = 1 时 u~ 退化为 u、v = 1、
S_e1 = S_e，问题退化为第 12 章的同步单模态识别。

实现说明
--------
1. 式 (10.42) 的两次求和按 ``E_k = D0^{1/2} (I + S D_k M) D0^{1/2}`` 改写，
   其中 ``M = D0^{-1/2} (u~ v u~^T) D0^{-1/2}``，``D0 = diag{S_er I_{n_r}}``。
   ``M`` 为 Hermitian 半正定矩阵且秩不超过 q，对其作一次特征分解，行列式与
   二次型都化为 n 个标量的累加，单次似然评估为 O(n^2 N_f)，与式 (10.42) 的
   数值结果一致，便于反复调用。
2. 局部振型的整体符号不可辨识：``t_r -> -t_r`` 同时把相干矩阵的第 r 行与
   第 r 列反号，谱矩阵不变。输出统一取「绝对值最大的分量为正」的符号约定，
   与第 12 章的同步问题一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ._util import LOG
from .single_mode import dynamic_factor as _dynamic_factor

_LOG_PI = float(np.log(np.pi))

# 阻尼比与谱密度的允许范围，与第 12 章的同步单模态问题保持一致。
_ZMIN = 1.0e-6
_ZMAX = 0.5
_SMIN = 1.0e-30
_SMAX = 1.0e30
# 复相干因子模的上界。式 (10.45) 的理论上界为 1；留出余量以保证相干矩阵
# 严格正定，避免谱矩阵在优化过程中接近奇异。
_VMAX = 0.99


@dataclass
class AsynchronousSetup:
    """一个频带内的异步数据集合。

    Attributes
    ----------
    freqs : ndarray, shape (Nf,)
        频带内各 FFT 点的频率（Hz）。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT，按列存放各频率点的 ``F_k``。异步数据保留复值本身的组间相位
        信息，不做共轭相乘或取实部，故此处直接保存 ``F_k``。
    groups : list of list of int
        测组划分，``groups[r]`` 为第 r 组各通道的 0 基索引。各组互不重叠且覆盖
        全部通道，``n = sum_r n_r``。
    """

    freqs: np.ndarray
    fft: np.ndarray
    groups: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.freqs = np.asarray(self.freqs, dtype=float).reshape(-1)
        self.fft = np.ascontiguousarray(np.asarray(self.fft, dtype=complex))
        self.groups = _validate_groups(self.groups, int(self.fft.shape[0]))
        if self.fft.shape[1] != self.freqs.size:
            raise ValueError("fft 的列数必须等于频率点数")

    @classmethod
    def from_spectral_data(cls, data, groups) -> "AsynchronousSetup":
        """由 :class:`bayoma.data.SpectralData` 与测组划分构造。"""
        return cls(freqs=data.freqs, fft=data.fft, groups=groups)

    @property
    def nch(self) -> int:
        """测点总数 n。"""
        return int(self.fft.shape[0])

    @property
    def nf(self) -> int:
        """频带内 FFT 点数 N_f。"""
        return int(self.freqs.size)

    @property
    def n_group(self) -> int:
        """测组数 q。"""
        return len(self.groups)

    @property
    def d_total(self) -> float:
        """``sum_k F_k^* F_k``。"""
        return float(np.sum(np.abs(self.fft) ** 2))


@dataclass
class AsynchronousResult:
    """异步单模态识别的输出。

    Attributes
    ----------
    f : float
        固有频率（Hz）。
    z : float
        阻尼比。
    S : float
        模态力谱密度。
    Se : ndarray, shape (q,)
        各测组的预测误差（通道噪声）谱密度 ``S_er``。
    phi : ndarray, shape (n,)
        模态形状 ``u`` 的全部分量，欧氏范数为 1；第 r 组的局部振型为
        ``t_r = u[groups[r]]``。
    groups : list of list of int
        测组划分。
    coherence : ndarray, shape (q, q), complex
        复相干矩阵 ``v``，Hermitian 且对角元为 1。
    nllf : float
        极小化后的负对数似然函数值，式 (10.42)。
    n_iter : int
        交替极小化的外层迭代次数。
    converged : bool
        是否满足收敛判据。
    """

    f: float
    z: float
    S: float
    Se: np.ndarray
    phi: np.ndarray
    groups: list
    coherence: np.ndarray
    nllf: float
    n_iter: int
    converged: bool

    @property
    def local(self) -> list:
        """各测组的局部振型 ``t_r``。"""
        u = np.asarray(self.phi, dtype=float)
        return [u[np.asarray(g, dtype=int)] for g in self.groups]


def _validate_groups(groups, nch: int) -> list:
    """校验测组划分并统一为整数数组列表。"""
    if groups is None:
        raise ValueError("必须给出测组划分 groups")
    out = []
    used: list[int] = []
    for r, idx in enumerate(groups):
        arr = np.asarray(idx, dtype=int).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"第 {r + 1} 个测组不含任何通道")
        if np.any(arr < 0) or np.any(arr >= int(nch)):
            raise ValueError(f"第 {r + 1} 个测组的通道索引越界：允许范围 0..{int(nch) - 1}")
        out.append(arr)
        used.extend(arr.tolist())
    if sorted(used) != list(range(int(nch))):
        raise ValueError("测组划分必须覆盖全部通道，且每个通道只属于一个测组")
    return out


def _dynamic(f: float, z: float, freqs: np.ndarray, q: int) -> np.ndarray:
    """式 (10.41) 的动态放大因子，零频处取 0 以避免无穷大。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        dk = _dynamic_factor(float(f), float(z), np.asarray(freqs, dtype=float), q)
    return np.where(np.isfinite(dk), dk, 0.0)


def coherence_matrix(params, q: int) -> np.ndarray:
    """由自由实参数构造对角元为 1 的 Hermitian 相干矩阵 ``v``（式 10.51）。

    Parameters
    ----------
    params : array_like
        长度 ``q(q-1)`` 的实向量，按上三角的行优先顺序依次给出各非对角元的
        实部与虚部，即 ``[Re v_12, Im v_12, Re v_13, Im v_13, ...]``。
    q : int
        测组数。

    Returns
    -------
    ndarray, shape (q, q), complex
        Hermitian 相干矩阵，对角元为 1，非对角元的模不超过 0.99。
    """
    ng = int(q)
    v = np.eye(ng, dtype=complex)
    arr = np.asarray(params, dtype=float).reshape(-1)
    need = ng * (ng - 1)
    if arr.size != need:
        raise ValueError(f"相干矩阵的自由参数应为 {need} 个，收到 {arr.size} 个")
    t = 0
    for i in range(ng):
        for j in range(i + 1, ng):
            zval = arr[t] + 1j * arr[t + 1]
            t += 2
            mod = abs(zval)
            if mod > _VMAX:
                zval = zval * (_VMAX / mod)
            v[i, j] = zval
            v[j, i] = np.conj(zval)
    return v


def _v_to_params(v: np.ndarray) -> np.ndarray:
    """相干矩阵的上三角非对角元展开为实向量，与 :func:`coherence_matrix` 互逆。"""
    v = np.asarray(v, dtype=complex)
    ng = v.shape[0]
    out = []
    for i in range(ng):
        for j in range(i + 1, ng):
            out.append(float(np.real(v[i, j])))
            out.append(float(np.imag(v[i, j])))
    return np.asarray(out, dtype=float)


def _v_from_polar(mod_phase: np.ndarray, ng: int) -> np.ndarray:
    """由「模、相位」参数对构造相干矩阵，供有界优化使用。"""
    v = np.eye(ng, dtype=complex)
    arr = np.asarray(mod_phase, dtype=float).reshape(-1)
    t = 0
    for i in range(ng):
        for j in range(i + 1, ng):
            mod = min(max(float(arr[t]), 0.0), _VMAX)
            ph = float(arr[t + 1])
            t += 2
            v[i, j] = mod * np.exp(1j * ph)
            v[j, i] = np.conj(v[i, j])
    return v


def _v_to_polar(v: np.ndarray) -> np.ndarray:
    """相干矩阵的「模、相位」参数展开，与 :func:`_v_from_polar` 互逆。"""
    v = np.asarray(v, dtype=complex)
    ng = v.shape[0]
    out = []
    for i in range(ng):
        for j in range(i + 1, ng):
            out.append(float(abs(v[i, j])))
            out.append(float(np.angle(v[i, j])))
    return np.asarray(out, dtype=float)


def block_operator(u: np.ndarray, groups) -> np.ndarray:
    """构造式 (10.50) 的分块振型矩阵 ``u~``，形状 ``(n, q)``。

    第 r 个列向量的第 r 组行块等于局部振型 ``t_r``，其余行块为零。
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    grp = _validate_groups(groups, u.size)
    uop = np.zeros((u.size, len(grp)), dtype=float)
    for r, idx in enumerate(grp):
        uop[idx, r] = u[idx]
    return uop


def noise_diagonal(se_list, groups) -> np.ndarray:
    """把各测组的预测误差谱密度铺成 ``n x 1`` 的对角元。"""
    grp = _validate_groups(groups, int(np.sum([np.asarray(g).size for g in groups])))
    se = np.asarray(se_list, dtype=float).reshape(-1)
    if se.size != len(grp):
        raise ValueError(f"预测误差谱密度应为 {len(grp)} 个，收到 {se.size} 个")
    if np.any(se <= 0.0) or np.any(~np.isfinite(se)):
        raise ValueError("预测误差谱密度必须为正的有限数")
    d0 = np.empty(int(np.sum([g.size for g in grp])))
    for r, idx in enumerate(grp):
        d0[idx] = se[r]
    return d0


def theoretical_psd_block(
    f: float,
    z: float,
    S: float,
    Se_list,
    u: np.ndarray,
    v: np.ndarray,
    freqs: np.ndarray,
    groups,
    q: int = 0,
) -> np.ndarray:
    """按式 (10.50) 计算各频率点的理论谱矩阵 ``E_k``。

    Parameters
    ----------
    f, z : float
        固有频率（Hz）与阻尼比。
    S : float
        模态力谱密度。
    Se_list : array_like, shape (q,)
        各测组的预测误差谱密度 ``S_er``。
    u : ndarray, shape (n,)
        模态形状的全部分量。
    v : ndarray, shape (q, q), complex
        复相干矩阵，Hermitian 且对角元为 1。
    freqs : ndarray, shape (Nf,)
        频率坐标（Hz）。
    groups : list of list of int
        测组划分。
    q : int, optional
        数据类型指数，``0`` 加速度、``1`` 速度、``2`` 位移（式 10.41）。

    Returns
    -------
    ndarray, shape (n, n, Nf), complex
        Hermitian 谱矩阵序列。
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    grp = _validate_groups(groups, u.size)
    vv = np.asarray(v, dtype=complex)
    if vv.shape != (len(grp), len(grp)):
        raise ValueError(f"相干矩阵应为 {len(grp)} x {len(grp)}，收到 {vv.shape}")
    dk = _dynamic(f, z, freqs, q)
    uop = block_operator(u, grp)
    base = uop @ vv @ uop.T  # n x n，Hermitian
    d0 = noise_diagonal(Se_list, grp)

    n = u.size
    out = (float(S) * dk)[None, None, :] * base[:, :, None]
    idx = np.arange(n)
    out[idx, idx, :] += d0[:, None]
    return out


def nllf(
    f: float,
    z: float,
    S: float,
    se_list,
    u: np.ndarray,
    v: np.ndarray,
    freqs: np.ndarray,
    fft: np.ndarray,
    groups,
    q: int = 0,
) -> float:
    """式 (10.42) 的负对数似然函数。

    Parameters
    ----------
    f, z, S : float
        固有频率（Hz）、阻尼比与模态力谱密度。
    se_list : array_like, shape (q,)
        各测组的预测误差谱密度。
    u : ndarray, shape (n,)
        模态形状。
    v : ndarray, shape (q, q), complex
        复相干矩阵。
    freqs : ndarray, shape (Nf,)
        频率坐标（Hz）。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT。
    groups : list of list of int
        测组划分。
    q : int, optional
        数据类型指数。

    Returns
    -------
    float
        负对数似然函数值；参数越出可行域时返回 ``inf``。

    Notes
    -----
    ``E_k = D0^{1/2}(I + S D_k M)D0^{1/2}``，``M = D0^{-1/2} u~ v u~^T D0^{-1/2}``
    为 Hermitian 半正定矩阵。对 ``M = Q diag(lambda) Q^*`` 作一次特征分解，得

        ln|E_k| = sum_i ln D0_ii + sum_i ln(1 + S D_k lambda_i)
        F_k^* E_k^{-1} F_k = sum_i |(Q^* D0^{-1/2} F_k)_i|^2 / (1 + S D_k lambda_i)

    与式 (10.42) 的 ``slogdet`` 与 ``solve`` 结果一致，但避免了逐频率的矩阵
    分解，单次评估为 O(n^2 N_f)。
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    fft = np.asarray(fft, dtype=complex)
    u = np.asarray(u, dtype=float).reshape(-1)
    n, nf = fft.shape
    if freqs.size != nf or u.size != n:
        raise ValueError("频率点数与 fft 的列数、模态形状长度与 fft 的行数必须一致")
    grp = _validate_groups(groups, n)

    f = float(f)
    z = float(z)
    S = float(S)
    if not (np.isfinite(f) and f > 0.0):
        return np.inf
    if not (np.isfinite(z) and 0.0 < z < 1.0):
        return np.inf
    if not (np.isfinite(S) and S > 0.0):
        return np.inf

    try:
        d0 = noise_diagonal(se_list, grp)
    except ValueError:
        return np.inf

    vv = np.asarray(v, dtype=complex)
    uop = block_operator(u, grp)
    base = uop @ vv @ uop.T
    s0 = np.sqrt(d0)
    # M = D0^{-1/2} (u~ v u~^T) D0^{-1/2}
    mmat = base / np.outer(s0, s0)
    lam, qvec = np.linalg.eigh(0.5 * (mmat + mmat.conj().T))
    lam = np.real(lam)

    dk = _dynamic(f, z, freqs, q)
    coef = S * dk  # S D_k
    denom = 1.0 + np.outer(coef, lam)  # (Nf, n)
    if not np.all(denom > 0.0) or not np.all(np.isfinite(denom)):
        return np.inf

    g = fft / s0[:, None]  # D0^{-1/2} F_k
    hh = qvec.conj().T @ g  # (n, Nf)
    quad = float(np.sum(np.abs(hh) ** 2 / denom.T))
    logdet = float(nf * np.sum(np.log(d0)) + np.sum(np.log(denom)))
    val = n * nf * _LOG_PI + logdet + quad
    if not np.isfinite(val):
        return np.inf
    return float(val)


def _initial_guess(
    freqs: np.ndarray,
    fft: np.ndarray,
    grp: list,
    q: int,
    f0: float | None,
    z0: float,
):
    """按第 12.3.1 节的高信噪比渐近式构造初值。

    频率初值取奇异值谱峰；各测组的局部振型初值取本组样本谱矩阵之和的主特征
    向量（组内同步，该求和矩阵仍具有同步单模态的秩一结构）；模态力谱密度与
    各组预测误差谱密度由组内主特征值与迹估计。
    """
    n, nf = fft.shape
    grp = _validate_groups(grp, n)
    ng = len(grp)
    dmat = np.real(fft[:, None, :] * np.conj(fft)[None, :, :])  # D_k

    sv = np.linalg.eigvalsh(np.transpose(dmat, (2, 0, 1)))
    k_peak = int(np.argmax(sv[:, -1]))
    if f0 is None:
        f_cur = float(freqs[k_peak])
    else:
        f_cur = float(f0)
        k_peak = int(np.argmin(np.abs(freqs - f_cur)))
    z_cur = float(z0)

    u = np.zeros(n)
    peak = np.empty(ng)
    trace = np.empty(ng)
    for r, idx in enumerate(grp):
        blk = dmat[np.ix_(idx, idx)]
        amat = np.einsum("ijk->ij", blk)
        amat = 0.5 * (amat + amat.T)
        vals, vecs = np.linalg.eigh(amat)
        tvec = vecs[:, -1]
        if tvec[int(np.argmax(np.abs(tvec)))] < 0.0:
            tvec = -tvec
        u[idx] = tvec
        peak[r] = float(vals[-1])
        trace[r] = float(np.trace(amat))

    nrm = float(np.linalg.norm(u))
    u = u / max(nrm, 1.0e-300)

    dk = _dynamic(f_cur, z_cur, freqs, q)
    asum = float(np.sum(dk))
    se = np.empty(ng)
    s_est = np.empty(ng)
    for r, idx in enumerate(grp):
        nr = idx.size
        if nr > 1:
            se[r] = max((trace[r] - peak[r]) / ((nr - 1) * nf), _SMIN)
        else:
            # 单通道测组无法从组内秩信息分离噪声，取总量的一半作保守估计
            se[r] = max(0.5 * trace[r] / nf, _SMIN)
        local = u[idx]
        # E_k 的组内块为 S D_k (t_r t_r^T) + S_er I，故
        # maxeig(sum_k 块) ≈ S ||t_r||^2 sum_k D_k + S_er N_f。
        s_est[r] = max(peak[r] - se[r] * nf, 0.0) / max(
            asum * float(local @ local), 1.0e-300
        )
    S_cur = float(np.mean(s_est))
    S_cur = min(max(S_cur, _SMIN), _SMAX)

    v = np.eye(ng, dtype=complex)
    return f_cur, z_cur, S_cur, se, u, v


def _fix_signs(u: np.ndarray, v: np.ndarray, grp: list):
    """固定各测组局部振型的符号：绝对值最大的分量为正。

    局部振型的整体符号不可辨识：``t_r -> -t_r`` 同时把相干矩阵第 r 行与第 r
    列反号，式 (10.50) 的谱矩阵不变。统一符号后，同一模态在不同数据组、不同
    程序实现之间可直接比较。
    """
    u = np.array(u, dtype=float, copy=True)
    v = np.array(v, dtype=complex, copy=True)
    for r, idx in enumerate(grp):
        loc = u[idx]
        k = int(np.argmax(np.abs(loc)))
        if loc[k] < 0.0:
            u[idx] = -loc
            v[r, :] *= -1.0
            v[:, r] *= -1.0
    v[np.diag_indices(v.shape[0])] = 1.0
    return u, v


def mpv(
    freqs: np.ndarray,
    fft: np.ndarray,
    groups,
    q: int = 0,
    f0: float | None = None,
    z0: float = 0.01,
    max_iter: int = 200,
    tol: float = 1e-8,
):
    """异步数据的最可能值（MPV），按第 10.4 节与第 12.3 节交替极小化。

    振动型相关参数 ``(u, v)`` 与谱参数 ``(f, zeta, S, {S_er})`` 分块交替优化：
    固定 ``(u, v)`` 时在自然对数空间对谱参数作有界拟牛顿极小化；固定谱参数时
    在模态形状的切空间内搜索 ``u``（切空间基与 ``u`` 正交，自动满足单位范数
    约束），同时以「模、相位」参数更新相干矩阵的非对角元，模约束在
    ``[0, 0.99]`` 内。

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
        频带内频率坐标（Hz）。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT。
    groups : list of list of int
        测组划分。
    q : int, optional
        数据类型指数，``0`` 加速度、``1`` 速度、``2`` 位移。
    f0 : float, optional
        频率初值；缺省取奇异值谱峰。
    z0 : float, optional
        阻尼比初值，默认 0.01。
    max_iter : int, optional
        外层交替迭代上限，默认 200。
    tol : float, optional
        收敛容差，用于负对数似然函数的相对变化与参数步长。

    Returns
    -------
    AsynchronousResult
    """
    from .posterior import tangent_basis

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    fft = np.ascontiguousarray(np.asarray(fft, dtype=complex))
    n, nf = fft.shape
    if freqs.size != nf:
        raise ValueError("频率点数必须等于 fft 的列数")
    grp = _validate_groups(groups, n)
    ng = len(grp)

    f_cur, z_cur, S_cur, se_cur, u, v = _initial_guess(freqs, fft, grp, q, f0, z0)
    x = np.log(np.concatenate([[f_cur, z_cur, S_cur], se_cur]))

    # 谱密度的上下界按数据量纲取：缩放 FFT 的平均谱能量给出 S 与 S_er 的自然
    # 尺度，相差八个数量级足以覆盖实际识别范围。若改用 1e-30~1e30 这类与数据
    # 无关的定界，有界拟牛顿法的线搜索会因搜索区间与曲率尺度失配而在初值处
    # 直接报告收敛，谱参数子问题不再前进。
    d_scale = float(np.mean(np.abs(fft) ** 2))
    d_scale = d_scale if np.isfinite(d_scale) and d_scale > 0.0 else 1.0
    s_lo = float(np.log(max(d_scale * 1.0e-8, _SMIN)))
    s_hi = float(np.log(min(d_scale * 1.0e8, _SMAX)))

    f_lo = max(float(freqs[0]) * 0.5, 1.0e-12)
    f_hi = float(freqs[-1]) * 2.0
    lb = np.concatenate([[np.log(f_lo), np.log(_ZMIN), s_lo], np.full(ng, s_lo)])
    ub = np.concatenate([[np.log(f_hi), np.log(_ZMAX), s_hi], np.full(ng, s_hi)])

    def unpack_spec(xv):
        return float(np.exp(xv[0])), float(np.exp(xv[1])), float(np.exp(xv[2])), np.exp(xv[3:])

    def spec_obj(xv):
        fv, zv, Sv, sev = unpack_spec(xv)
        val = nllf(fv, zv, Sv, sev, u, v, freqs, fft, grp, q)
        return val if np.isfinite(val) else 1.0e300

    nllf_val = nllf(f_cur, z_cur, S_cur, se_cur, u, v, freqs, fft, grp, q)
    if not np.isfinite(nllf_val):
        raise ValueError("初值处的负对数似然函数非有限，请检查数据与测组划分")

    converged = False
    n_iter = 0
    step_tol = max(np.sqrt(tol), 1.0e-6)

    for n_iter in range(1, int(max_iter) + 1):
        nllf_old = nllf_val
        x_old = x.copy()
        u_old = u.copy()
        v_old = v.copy()

        # ---- 谱参数子问题：自然对数空间的有界拟牛顿极小化 ----
        res = minimize(
            spec_obj,
            x,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxiter": 200, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
        )
        if np.all(np.isfinite(res.x)):
            x = res.x
        f_cur, z_cur, S_cur, se_cur = unpack_spec(x)

        # ---- 模态形状与相干矩阵子问题：切空间参数化 ----
        base = tangent_basis(u)
        ntan = int(base.shape[1])
        npair = ng * (ng - 1) // 2
        y0 = np.concatenate([np.zeros(ntan), _v_to_polar(v)])
        lb_y = np.concatenate([np.full(ntan, -1.0), np.tile([0.0, -np.pi], npair)])
        ub_y = np.concatenate([np.full(ntan, 1.0), np.tile([_VMAX, np.pi], npair)])
        u_base = u.copy()

        def shape_obj(yv):
            theta = yv[:ntan]
            unew = u_base + base @ theta
            nrm = float(np.linalg.norm(unew))
            if nrm <= 1.0e-12:
                return 1.0e300
            unew = unew / nrm
            vnew = _v_from_polar(yv[ntan:], ng)
            val = nllf(f_cur, z_cur, S_cur, se_cur, unew, vnew, freqs, fft, grp, q)
            return val if np.isfinite(val) else 1.0e300

        res2 = minimize(
            shape_obj,
            y0,
            method="L-BFGS-B",
            bounds=list(zip(lb_y, ub_y)),
            options={"maxiter": 200, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
        )
        if np.all(np.isfinite(res2.x)):
            y = res2.x
        else:
            y = y0
        unew = u_base + base @ y[:ntan]
        unew = unew / max(float(np.linalg.norm(unew)), 1.0e-300)
        v = _v_from_polar(y[ntan:], ng)
        u = unew

        nllf_val = nllf(f_cur, z_cur, S_cur, se_cur, u, v, freqs, fft, grp, q)
        if not np.isfinite(nllf_val):
            # 数值退化时退回上一步的可行点
            u = u_old
            v = v_old
            x = x_old
            f_cur, z_cur, S_cur, se_cur = unpack_spec(x)
            nllf_val = nllf_old
            break

        rel_l = abs(nllf_old - nllf_val) / max(abs(nllf_old), 1.0)
        step_spec = float(
            np.max(
                np.abs(
                    [
                        (f_cur - np.exp(x_old[0])) / max(f_cur, 1.0e-300),
                        (z_cur - np.exp(x_old[1])) / max(z_cur, 1.0e-300),
                    ]
                )
            )
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_rest = np.abs(x[2:] / np.where(np.abs(x_old[2:]) > 0.0, x_old[2:], 1.0) - 1.0)
        step_spec = max(step_spec, float(np.max(rel_rest)) if rel_rest.size else 0.0)
        step_shape = float(np.max(np.abs(u - u_old)))
        step_v = float(np.max(np.abs(v - v_old))) if ng > 1 else 0.0
        step = max(step_spec, step_shape, step_v)
        if rel_l < tol and step < step_tol:
            converged = True
            break

    u, v = _fix_signs(u, v, grp)
    LOG.debug(
        "异步单模态识别：f=%.6g Hz, zeta=%.6g, 测组数 %d, 迭代 %d 次, 收敛=%s",
        f_cur,
        z_cur,
        ng,
        n_iter,
        converged,
    )
    return AsynchronousResult(
        f=float(f_cur),
        z=float(z_cur),
        S=float(S_cur),
        Se=np.asarray(se_cur, dtype=float),
        phi=np.asarray(u, dtype=float),
        groups=grp,
        coherence=np.asarray(v, dtype=complex),
        nllf=float(nllf_val),
        n_iter=int(n_iter),
        converged=bool(converged),
    )


def posterior_covariance(
    result,
    freqs: np.ndarray,
    fft: np.ndarray,
    groups,
    q: int = 0,
    rel_step: float = 1e-4,
):
    """异步单模态问题的后验协方差矩阵。

    自由参数向量取

        y = [f/f_hat, z/z_hat, S/S_hat, {S_er/S_er_hat}, theta, {Re dv_ij, Im dv_ij}]

    其中 ``theta`` 为模态形状在切空间中的坐标（``u = (u_hat + Z theta)/||...||``，
    ``Z`` 与 ``u_hat`` 正交，自动满足单位范数约束 ``u^T u = 1``），``dv_ij`` 为
    相干矩阵非对角元的实、虚部增量。谱参数按其最可能值无量纲化，使各自由参数
    在对角尺度上同量级；数值 Hessian 与正则化求逆沿用
    :func:`bayoma.posterior._hessian` 与 :func:`bayoma.posterior._regularised_inverse`。

    Parameters
    ----------
    result : AsynchronousResult
        ``mpv`` 的返回结果。
    freqs : ndarray, shape (Nf,)
        频带内频率坐标（Hz）。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT。
    groups : list of list of int
        测组划分。
    q : int, optional
        数据类型指数。
    rel_step : float, optional
        无量纲坐标中的数值微分步长，默认 ``1e-4``。

    Returns
    -------
    cov : ndarray, shape (p, p)
        自由参数的协方差矩阵，参数顺序见上；谱参数块的右端对应变异系数的平方。
    coefv : dict
        各参数的变异系数，键为 ``f``、``z``、``S``、``Se``（列表，按测组）、
        ``phi``（模态形状全部自由度方差之和的平方根）、``v``（相干矩阵非对角元
        方差之和的平方根）。
    """
    from .posterior import _hessian, _regularised_inverse, tangent_basis

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    fft = np.asarray(fft, dtype=complex)
    n = int(fft.shape[0])
    grp = _validate_groups(groups, n)
    ng = len(grp)

    f_hat = float(result.f)
    z_hat = float(result.z)
    S_hat = float(result.S)
    se_hat = np.asarray(result.Se, dtype=float).reshape(-1)
    u_hat = np.asarray(result.phi, dtype=float).reshape(-1)
    u_hat = u_hat / max(float(np.linalg.norm(u_hat)), 1.0e-300)
    v_hat = np.asarray(result.coherence, dtype=complex)

    scale = np.concatenate([[f_hat, z_hat, S_hat], se_hat])
    nspec = int(scale.size)
    base = tangent_basis(u_hat)
    ntan = int(base.shape[1])
    pairs = [(i, j) for i in range(ng) for j in range(i + 1, ng)]
    nfree = nspec + ntan + 2 * len(pairs)

    def expand(yv):
        spec = yv[:nspec] * scale
        theta = yv[nspec : nspec + ntan]
        u = u_hat + base @ theta
        u = u / max(float(np.linalg.norm(u)), 1.0e-300)
        v = np.eye(ng, dtype=complex)
        for t, (i, j) in enumerate(pairs):
            v[i, j] = yv[nspec + ntan + 2 * t] + 1j * yv[nspec + ntan + 2 * t + 1]
            v[j, i] = np.conj(v[i, j])
        return spec, u, v

    def nllf_free(yv):
        spec, u, v = expand(yv)
        f, z, S = float(spec[0]), float(spec[1]), float(spec[2])
        se = spec[3:]
        if np.any(se <= 0.0) or f <= 0.0 or S <= 0.0 or not (0.0 < z < 1.0):
            return 1.0e300
        val = nllf(f, z, S, se, u, v, freqs, fft, grp, q)
        return val if np.isfinite(val) else 1.0e300

    y0 = np.concatenate([np.ones(nspec), np.zeros(ntan), _v_to_params(v_hat)])
    steps = np.full(nfree, float(rel_step))
    hess = _hessian(nllf_free, y0, steps)
    cov = _regularised_inverse(hess)

    diag = np.maximum(np.diag(cov), 0.0)
    sd = np.sqrt(diag)
    coefv = {
        "f": float(sd[0]),
        "z": float(sd[1]),
        "S": float(sd[2]),
        "Se": [float(x) for x in sd[3:nspec]],
        "phi": float(np.sqrt(max(float(np.sum(diag[nspec : nspec + ntan])), 0.0))),
        "v": float(np.sqrt(max(float(np.sum(diag[nspec + ntan :])), 0.0))),
    }
    return cov, coefv


def simulate_asynchronous(
    f: float,
    z: float,
    S: float,
    Se_list,
    u: np.ndarray,
    v: np.ndarray,
    groups,
    fs: float,
    duration: float,
    seed: int | None = None,
    q: int = 0,
) -> np.ndarray:
    """按式 (10.50) 的谱矩阵合成异步环境振动数据。

    合成采用频域法：先在全频段上按式 (10.50) 构造 ``E_k``，再对每个频率点抽取
    协方差等于 ``E_k`` 的圆对称复高斯向量，最后由单边谱补全负频率半轴并逆变换
    回时程（与第 10.1 节的缩放 FFT 约定一致，``E[F_k F_k^*] = E_k``）。由此得到
    的数据，其缩放 FFT 的样本协方差在统计意义上严格等于式 (10.50) 的谱矩阵，
    适合校核识别算法与后验不确定性。组间相干矩阵的复相位对应各组之间的时间
    漂移，合成数据在时域上以记录长度为周期，这是频域合成的固有性质。

    Parameters
    ----------
    f, z, S : float
        固有频率（Hz）、阻尼比与模态力谱密度。
    Se_list : array_like, shape (q,)
        各测组的预测误差谱密度。
    u : ndarray, shape (n,)
        模态形状，欧氏范数应为 1。
    v : ndarray, shape (q, q), complex
        复相干矩阵，Hermitian 且对角元为 1。
    groups : list of list of int
        测组划分。
    fs : float
        采样频率（Hz）。
    duration : float
        记录时长（s）。
    seed : int, optional
        随机数种子。
    q : int, optional
        数据类型指数。

    Returns
    -------
    ndarray, shape (N, n)
        时程数据，按列存放各通道。
    """
    from .synth import synthesize_time_history

    fs = float(fs)
    nsam = int(round(float(duration) * fs))
    if nsam < 32:
        raise ValueError("记录时长过短，至少需要 32 个样本")
    freqs_all = np.fft.rfftfreq(nsam, d=1.0 / fs)
    psd = theoretical_psd_block(f, z, S, Se_list, u, v, freqs_all, groups, q)
    data = synthesize_time_history(psd, fs, nsam, seed=seed)
    LOG.info(
        "合成异步数据：%d 通道、%d 个测组、%d 样本、%.1f s、fs=%.4g Hz",
        data.shape[1],
        len(groups),
        data.shape[0],
        duration,
        fs,
    )
    return data
