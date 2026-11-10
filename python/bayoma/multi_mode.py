"""多模态贝叶斯识别（课本第 13 章）。

实现内容
--------
1. 模态形状子空间分解 ``U = B0 alpha``（式 13.6）。``B0`` 为 ``n x m0`` 正交基，
   ``alpha`` 为 ``m0 x m`` 坐标；子空间维数 ``m0 = min(m, n)``。
2. 降维负对数似然函数（式 13.15 与 13.47 的等价形式），把 ``E_k`` 的逆与行列式
   从 ``n x n`` 降到 ``m0 x m0``::

       L = n*N_f*ln(pi) + (n-m0)*N_f*ln(S_e) + (d - d0)/S_e
           + sum_k ln|E0_k| + sum_k tr(E0_k^{-1} D'_k)

   其中 ``E0_k = alpha diag(h_k) S diag(h_k)^* alpha^H + S_e I_{m0}``（式 13.10），
   ``D'_k = B0^T D_k B0``，``d0 = sum_k tr(D'_k)``。
3. 模态力谱密度矩阵的结构化参数化（式 13.50–13.57），自动满足 Hermitian 与
   半正定要求。
4. 最可能值（MPV）按第 13.5 节的参数分组交替更新：谱参数 ``{f, z, S, S_e}``
   与振型坐标 ``alpha`` 交替极小化同一 NLLF，直至参数相对变化小于容差。

记号
----
- ``n``：测点数；``m``：频带内模态数；``m0 = min(m, n)``
- ``D_k = Re(F_k F_k^*)``（式 12.12 / 13.16）
- ``d = sum_k F_k^* F_k``（式 13.16）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ._util import LOG

_ZMIN = 1.0e-6
_ZMAX = 0.5
_SMIN = 1.0e-30


@dataclass
class MultiModeResult:
    """多模态识别的输出。

    Attributes
    ----------
    f : ndarray, shape (m,)
        各阶固有频率（Hz）。
    z : ndarray, shape (m,)
        各阶阻尼比。
    S : ndarray, shape (m, m), complex
        模态力谱密度矩阵（Hermitian）。
    Se : float
        预测误差谱密度。
    phi : ndarray, shape (n, m)
        模态形状矩阵，各列欧氏范数为 1。
    nllf : float
        极小化后的负对数似然函数值。
    n_iter : int
        外层交替迭代次数。
    converged : bool
        是否满足收敛判据。
    subspace_rank : int
        模态形状子空间维数 ``m0``。
    basis : ndarray, shape (n, m0)
        子空间正交基 ``B0``。
    alpha : ndarray, shape (m0, m), complex
        子空间坐标 ``alpha``，各列欧氏范数为 1。
    restarts : int
        多起点重启次数。
    """

    f: np.ndarray
    z: np.ndarray
    S: np.ndarray
    Se: float
    phi: np.ndarray
    nllf: float
    n_iter: int
    converged: bool
    subspace_rank: int
    basis: np.ndarray
    alpha: np.ndarray
    restarts: int = 0


def transfer_matrix(
    freqs_hz: np.ndarray, damping: np.ndarray, freqs: np.ndarray, q: int
) -> np.ndarray:
    """各阶模态在频带内的传递函数矩阵 ``h``，形状 ``(m, Nf)``，式 (10.8)。

    ``h_ik = (2*pi*i*f_k)^{-q} / (1 - beta_ik^2 - 2*i*z_i*beta_ik)``，
    ``beta_ik = f_i/f_k``。
    """
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    z = np.asarray(damping, dtype=float).reshape(-1)
    fr = np.asarray(freqs, dtype=float)
    beta = f[:, None] / fr[None, :]
    om = (2.0 * np.pi * fr)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = om ** (-q) / (1.0 - beta ** 2 - 2.0j * z[:, None] * beta)
    h[~np.isfinite(h)] = 0.0
    return h


def structured_modal_psd(params: np.ndarray, n_mode: int, scales: np.ndarray) -> np.ndarray:
    """由自由参数重建模态力谱密度矩阵 ``S``（式 13.50–13.57）。

    ``S = diag(s) S0 diag(s)``，``S0_ij = exp(i v_ij) sin(u_ij)``（``i > j``），
    对角元为 1。该参数化保证 ``S`` Hermitian 半正定，且 ``|S0_ij| <= 1``。

    Parameters
    ----------
    params : ndarray
        长度 ``m*(m-1)``：前一半为 ``u_ij``，后一半为 ``v_ij``。
    n_mode : int
        模态数 ``m``。
    scales : ndarray, shape (m,)
        各阶模态力谱密度的尺度 ``sqrt(S_ii)``。

    Returns
    -------
    ndarray, shape (m, m), complex
    """
    m = int(n_mode)
    half = m * (m - 1) // 2
    u = np.asarray(params[:half], dtype=float)
    v = np.asarray(params[half : 2 * half], dtype=float)
    corr = np.eye(m, dtype=complex)
    idx = 0
    for i in range(m):
        for j in range(i):
            amp = np.sin(u[idx])
            phase = np.exp(1j * v[idx])
            corr[i, j] = amp * phase
            corr[j, i] = amp * np.conj(phase)
            idx += 1
    scale = np.asarray(scales, dtype=float)
    return scale[:, None] * corr * scale[None, :]


def _project_psd(dhat: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, float]:
    """计算 ``D'_k = B0^T D_k B0`` 与 ``d0 = sum_k tr(D'_k)``。

    ``d0`` 仅用于式 (13.47) 的 ``(d - d0)/S_e`` 项与噪声谱初值；似然函数中依赖
    子空间的二次型必须直接用复值投影 ``F'_k = B0^T F_k``，见 :func:`_project_fft`。
    """
    red = np.einsum("ai,abk,bj->ijk", basis, dhat, basis)
    return np.ascontiguousarray(red), float(np.trace(np.einsum("ijk->ij", red)))


def _project_fft(fft: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """子空间投影后的复值缩放 FFT ``F'_k = B0^T F_k``，形状 ``(m0, Nf)``。"""
    return np.einsum("ai,ak->ik", basis, fft)


def _full_basis(basis: np.ndarray, nch: int) -> np.ndarray:
    """把 ``n × m0`` 的列正交基扩充为 ``n × n`` 的正交矩阵。

    前 ``m0`` 列即输入基，其余列为正交补空间的一组正交基。子空间旋转（式 13.22）
    需要整个 ``R^n`` 的正交基，才能把子空间转出初始子空间之外。
    """
    basis = np.asarray(basis, dtype=float)
    m0 = basis.shape[1]
    if m0 >= nch:
        return basis.copy()
    q, _ = np.linalg.qr(np.column_stack([basis, np.eye(nch)]))
    # 用输入基重新正交化前 m0 列，保证前 m0 列与输入基张成同一子空间
    q[:, :m0] = basis
    q, _ = np.linalg.qr(q)
    # 固定前 m0 列的符号，使基与输入一致
    for i in range(m0):
        if q[:, i] @ basis[:, i] < 0.0:
            q[:, i] = -q[:, i]
    return q


def _rotate_basis(bfull: np.ndarray, m0: int, angles: np.ndarray) -> np.ndarray:
    """按超角 Givens 旋转更新模态形状子空间，式 (13.22)–(13.23)。

    ``B(α) = [B0, B0⊥] R(α)``，取前 ``m0`` 列。``R(α)`` 由 ``m0 × (n - m0)`` 个
    平面旋转连乘而成，第 ``(i, j)`` 个旋转在第 ``i`` 个轴与第 ``m0 + j`` 个轴之间。
    这样 ``B(α)`` 的列始终保持正交归一，同时子空间本身可以在 ``R^n`` 中转动。
    """
    bfull = np.asarray(bfull, dtype=float)
    nch = bfull.shape[1]
    n_out = nch - m0
    if n_out <= 0:
        return bfull[:, :m0].copy()
    rot = np.eye(nch)
    k = 0
    for i in range(m0 - 1, -1, -1):
        for j in range(n_out - 1, -1, -1):
            a = float(angles[k])
            k += 1
            c, s = np.cos(a), np.sin(a)
            p = m0 + j
            g = np.eye(nch)
            g[i, i] = c
            g[i, p] = -s
            g[p, i] = s
            g[p, p] = c
            rot = rot @ g
    return (bfull @ rot)[:, :m0]


def nllf_multi(
    f: np.ndarray,
    z: np.ndarray,
    S: np.ndarray,
    Se: float,
    alpha: np.ndarray,
    freqs: np.ndarray,
    fproj: np.ndarray,
    d_total: float,
    q: int,
    nch: int,
) -> float:
    """降维负对数似然函数，式 (13.15) / (13.47)。

    式 (13.47) 的最后一项是复值二次型 ``sum_k F'^*_k E0_k^{-1} F'_k``，其中
    ``F'_k = B0^T F_k`` 是子空间投影后的复值缩放 FFT。该项必须保留 ``F_k`` 的
    相位：若改用实值 ``D'_k = B0^T Re(F_k F_k^*) B0``，则 ``1/S_e`` 之外的互谱
    虚部信息全部丢失，模态力谱密度矩阵 ``S`` 的非对角元失去约束，密集模态下
    尤其如此。

    Parameters
    ----------
    fproj : ndarray, shape (m0, Nf), complex
        子空间投影后的缩放 FFT，见 :func:`_project_fft`。

    Notes
    -----
    ``d0 = sum_k tr(D'_k) = sum_k |F'_k|^2`` 由 ``fproj`` 直接给出，故不再单独传入。
    """
    nf = freqs.size
    m0 = fproj.shape[0]
    if not np.isfinite(Se) or Se <= 0.0:
        return np.inf
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=float)
    if np.any(f <= 0.0) or np.any(z <= 0.0) or np.any(z >= 1.0):
        return np.inf
    h = transfer_matrix(f, z, freqs, q)  # (m, Nf)
    if np.any(~np.isfinite(h)):
        return np.inf

    # ah[k] = alpha * h[:, k][None, :]，形状 (Nf, m0, m)
    ah = np.einsum("ip,pk->kip", alpha, h)
    # e0[k] = ah[k] @ S @ ah[k]^H + Se I
    e0 = np.einsum("kip,pq,kjq->kij", ah, S, ah.conj()) + Se * np.eye(m0)
    sign, logdet = np.linalg.slogdet(e0)
    if np.any(sign <= 0.0) or not np.all(np.isfinite(logdet)):
        return np.inf
    try:
        inv = np.linalg.inv(e0)
    except np.linalg.LinAlgError:
        return np.inf
    # sum_k F'^*_k E0_k^{-1} F'_k（实值）
    quad = np.einsum("ik,kij,jk->k", fproj.conj(), inv, fproj)
    quad_sum = float(np.sum(quad.real))
    d0 = float(np.sum(np.abs(fproj) ** 2))

    total = nch * nf * np.log(np.pi) + (nch - m0) * nf * np.log(Se)
    total += (d_total - d0) / Se
    total += float(np.sum(logdet)) + quad_sum
    return float(total)


def _initial_basis(dhat: np.ndarray, m0: int) -> np.ndarray:
    """子空间基初值：``A0 = sum_k D_k`` 的前 ``m0`` 个最大特征向量（第 13.6.1 节）。"""
    a0 = np.einsum("ijk->ij", dhat)
    a0 = 0.5 * (a0 + a0.T)
    vals, vecs = np.linalg.eigh(a0)
    order = np.argsort(vals)[::-1]
    return vecs[:, order[:m0]]


def _sv_curves(dhat: np.ndarray) -> np.ndarray:
    """奇异值谱：各频点样本谱矩阵特征值按降序排列，形状 ``(n, Nf)``。"""
    nch, _, nf = dhat.shape
    out = np.empty((nch, nf))
    for k in range(nf):
        out[:, k] = np.linalg.eigvalsh(dhat[:, :, k])[::-1]
    return out


def _initial_frequencies(
    freqs: np.ndarray, dhat: np.ndarray, n_mode: int
) -> np.ndarray:
    """频率初值：在第一条奇异值谱曲线上取显著谱峰（第 7.3 节）。

    共振带内第一条奇异值谱曲线高于噪声基线的谱峰对应各阶模态。对第一条曲线
    作峰提取（按突出度排序取前 ``n_mode`` 个），比逐条曲线取最大值稳健：后者
    在后继曲线的峰值只是噪声起伏时会把初值放到无关频点上。

    当谱峰数目不足（密集模态在奇异值谱上并成一个峰）时，在突出度最大的谱峰
    两侧按半功率带宽补足，保证初值互不重合。
    """
    from scipy.signal import find_peaks

    nch = dhat.shape[0]
    nf = freqs.size
    sv = _sv_curves(dhat)
    main = sv[0]

    df = float(freqs[1] - freqs[0]) if nf > 1 else 0.0
    span = float(freqs[-1] - freqs[0])
    # 峰间隔下限取半功率带宽的三分之一与 3 个频率分辨率中的较大者
    min_sep = max(3.0 * df, 0.02 * span)
    peaks, props = find_peaks(
        main,
        prominence=1.0e-6 * max(float(main.max()), 1.0e-30),
        distance=max(int(round(min_sep / df)) if df > 0.0 else 1, 1),
    )
    order = np.argsort(props["prominences"])[::-1]
    chosen = [int(peaks[i]) for i in order[:n_mode]]

    if len(chosen) < n_mode:
        # 谱峰不足：在突出度最大的峰两侧按 min_sep 补点
        if chosen:
            base = chosen[0]
        else:
            base = int(np.argmax(main))
        cand = [base]
        step = 1
        while len(cand) < n_mode and step * min_sep < span:
            for sign in (1, -1):
                idx = base + sign * step
                if 0 <= idx < nf:
                    cand.append(int(idx))
                if len(cand) >= n_mode:
                    break
            step += 1
        chosen = cand[:n_mode]

    out = np.sort(np.asarray([float(freqs[i]) for i in chosen], dtype=float))
    # 兜底：任何两阶初值重合时按 min_sep 拉开
    for i in range(1, out.size):
        if out[i] - out[i - 1] < min_sep:
            out[i] = min(out[i - 1] + min_sep, float(freqs[-1]))
    return out


def _initial_coordinates(
    dhat: np.ndarray, basis: np.ndarray, freqs: np.ndarray, f0: np.ndarray
) -> np.ndarray:
    """子空间坐标初值（第 13.6.1 节的高信噪比渐近结果）。

    各阶模态形状初值取对应初始频率处样本谱矩阵 ``D_k`` 的主特征向量：在该
    频率上主导模态的信噪比最高，主特征向量即该阶模态形状的最佳一阶近似。
    把各列投影到子空间基 ``B0`` 后作正交归一化，得到 ``alpha`` 的初值。

    若两阶初始频率落在同一 FFT 点上（密集模态），主特征向量退化为同一方向，
    此时改用 ``A0 = sum_k D_k`` 的前 ``m`` 个最大特征向量，保证各列线性无关。
    """
    m0 = basis.shape[1]
    m = int(f0.size)
    cols = []
    for fi in f0:
        idx = int(np.argmin(np.abs(freqs - fi)))
        dk = 0.5 * (dhat[:, :, idx] + dhat[:, :, idx].T)
        _, vecs = np.linalg.eigh(dk)
        cols.append(basis.T @ vecs[:, -1])
    mat = np.column_stack(cols)
    # 若各列近似共线（密集模态），改用 A0 的特征向量
    if mat.shape[1] > 1:
        sv = np.linalg.svd(mat, compute_uv=False)
        degenerate = sv[-1] < 1.0e-3 * sv[0]
    else:
        degenerate = False
    if degenerate:
        a0 = np.einsum("ijk->ij", dhat)
        a0 = 0.5 * (a0 + a0.T)
        vals, vecs = np.linalg.eigh(a0)
        order = np.argsort(vals)[::-1]
        mat = basis.T @ vecs[:, order[:m]]
    # 在 m0 维坐标空间内作 QR 正交化，使各列线性无关
    q, r = np.linalg.qr(mat)
    sign = np.sign(np.diag(r))
    sign[sign == 0.0] = 1.0
    return q * sign[None, :]


def _noise_guess(dhat: np.ndarray, d_total: float, m0: int, nf: int) -> float:
    """噪声谱初值，按式 (13.60) 的思路用高阶奇异值估计。

    ``S_e ≈ (d - d0) / ((n - m0) * N_f)``，其中 ``d0`` 取 ``A0`` 前 ``m0`` 个
    最大特征值之和。
    """
    nch = dhat.shape[0]
    a0 = np.einsum("ijk->ij", dhat)
    vals = np.linalg.eigvalsh(0.5 * (a0 + a0.T))[::-1]
    d0 = float(np.sum(vals[:m0]))
    denom = max((nch - m0) * nf, 1)
    return max((d_total - d0) / denom, 1.0e-30)


def mpv(
    freqs: np.ndarray,
    fft: np.ndarray,
    n_mode: int,
    q: int = 0,
    f0: np.ndarray | None = None,
    max_iter: int = 40,
    tol: float = 1.0e-7,
    n_restart: int = 3,
) -> MultiModeResult:
    """多模态最可能值（MPV），第 13.5 节。

    交替更新两组参数：谱参数 ``{f, z, S, S_e}``（``S`` 用式 13.50–13.57 的
    结构化参数化）与振型坐标 ``alpha``。每次更新都是对同一 NLLF 的有界拟牛顿
    极小化，交替至参数相对变化小于 ``tol``。

    为降低对初值的敏感，在初始频率附近做 ``n_restart`` 次扰动重启，取 NLLF
    最小的结果。

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
        频带内频率坐标。
    fft : ndarray, shape (n, Nf), complex
        频带内的缩放 FFT ``F_k``（式 10.1）。似然函数中式 (13.47) 的二次型
        需要复值 ``F_k``，因此本函数以 ``F_k`` 而非样本谱矩阵为输入。
    n_mode : int
        频带内模态数 ``m``。
    q : int, optional
        数据类型指数。
    f0 : ndarray, optional
        频率初值；缺省由第一条奇异值谱曲线的谱峰给出。
    max_iter : int, optional
        交替迭代上限。
    tol : float, optional
        收敛容差。
    n_restart : int, optional
        扰动重启次数，默认 3。

    Returns
    -------
    MultiModeResult
    """
    fft = np.asarray(fft, dtype=complex)
    if fft.ndim != 2:
        raise ValueError("fft 必须是二维数组，形状 (n, Nf)")
    nch = fft.shape[0]
    freqs = np.asarray(freqs, dtype=float)
    nf = freqs.size
    if fft.shape[1] != nf:
        raise ValueError("fft 的列数必须与 freqs 的长度一致")
    m = int(n_mode)
    if m < 1:
        raise ValueError("n_mode 必须为正整数")
    m0 = int(min(m, nch))

    dhat = np.einsum("ak,bk->abk", fft, fft.conj()).real
    d_total = float(np.sum(np.abs(fft) ** 2))

    basis = _initial_basis(dhat, m0)
    fproj = _project_fft(fft, basis)

    f_base = _initial_frequencies(freqs, dhat, m)
    if f0 is not None:
        arr = np.asarray(f0, dtype=float).reshape(-1)
        if arr.size == m:
            f_base = np.sort(arr)

    df = float(freqs[1] - freqs[0]) if nf > 1 else 0.0
    best: MultiModeResult | None = None
    restarts = 0
    for trial in range(max(1, int(n_restart))):
        if trial == 0:
            f_start = f_base.copy()
        else:
            shift = (trial - (n_restart - 1) / 2.0) * 2.0 * df
            f_start = np.clip(f_base + shift, freqs[0], freqs[-1])
        res = _solve(
            freqs, dhat, fft, fproj, d_total, m, m0, nch, q, f_start, basis, max_iter, tol
        )
        restarts += 1
        if best is None or res.nllf < best.nllf:
            best = res
    assert best is not None
    best.restarts = restarts
    return best


def _solve(
    freqs: np.ndarray,
    dhat: np.ndarray,
    fft: np.ndarray,
    fproj: np.ndarray,
    d_total: float,
    m: int,
    m0: int,
    nch: int,
    q: int,
    f_start: np.ndarray,
    basis: np.ndarray,
    max_iter: int,
    tol: float,
) -> MultiModeResult:
    """从给定频率初值求解一次，返回该起点的最优结果。

    按第 13.5 节的参数分组做交替极小化：先固定振型坐标更新谱参数
    ``{f, z, S, S_e}``，再固定谱参数更新振型坐标 ``alpha``，交替至收敛。
    把一次高维极小化拆成两个低维子问题，可避免参数尺度差异引起的病态。

    交替解之后追加一次联合精化，参数向量同时包含谱参数、振型坐标与子空间
    超角（式 13.22–13.23）。固定子空间基时，密集模态的真实解会转出初始子空间，
    交替极小化只能停在该子空间内的鞍点；放开超角才能到达局部极小点。
    """
    nf = freqs.size
    half = m * (m - 1) // 2

    z_cur = np.full(m, 0.01)
    se_cur = _noise_guess(dhat, d_total, m0, nf)
    alpha = _initial_coordinates(dhat, basis, freqs, f_start)

    # 模态力尺度初值：按第 13.6.1 节，用各阶频率处样本谱峰与动力放大因子估计
    # S_ii ≈ sigma_peak * 4 z_i^2 * (2 pi f_i)^{2q}
    scales = np.empty(m)
    for i in range(m):
        idx = int(np.argmin(np.abs(freqs - f_start[i])))
        peak = float(np.linalg.eigvalsh(0.5 * (dhat[:, :, idx] + dhat[:, :, idx].T))[-1])
        scales[i] = max(
            np.sqrt(max(peak, 1.0e-30)) * 2.0 * z_cur[i] * (2.0 * np.pi * f_start[i]) ** q,
            1.0e-12,
        )

    corr = np.concatenate([np.full(half, 0.1), np.zeros(half)])

    def pack_spectrum(fv, zv, sev, sc, cp):
        return np.concatenate(
            [np.log(fv), np.log(zv), [np.log(sev)], np.log(sc), cp]
        )

    def unpack_spectrum(vec):
        fv = np.exp(vec[:m])
        zv = np.exp(vec[m : 2 * m])
        sev = float(np.exp(vec[2 * m]))
        sc = np.exp(vec[2 * m + 1 : 3 * m + 1])
        cp = vec[3 * m + 1 :]
        return fv, zv, sev, sc, cp

    lower_s = np.concatenate(
        [
            np.log(np.full(m, max(freqs[0] * 0.5, 1.0e-12))),
            np.log(np.full(m, _ZMIN)),
            [np.log(_SMIN)],
            np.log(np.full(m, _SMIN)),
            np.full(2 * half, -np.inf),
        ]
    )
    upper_s = np.concatenate(
        [
            np.log(np.full(m, freqs[-1] * 2.0)),
            np.log(np.full(m, _ZMAX)),
            [np.log(1.0e30)],
            np.log(np.full(m, 1.0e30)),
            np.full(2 * half, np.inf),
        ]
    )

    def normalise(al):
        norms = np.linalg.norm(al, axis=0)
        return al / np.where(norms > 0.0, norms, 1.0)[None, :]

    def spectrum_objective(vec, al):
        fv, zv, sev, sc, cp = unpack_spectrum(vec)
        smat = structured_modal_psd(cp, m, sc)
        return nllf_multi(fv, zv, smat, sev, al, freqs, fproj, d_total, q, nch)

    def alpha_objective(vec, fv, zv, sev, sc, cp):
        al = normalise(vec[: m0 * m].reshape(m0, m) + 1j * vec[m0 * m :].reshape(m0, m))
        smat = structured_modal_psd(cp, m, sc)
        return nllf_multi(fv, zv, smat, sev, al, freqs, fproj, d_total, q, nch)

    xs = pack_spectrum(f_start, z_cur, se_cur, scales, corr)
    xa = np.concatenate([alpha.real.reshape(-1), alpha.imag.reshape(-1)])

    f_cur, z_cur, se_cur, sc_cur, cp_cur = unpack_spectrum(xs)
    best_val = spectrum_objective(xs, normalise(xa[: m0 * m].reshape(m0, m) + 1j * xa[m0 * m :].reshape(m0, m)))
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        al_cur = normalise(xa[: m0 * m].reshape(m0, m) + 1j * xa[m0 * m :].reshape(m0, m))

        xs_old = xs.copy()
        xa_old = xa.copy()
        res_s = minimize(
            lambda v: spectrum_objective(v, al_cur),
            xs,
            method="L-BFGS-B",
            bounds=list(zip(lower_s, upper_s)),
            options={"maxiter": 400, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
        )
        if np.all(np.isfinite(res_s.x)):
            xs = res_s.x
        f_cur, z_cur, se_cur, sc_cur, cp_cur = unpack_spectrum(xs)

        res_a = minimize(
            lambda v: alpha_objective(v, f_cur, z_cur, se_cur, sc_cur, cp_cur),
            xa,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
        )
        if np.all(np.isfinite(res_a.x)):
            xa = res_a.x
        alpha = normalise(xa[: m0 * m].reshape(m0, m) + 1j * xa[m0 * m :].reshape(m0, m))

        val = spectrum_objective(xs, alpha)
        step = float(np.max(np.abs(np.concatenate([xs - xs_old, xa - xa_old]))))
        rel = float(abs(best_val - val) / max(abs(best_val), 1.0))
        if np.isfinite(val) and val <= best_val + 1.0e-9:
            best_val = val
        # 参数步长或似然函数相对变化任一达到容差即判定收敛：交替极小化中
        # 参数会在最优点附近小幅摆动，单看参数步长容易误判未收敛。
        if step < tol or rel < max(tol * 1.0e-2, 1.0e-10):
            converged = True
            break

    f_fin, z_fin, se_fin, sc_fin, cp_fin = unpack_spectrum(xs)

    # 联合精化：交替极小化在密集模态下可能停在鞍点——两个子问题各自下降，但
    # 联合方向仍可下降。此处以交替解为起点做一次联合极小化，参数向量同时包含
    # 谱参数、振型坐标与子空间超角（式 13.22–13.23），确保输出点是负对数似然的
    # 局部极小点，后验协方差才有意义。
    n_spec = xs.size
    n_coord = xa.size
    n_ang = m0 * (nch - m0)
    bfull = _full_basis(basis, nch)

    def expand_joint(vec):
        """把联合参数向量展开为 ``(谱参数, 振型坐标, 子空间基)``。"""
        fv, zv, sev, sc, cp = unpack_spectrum(vec[:n_spec])
        coords = vec[n_spec : n_spec + n_coord]
        al = normalise(coords[: m0 * m].reshape(m0, m) + 1j * coords[m0 * m :].reshape(m0, m))
        if n_ang > 0:
            ang = vec[n_spec + n_coord : n_spec + n_coord + n_ang]
            bs = _rotate_basis(bfull, m0, ang)
        else:
            bs = bfull[:, :m0]
        return fv, zv, sev, sc, cp, al, bs

    def joint_objective(vec):
        fv, zv, sev, sc, cp, al, bs = expand_joint(vec)
        if bs is basis or np.array_equal(bs, basis):
            fp = fproj
        else:
            fp = _project_fft(fft, bs)
        smat = structured_modal_psd(cp, m, sc)
        return nllf_multi(fv, zv, smat, sev, al, freqs, fp, d_total, q, nch)

    lower_j = np.concatenate([lower_s, np.full(n_coord, -np.inf), np.full(n_ang, -np.inf)])
    upper_j = np.concatenate([upper_s, np.full(n_coord, np.inf), np.full(n_ang, np.inf)])

    x_j = np.concatenate([xs, xa, np.zeros(n_ang)])
    val_before = joint_objective(x_j)
    res_j = minimize(
        joint_objective,
        x_j,
        method="L-BFGS-B",
        bounds=list(zip(lower_j, upper_j)),
        options={"maxiter": 3000, "ftol": 1.0e-15, "gtol": 1.0e-11, "maxls": 60},
    )
    if np.all(np.isfinite(res_j.x)) and np.isfinite(res_j.fun) and res_j.fun <= val_before:
        x_j = res_j.x
        polished = bool(res_j.fun < val_before - 1.0e-9 * max(abs(val_before), 1.0))
    else:
        polished = False
    xs = x_j[:n_spec]
    xa = x_j[n_spec : n_spec + n_coord]

    f_fin, z_fin, se_fin, sc_fin, cp_fin, alpha, basis = expand_joint(x_j)
    s_fin = structured_modal_psd(cp_fin, m, sc_fin)
    phi = basis @ alpha
    best_val = float(joint_objective(x_j))
    if polished:
        converged = True

    # 按频率升序排列各阶模态：极小化过程不约束频率次序，多阶模态可能互换位置，
    # 输出前统一排序，使结果与输入初值的顺序无关。
    order = np.argsort(f_fin)
    f_fin = np.asarray(f_fin, dtype=float)[order]
    z_fin = np.asarray(z_fin, dtype=float)[order]
    sc_fin = np.asarray(sc_fin, dtype=float)[order]
    phi = phi[:, order]
    alpha = alpha[:, order]
    s_fin = s_fin[np.ix_(order, order)]

    for i in range(m):
        nrm = float(np.linalg.norm(phi[:, i]))
        if nrm > 0.0:
            phi[:, i] = phi[:, i] / nrm
            alpha[:, i] = alpha[:, i] / nrm

    # 模态形状的相位是模型的不确定量：把 alpha 的第 i 列乘以 e^{i theta_i} 并把
    # S 同步变换为 diag(e^{i theta}) S diag(e^{-i theta})，式 (13.10) 的二次型
    # 完全不变。为使输出可比较，取基准相位使每列绝对值最大的分量成为正实数，
    # 再把该相位差并回 S。phi、alpha、S 三者必须同步变换：只改 phi 的符号会使
    # 三者在后续似然计算中互不一致。
    theta = np.zeros(m)
    for i in range(m):
        idx = int(np.argmax(np.abs(phi[:, i])))
        if abs(phi[idx, i]) > 0.0:
            theta[i] = float(np.angle(phi[idx, i]))
    phase = np.exp(1j * theta)
    alpha = alpha * np.conj(phase)[None, :]
    phi = basis @ alpha
    s_fin = (phase[:, None] * s_fin) * np.conj(phase)[None, :]

    return MultiModeResult(
        f=np.asarray(f_fin, dtype=float),
        z=np.asarray(z_fin, dtype=float),
        S=s_fin,
        Se=float(se_fin),
        phi=phi,
        nllf=float(best_val),
        n_iter=int(n_iter),
        converged=bool(converged),
        subspace_rank=int(m0),
        basis=basis,
        alpha=alpha,
        restarts=1,
    )
