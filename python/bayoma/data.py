"""数据预处理、缩放 FFT 与样本谱矩阵。

本模块实现课本第 10.1 节的缩放 FFT 定义（式 10.1）与第 7.3 节的奇异值谱，
为后续单模态、多模态与多测组识别提供统一的频域数据入口。

记号约定
--------
- ``n``  ：测点数（通道数）
- ``N``  ：每通道样本数
- ``dt`` ：采样时间间隔，``dt = 1/fs``
- ``F_k``：缩放 FFT，``F_k = sqrt(dt/N) * sum_j y_j exp(-2*pi*i*j*k/N)``
- ``f_k``：``k/(N*dt)``

缩放方式使 ``E[F_k F_k^*]`` 直接等于数据的单边功率谱密度矩阵，单位与
``S_ii``、``S_e`` 一致（课本第 10.1 节）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._util import LOG, band_pairs, ensure_finite, positive


@dataclass
class SpectralData:
    """一个频带内的频域数据集合。

    Attributes
    ----------
    freqs : ndarray, shape (Nf,)
        频带内各 FFT 点的频率（Hz）。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT，按列存放各频率点的 ``F_k``。
    dfreq : float
        频率分辨率 ``1/(N*dt)``。
    nch : int
        通道数 ``n``。
    nf : int
        频带内 FFT 点数 ``N_f``。
    band : tuple[float, float]
        频带上下界（Hz）。
    """

    freqs: np.ndarray
    fft: np.ndarray
    dfreq: float
    nch: int
    nf: int
    band: tuple[float, float]

    @property
    def psd_matrices(self) -> np.ndarray:
        """返回 ``D_k = Re(F_k F_k^*)``，形状 ``(n, n, Nf)``。"""
        return real_psd_matrices(self.fft)

    @property
    def d_total(self) -> float:
        """返回 ``d = sum_k F_k^* F_k``（式 12.12 / 14.33）。"""
        return float(np.sum(np.abs(self.fft) ** 2))


def remove_mean_and_trend(data: np.ndarray, detrend: bool = True) -> np.ndarray:
    """去掉各通道均值（可选再去除线性趋势）。

    环境振动数据的均值与缓慢漂移不携带模态信息，去除后可避免在零频附近
    产生虚假的能量集中。
    """
    out = np.asarray(data, dtype=float, copy=True)
    out -= out.mean(axis=0, keepdims=True)
    if detrend:
        nsam = out.shape[0]
        t = np.arange(nsam, dtype=float)
        t -= t.mean()
        denom = float(np.dot(t, t))
        if denom > 0.0:
            slope = (t @ out) / denom
            out -= np.outer(t, slope)
    return out


def scaled_fft(
    data: np.ndarray,
    fs: float,
    band: tuple[float, float],
    p: np.ndarray | None = None,
    detrend: bool = True,
) -> SpectralData:
    """计算指定频带内的缩放 FFT。

    Parameters
    ----------
    data : ndarray, shape (N, n)
        时程数据，按列存放各通道。
    fs : float
        采样频率（Hz）。
    band : tuple[float, float]
        频带上下界（Hz）。
    p : ndarray, shape (n,), optional
        各通道的单位换算指数。第 i 通道的缩放 FFT 乘以 ``(1j*2*pi*f_k)**p[i]``。
        ``p=0`` 表示原始数据本身即为加速度；``p=1`` 为速度；``p=2`` 为位移。
    detrend : bool, optional
        是否在变换前去除线性趋势，默认 ``True``。

    Returns
    -------
    SpectralData
        频带内的缩放 FFT 及频率坐标。
    """
    fs = positive(fs, "fs")
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("tdata 必须是二维数组，行对应时刻、列对应通道")
    nsam, nch = arr.shape
    if nsam < 16:
        raise ValueError("样本数过少，无法进行频域识别")
    ensure_finite(arr, "tdata")

    dt = 1.0 / fs
    arr = remove_mean_and_trend(arr, detrend=detrend)

    spec = np.fft.rfft(arr, axis=0)
    freqs_all = np.fft.rfftfreq(nsam, d=dt)
    scale = np.sqrt(dt / nsam)
    spec *= scale

    if p is not None:
        pexp = np.asarray(p, dtype=float).reshape(-1)
        if pexp.size == 1:
            pexp = np.full(nch, float(pexp[0]))
        if pexp.size != nch:
            raise ValueError(f"p 的长度 {pexp.size} 与通道数 {nch} 不一致")
        omega = 2.0 * np.pi * freqs_all
        # 对零频点，负指数会产生无穷大，但其不在识别频带内，先以 1 替代后覆盖。
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.power((1j * omega)[:, None], pexp[None, :])
        factor[0, :] = 0.0
        spec = spec * factor

    lo, hi = float(band[0]), float(band[1])
    sel = (freqs_all >= lo) & (freqs_all <= hi) & (freqs_all > 0.0)
    if not np.any(sel):
        raise ValueError(
            f"频带 [{lo}, {hi}] Hz 内没有 FFT 点，请检查 fs 与频带设置"
        )
    freqs = freqs_all[sel]
    fft = spec[sel, :].T  # (n, Nf)

    return SpectralData(
        freqs=freqs,
        fft=np.ascontiguousarray(fft),
        dfreq=fs / nsam,
        nch=nch,
        nf=int(freqs.size),
        band=(lo, hi),
    )


def real_psd_matrices(fft: np.ndarray) -> np.ndarray:
    """由缩放 FFT 计算 ``D_k = Re(F_k F_k^*)``。

    ``D_k`` 为实对称半正定矩阵，是课本中式 (12.12) 与 (14.33) 的定义。
    """
    outer = fft[:, None, :] * np.conj(fft)[None, :, :]
    return np.real(outer)


def sample_psd_welch(
    data: np.ndarray,
    fs: float,
    nblock: int | None = None,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch 平均的样本功率谱密度矩阵，仅用于奇异值谱与选带。

    识别所用的似然函数基于整段记录的缩放 FFT（式 10.1），本函数不参与
    似然计算，只提供平滑的谱估计以稳定峰值搜索。

    Returns
    -------
    freqs : ndarray, shape (Nf,)
    psd : ndarray, shape (n, n, Nf)
    """
    fs = positive(fs, "fs")
    arr = remove_mean_and_trend(np.asarray(data, dtype=float))
    nsam, nch = arr.shape
    if nblock is None:
        nblock = int(min(nsam, max(256, 2 ** int(np.floor(np.log2(nsam / 8))))))
    nblock = int(min(nblock, nsam))
    step = max(1, int(round(nblock * (1.0 - overlap))))
    starts = list(range(0, nsam - nblock + 1, step))
    if not starts:
        starts = [0]

    dt = 1.0 / fs
    win = np.hanning(nblock)
    win_power = float(np.sum(win ** 2))
    acc = np.zeros((nch, nch, nblock // 2 + 1), dtype=complex)
    for s in starts:
        seg = arr[s : s + nblock] * win[:, None]
        spec = np.fft.rfft(seg, axis=0).T  # (nch, Nf)
        acc += spec[:, None, :] * np.conj(spec)[None, :, :]
    # 归一化使结果与单边 PSD 同尺度：2*dt/win_power，直流与 Nyquist 不乘 2。
    acc *= 2.0 * dt / win_power
    acc /= len(starts)
    acc[..., 0] /= 2.0
    acc[..., -1] /= 2.0
    freqs = np.fft.rfftfreq(nblock, d=dt)
    return freqs, np.real(acc)


def singular_value_spectrum(psd: np.ndarray) -> np.ndarray:
    """奇异值谱：``Re(E_k)`` 的特征值，按降序排列。

    课本第 7.3 节指出，在共振带内 ``Re(E_k)`` 的前若干个特征值显著高于其余
    特征值，其个数等于该频带内主导模态的数目。
    """
    nch = psd.shape[0]
    nf = psd.shape[-1]
    eig = np.empty((nch, nf), dtype=float)
    for k in range(nf):
        mat = 0.5 * (psd[:, :, k] + psd[:, :, k].T)
        vals = np.linalg.eigvalsh(mat)
        eig[:, k] = vals[::-1]
    return eig


def select_channels(data: np.ndarray, ichan: list[int] | None) -> np.ndarray:
    """按 0 基索引选择通道；``None`` 表示全部保留。"""
    if ichan is None:
        return data
    arr = np.asarray(data, dtype=float)
    idx = np.asarray(ichan, dtype=int)
    if idx.size == 0:
        return arr
    if np.any(idx < 0) or np.any(idx >= arr.shape[1]):
        raise ValueError(f"ichan 索引越界：允许范围 0..{arr.shape[1] - 1}")
    return arr[:, idx]


def build_p_vector(p: float | list[float] | None, nch: int) -> np.ndarray:
    """把 ``p`` 输入规整成长度为 ``nch`` 的数组。"""
    if p is None:
        return np.zeros(nch, dtype=float)
    arr = np.asarray(p, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(nch, float(arr[0]))
    if arr.size != nch:
        raise ValueError(f"p 的长度 {arr.size} 与通道数 {nch} 不一致")
    return arr


def resolve_bands(f1f2: np.ndarray | list, f1f2d: np.ndarray | list | None) -> np.ndarray:
    """校验频带输入，并剔除需从似然函数排除的频带。

    ``f1f2d`` 用于排除含有干扰（如电机谐波、拍频）的频带，被排除的频带不参与
    任何模态参数的识别。
    """
    bands = band_pairs(f1f2, "f1f2")
    if f1f2d is None:
        return bands
    excluded = np.asarray(f1f2d, dtype=float)
    if excluded.size == 0:
        return bands
    if excluded.ndim == 1:
        excluded = excluded.reshape(1, -1)
    keep = []
    for row in bands:
        blocked = False
        for ex in excluded:
            if not (row[1] <= ex[0] or row[0] >= ex[1]):
                blocked = True
                break
        if blocked:
            LOG.info("频带 [%.4g, %.4g] Hz 与排除频带重叠，已跳过", row[0], row[1])
        else:
            keep.append(row)
    if not keep:
        raise ValueError("所有频带均被 f1f2d 排除，没有可识别的频带")
    return np.asarray(keep, dtype=float)


def mode_shape_from_subspace(
    basis: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """由子空间基与坐标重建模态形状矩阵 ``U = B0 alpha``（式 13.6）。"""
    return basis @ alpha


def orthonormal_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    """取矩阵前 ``rank`` 个左奇异向量作为正交基。

    课本式 (13.8) 用 ``U = L K R^T`` 的 ``L`` 作为 ``B0``，即 ``U`` 的列空间
    的一组正交基。
    """
    u_mat, _, _ = np.linalg.svd(matrix, full_matrices=False)
    r = int(min(rank, u_mat.shape[1]))
    return u_mat[:, :r]


def align_mode_shapes(phi: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """统一各阶模态形状的符号，使最大分量恒为正。

    模态形状的符号在理论上不可识别，固定符号便于跨频带、跨测组的结果比较。
    """
    out = np.array(phi, dtype=float, copy=True)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    for j in range(out.shape[1]):
        col = out[:, j]
        if reference is not None and reference.shape[1] > j:
            ref = np.asarray(reference, dtype=float)[:, j]
            if float(np.dot(col, ref)) < 0.0:
                out[:, j] = -col
        else:
            idx = int(np.argmax(np.abs(col)))
            if col[idx] < 0.0:
                out[:, j] = -col
    return out


def mac_matrix(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """模态置信准则矩阵（MAC）。

    ``MAC_ij = (phi_a[:,i]^T phi_b[:,j])^2 / (||phi_a[:,i]||^2 ||phi_b[:,j]||^2)``。
    """
    a = np.asarray(phi_a, dtype=float)
    b = np.asarray(phi_b, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    num = (a.T @ b) ** 2
    den = np.outer(np.sum(a ** 2, axis=0), np.sum(b ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.nan_to_num(out)
