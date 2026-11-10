"""自动频带选择与模态阶数判定（课本第 7.3 节奇异值谱）。

环境振动数据在共振带内的主导模态数，可由奇异值谱（``Re(E_k)`` 的特征值按降序
排列）判定：某频率点上显著高于噪声基线的曲线条数即该处的主导模态数。本模块
据此自动划分频带、给出每个频带内的模态阶数，并提供按不确定性定律校核选带
质量的工具，使识别过程在缺少先验信息时也能自动完成。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .single_mode import dynamic_factor


@dataclass
class BandSelection:
    """自动选带结果。

    Attributes
    ----------
    bands : ndarray, shape (nb, 2)
        各频带的下、上界（Hz）。
    n_modes : ndarray, shape (nb,), int
        各频带内的模态阶数。
    peaks : ndarray, shape (npk,)
        判定的谱峰频率（Hz）。
    sv : ndarray, shape (n, Nf)
        奇异值谱曲线，按降序排列。
    noise_floor : float
        噪声基线，取高频段奇异值谱的中位数。
    """

    bands: np.ndarray
    n_modes: np.ndarray
    peaks: np.ndarray
    sv: np.ndarray
    noise_floor: float


def sv_curves(dhat: np.ndarray) -> np.ndarray:
    """奇异值谱曲线，形状 ``(n, Nf)``，按降序排列（第 7.3 节）。"""
    arr = np.asarray(dhat, dtype=float)
    vals = np.linalg.eigvalsh(np.moveaxis(0.5 * (arr + np.swapaxes(arr, 0, 1)), 2, 0))
    return np.sort(vals, axis=1)[:, ::-1].T


def _moving_average(curve: np.ndarray, nwin: int) -> np.ndarray:
    """长度为 ``nwin`` 的滑动平均，两端按可用点数归一。"""
    c = np.asarray(curve, dtype=float)
    nwin = max(int(nwin), 1)
    if nwin == 1 or c.size < 2:
        return c.copy()
    nwin = min(nwin, c.size)
    kern = np.ones(nwin, dtype=float)
    num = np.convolve(c, kern, mode="same")
    den = np.convolve(np.ones_like(c), kern, mode="same")
    return num / np.maximum(den, 1.0)


def estimate_noise_floor(sv: np.ndarray, frac: float = 0.2) -> float:
    """噪声基线估计：最高阶奇异值曲线在全频段上的中位数。

    最高阶曲线在共振带内几乎不含模态贡献，其幅值由通道噪声决定；取中位数可
    避免个别频率点上的起伏影响判据。

    注意：``sv`` 必须由**经平均的谱矩阵**得到（如 Welch 平均的谱密度矩阵），
    单次实现的 ``Re(F_k F_k^*)`` 秩为 1，其高阶奇异值恒为零，不能用于判定
    模态数。
    """
    curve = np.asarray(sv)[-1]
    return float(np.median(curve))


def count_modes_at(
    sv: np.ndarray,
    idx: int,
    noise_floor: float,
    factor: float = 3.0,
    rel_top: float = 1.0e-2,
) -> int:
    """指定频率点上的主导模态数：与最高阶谱线同量级且高于噪声基线的曲线条数。

    Parameters
    ----------
    sv : ndarray, shape (n, Nf)
        奇异值谱曲线。
    idx : int
        频率点索引。
    noise_floor : float
        噪声基线。
    factor : float, optional
        相对噪声基线的倍数下限，默认 3。
    rel_top : float, optional
        相对该频率点最高阶谱线的比例下限，默认 ``1e-2``。
    """
    vals = np.asarray(sv)[:, idx]
    top = float(vals[0]) if vals.size else 0.0
    thr = max(noise_floor * factor, rel_top * top, 1.0e-300)
    return int(np.count_nonzero(vals > thr))


def select_bands(
    freqs: np.ndarray,
    dhat: np.ndarray,
    *,
    n_bands: int | None = None,
    kappa: float = 10.0,
    min_sep_hz: float | None = None,
    prominence: float = 1.0e-3,
    max_modes: int = 4,
    noise_factor: float = 3.0,
    peak_factor: float = 5.0,
    rel_peak: float = 5.0e-2,
    smooth_hz: float | None = None,
) -> BandSelection:
    """自动划分频带并给出各频带内的模态阶数。

    步骤：

    1. 计算奇异值谱并估计噪声基线（最高阶曲线的中位数）；
    2. 对第一条奇异值谱曲线做滑动平均，在平滑曲线上提取谱峰，谱峰间隔不小于
       ``min_sep_hz``；
    3. 每个谱峰对应一个频带，频带半宽取 ``kappa`` 倍阻尼比对应的半功率带宽
       （阻尼比未知时按 1% 估计，``kappa=10`` 即约 ±10% 的带宽），左右相邻
       频带重叠时按中点切开；
    4. 在谱峰处按 ``noise_factor`` 倍噪声基线计数显著曲线，得该频带模态阶数，
       上限 ``max_modes``。

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
        频率坐标（Hz）。
    dhat : ndarray, shape (n, n, Nf)
        样本谱矩阵 ``D_k``。
    n_bands : int, optional
        期望频带数；缺省由谱峰判定。
    kappa : float, optional
        频带半宽相对半功率带宽的倍数，默认 10。
    min_sep_hz : float, optional
        相邻谱峰的最小间隔（Hz）；缺省取 3 倍频率分辨率与 2% 频率跨度中的较大者。
    prominence : float, optional
        谱峰相对突出度阈值（对扣除基线后的最大值归一），默认 ``1e-3``。
    max_modes : int, optional
        单个频带允许的最大模态阶数，默认 4。
    noise_factor : float, optional
        判定模态数时相对噪声基线的倍数，默认 3。
    peak_factor : float, optional
        谱峰必须高于噪声基线的倍数，默认 5；低于该阈值的局部极大判为噪声起伏。
    rel_peak : float, optional
        谱峰相对最高谱峰的比例下限，默认 ``5e-2``。只有噪声基线的宽带起伏
        可能高出基线若干倍，但远低于真实共振峰；该判据把这类起伏排除在外。
    smooth_hz : float, optional
        谱峰搜索前的滑动平均窗宽（Hz）；缺省取 10 倍频率分辨率与 2% 频率跨度
        中的较大者。

    Returns
    -------
    BandSelection
    """
    from scipy.signal import find_peaks

    fr = np.asarray(freqs, dtype=float)
    nf = fr.size
    df = float(fr[1] - fr[0]) if nf > 1 else 0.0
    span = float(fr[-1] - fr[0])
    sv = sv_curves(dhat)
    floor = estimate_noise_floor(sv)

    # 单次实现的谱估计起伏剧烈，直接在原始曲线上找峰会把噪声起伏当作模态。
    # 先按平滑窗宽做滑动平均，再在平滑后的曲线上判定谱峰。平滑窗宽只消除
    # 逐点起伏，不得掩盖相邻模态之间的间隔。
    width = smooth_hz if smooth_hz is not None else max(4.0 * df, 0.002 * span)
    nwin = max(int(round(width / df)) if df > 0.0 else 1, 1)
    main = _moving_average(sv[0], nwin)

    sep = min_sep_hz if min_sep_hz is not None else max(3.0 * df, 0.002 * span)
    above = np.maximum(main - floor, 0.0)
    peaks, _ = find_peaks(
        above,
        prominence=prominence * max(float(above.max()), 1.0e-300),
        distance=max(int(round(sep / df)) if df > 0.0 else 1, 1),
    )
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(above))])
    # 宽带噪声起伏也会形成局部极大；只保留显著高于噪声基线、且幅值不远离
    # 最高谱峰的峰，使判定结果对应真实的共振带而不是噪声起伏。仅用噪声基线
    # 作判据不够：噪声起伏可高出基线数倍，却远低于真实共振峰。
    strong = peaks[main[peaks] > peak_factor * floor]
    if strong.size:
        peaks = strong
    if peaks.size:
        top_peak = float(np.max(main[peaks]))
        peaks = peaks[main[peaks] >= rel_peak * top_peak]
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(above))])
    order = np.argsort(above[peaks])[::-1][: n_bands if n_bands else peaks.size]
    idx = np.sort(peaks[order])
    pfreq = fr[idx]

    # 频带半宽：按 1% 阻尼比的半功率带宽乘以 kappa
    zeta0 = 0.01
    half = np.maximum(kappa * zeta0 * pfreq, 5.0 * df if df > 0.0 else 0.0)
    lower = pfreq - half
    upper = pfreq + half
    for i in range(1, pfreq.size):
        mid = 0.5 * (pfreq[i - 1] + pfreq[i])
        if lower[i] < mid:
            lower[i] = mid
        if upper[i - 1] > mid:
            upper[i - 1] = mid
    lower = np.clip(lower, fr[0], fr[-1])
    upper = np.clip(upper, fr[0], fr[-1])

    modes = np.array(
        [min(max(count_modes_at(sv, int(i), floor, noise_factor), 1), max_modes) for i in idx],
        dtype=int,
    )
    keep = upper > lower
    return BandSelection(
        bands=np.column_stack([lower[keep], upper[keep]]),
        n_modes=modes[keep],
        peaks=pfreq[keep],
        sv=sv,
        noise_floor=floor,
    )


def uncertainty_law_cov(
    f: float,
    z: float,
    S: float,
    Se: float,
    kappa: float,
    duration: float,
    nch: int,
    q: int = 0,
) -> dict[str, float]:
    """用第 15 章的不确定度定律预测各参数后验变异系数。

    该预测只需模态参数与记录长度，无需重新计算 Hessian，因此可用于比较不同
    选带方案与试验时长的预期精度。返回 ``f``、``z``、``S``、``Se``、``phi``
    五项系数。

    Parameters
    ----------
    f, z : float
        固有频率（Hz）与阻尼比。
    S, Se : float
        模态力谱密度与预测误差谱密度。
    kappa : float
        带宽因子，频带取 ``f(1 ± kappa*z)``。
    duration : float
        记录时长（s）。
    nch : int
        测点数。
    q : int, optional
        数据类型指数。

    Returns
    -------
    dict[str, float]
        各参数的变异系数平方根。
    """
    f = float(f)
    z = float(z)
    nf = 2.0 * kappa * z * f * float(duration)
    nc = nf / (2.0 * kappa * z)
    if nf <= 0.0 or nc <= 0.0:
        return {k: np.inf for k in ("f", "z", "S", "Se", "phi")}
    mu = float(Se) / max(float(S), 1.0e-300) * (2.0 * np.pi * f) ** (2.0 * q)
    c = 1.0 / (4.0 * mu * z * z) if mu > 0.0 else np.inf

    at = np.arctan(kappa)
    bracket_f = at - kappa / (kappa * kappa + 1.0)
    b_f = 2.0 / np.pi * bracket_f
    a_f = 4.0 * (kappa - at) / bracket_f
    b_z = 2.0 / np.pi * (at + kappa / (kappa * kappa + 1.0) - 2.0 * at * at / kappa)
    a_z = (
        4.0
        * (kappa * kappa + 1.0)
        * (3.0 * at - 3.0 * kappa + kappa * kappa * at)
        * at
        / (3.0 * ((kappa * kappa + 1.0) * (kappa - 2.0 * at) * at + kappa * kappa))
    )
    b_s = 1.0 - 2.0 / kappa * at * at / (at + kappa / (kappa * kappa + 1.0))
    b_par = at + kappa / (kappa * kappa + 1.0)
    a_s = (
        2.0
        + (2.0 / 3.0) * kappa * kappa
        - at * at
        / (2.0 * at * at - b_par * kappa)
        * (8.0 * at / b_par - 8.0 * kappa / at + (4.0 / 3.0) * kappa * kappa + 4.0)
    )
    b_u = 2.0 / np.pi * at

    def scaled(x0: float, alpha: float) -> float:
        if c <= 0.0 or not np.isfinite(c):
            return x0
        return x0 * (1.0 + alpha / c)

    out = {
        "f": float(np.sqrt(max(scaled(z / (2.0 * np.pi * nc * b_f), a_f), 0.0))),
        "z": float(np.sqrt(max(scaled(1.0 / (2.0 * np.pi * z * nc * b_z), a_z), 0.0))),
        "S": float(np.sqrt(max(scaled(1.0 / (nf * b_s), a_s), 0.0))),
        "Se": float(np.sqrt(max(1.0 / ((nch - 1) * nf), 0.0))),
        "phi": float(np.sqrt(max(2.0 * (nch - 1) * mu * z / (np.pi * nc * b_u), 0.0))),
    }
    return out


def optimal_duration(
    f: float, z: float, S: float, Se: float, kappa: float, nch: int, target: float = 0.01, q: int = 0
) -> float:
    """满足给定频率变异系数所需的最短记录时长（s）。

    以第 15 章的零阶不确定度定律反解 ``T_d``，用于试验规划。
    """
    law = uncertainty_law_cov(f, z, S, Se, kappa, 1.0, nch, q)
    base = law["f"]
    if not np.isfinite(base) or base <= 0.0:
        return np.inf
    return float((base / float(target)) ** 2)
