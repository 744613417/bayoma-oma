"""合成环境振动数据生成。

生成方法为频域合成：直接按课本式 (10.11) 构造缩放 FFT 的协方差矩阵
``E_k = U H_k U^T + S_e I_n``，对每个频率点抽取圆对称复高斯向量，再由逆
FFT 还原时程。这样生成的数据，其缩放 FFT 的样本协方差在统计意义上严格
等于课本的谱矩阵模型，适合用于校验识别算法与后验不确定性。

该方法生成的数据在时域上以记录长度为周期，这是频域合成的固有性质；用于
算法验证不受影响，若需要非周期数据可采用时域滤波方法另行生成。
"""

from __future__ import annotations

import numpy as np

from ._util import LOG


def theoretical_psd(
    freqs: np.ndarray,
    freqs_hz: np.ndarray,
    damping: np.ndarray,
    phi: np.ndarray,
    S: np.ndarray,
    Se: float,
    q: int = 0,
) -> np.ndarray:
    """计算各频率点的理论谱矩阵 ``E_k``，对应式 (10.10)–(10.11)。

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
        频率坐标（Hz）。
    freqs_hz : ndarray, shape (m,)
        各阶固有频率（Hz）。
    damping : ndarray, shape (m,)
        各阶阻尼比。
    phi : ndarray, shape (n, m)
        模态形状矩阵。
    S : ndarray, shape (m, m), complex
        模态力单边谱密度矩阵。
    Se : float
        预测误差单边谱密度。
    q : int, optional
        数据类型指数，``0`` 加速度、``1`` 速度、``2`` 位移。

    Returns
    -------
    ndarray, shape (n, n, Nf), complex
        Hermitian 谱矩阵。
    """
    fr = np.asarray(freqs, dtype=float)
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    z = np.asarray(damping, dtype=float).reshape(-1)
    phim = np.asarray(phi, dtype=float)
    Smat = np.asarray(S, dtype=complex)

    beta = f[:, None] / fr[None, :]
    om = (2.0 * np.pi * fr)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = om ** (-q) / (1.0 - beta ** 2 - 2.0j * z[:, None] * beta)
    h[~np.isfinite(h)] = 0.0

    # E_k = U diag(h_k) S diag(h_k)^* U^T + Se I
    n = phim.shape[0]
    nf = fr.size
    out = np.zeros((n, n, nf), dtype=complex)
    for k in range(nf):
        hk = h[:, k]
        out[:, :, k] = (phim * hk[None, :]) @ Smat @ (phim * np.conj(hk)[None, :]).T
    out += Se * np.eye(n)[:, :, None]
    return out


def _complex_gaussian_samples(cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """抽取圆对称复高斯向量，使其协方差等于给定 Hermitian 矩阵。"""
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.conj().T))
    vals = np.clip(vals, 0.0, None)
    root = vecs @ np.diag(np.sqrt(vals))
    n = cov.shape[0]
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    return root @ noise


def synthesize_time_history(
    psd_matrices: np.ndarray,
    fs: float,
    nsam: int,
    seed: int | None = None,
) -> np.ndarray:
    """由谱矩阵序列合成时程数据。

    Parameters
    ----------
    psd_matrices : ndarray, shape (n, n, Nf), complex
        正频率半轴上的单边谱矩阵，``Nf = nsam//2 + 1``。
    fs : float
        采样频率（Hz）。
    nsam : int
        样本数。
    seed : int, optional
        随机数种子。

    Returns
    -------
    ndarray, shape (nsam, n)
        时程数据。
    """
    rng = np.random.default_rng(seed)
    n, _, nf = psd_matrices.shape
    if nf != nsam // 2 + 1:
        raise ValueError("谱矩阵的频率点数必须等于 nsam//2 + 1")
    dt = 1.0 / fs
    scale = np.sqrt(dt / nsam)

    spec = np.zeros((nf, n), dtype=complex)
    # 直流与 Nyquist 点不做复高斯抽样，避免破坏实信号的共轭对称性
    for k in range(1, nf - 1):
        spec[k, :] = _complex_gaussian_samples(psd_matrices[:, :, k], rng)
    # 由缩放 FFT 还原 numpy 的 rfft 系数
    spec /= scale
    data = np.fft.irfft(spec, n=nsam, axis=0)
    return np.asarray(data, dtype=float)


def simulate_oma_data(
    freqs_hz: np.ndarray,
    damping: np.ndarray,
    phi: np.ndarray,
    S: np.ndarray,
    Se: float,
    fs: float,
    duration: float,
    seed: int | None = None,
    q: int = 0,
) -> np.ndarray:
    """生成单测组环境振动数据。

    Parameters
    ----------
    freqs_hz : ndarray, shape (m,)
        各阶固有频率（Hz）。
    damping : ndarray, shape (m,)
        各阶阻尼比。
    phi : ndarray, shape (n, m)
        模态形状矩阵，各列欧氏范数应为 1。
    S : ndarray, shape (m, m), complex
        模态力单边谱密度矩阵。
    Se : float
        预测误差（通道噪声）单边谱密度。
    fs : float
        采样频率（Hz）。
    duration : float
        记录时长（s）。
    seed : int, optional
        随机数种子。
    q : int, optional
        数据类型指数，``0`` 加速度、``1`` 速度、``2`` 位移。

    Returns
    -------
    ndarray, shape (N, n)
        环境振动数据，按列存放各通道。
    """
    fs = float(fs)
    nsam = int(round(float(duration) * fs))
    if nsam < 32:
        raise ValueError("记录时长过短，至少需要 32 个样本")
    freqs_all = np.fft.rfftfreq(nsam, d=1.0 / fs)
    psd = theoretical_psd(freqs_all, freqs_hz, damping, phi, S, Se, q=q)
    data = synthesize_time_history(psd, fs, nsam, seed=seed)
    LOG.info(
        "合成数据：%d 通道、%d 样本、%.1f s、%d 阶模态、fs=%.4g Hz",
        data.shape[1],
        data.shape[0],
        duration,
        len(np.asarray(freqs_hz).reshape(-1)),
        fs,
    )
    return data


def selection_matrix(dofs: list[int] | np.ndarray, n_dof: int) -> np.ndarray:
    """构造测组选择矩阵 ``L_r``（课本第 10.3 节、式 10.35）。

    ``L_r`` 为 ``(n_r, n)``，第 i 行在第 ``dofs[i]`` 列为 1，其余为 0，使
    局部模态形状 ``t_r = L_r u``。
    """
    idx = np.asarray(dofs, dtype=int).reshape(-1)
    if idx.size == 0:
        raise ValueError("测组的自由度索引不能为空")
    if np.any(idx < 0) or np.any(idx >= n_dof):
        raise ValueError(f"自由度索引越界：允许范围 0..{n_dof - 1}")
    mat = np.zeros((idx.size, n_dof))
    mat[np.arange(idx.size), idx] = 1.0
    return mat


def simulate_multi_setup_data(
    setups: list[dict],
    mode_shape: np.ndarray,
    n_dof: int,
    seed: int | None = None,
    q: int = 0,
) -> list[np.ndarray]:
    """生成多测组数据。

    各测组共用全局模态形状 ``mode_shape``（``(n, m)``），通过选择矩阵取得局部
    模态形状 ``t_r = L_r u``。不同测组可具有不同的采样频率、记录时长、测点
    子集、噪声水平，以及不同的频率与阻尼比（课本第 10.3 节允许如此）。

    Parameters
    ----------
    setups : list[dict]
        每个元素描述一个测组，可用字段：

        - ``dofs``：该测组对应的全局自由度索引（0 基），长度等于测点数；
        - ``fs``：采样频率（Hz）；
        - ``duration``：记录时长（s）；
        - ``Se``：通道噪声单边谱密度；
        - ``f``：该测组的固有频率（Hz），缺省取 ``freqs_hz``；
        - ``z``：该测组的阻尼比，缺省取 ``damping``；
        - ``S``：该测组的模态力谱密度矩阵，缺省取 ``S``。
    mode_shape : ndarray, shape (n, m)
        全局模态形状矩阵。
    n_dof : int
        全局自由度数。
    seed : int, optional
        随机数种子。
    q : int, optional
        数据类型指数。

    Returns
    -------
    list[ndarray]
        各测组的数据矩阵。
    """
    rng = np.random.default_rng(seed)
    glob = np.asarray(mode_shape, dtype=float)
    if glob.ndim == 1:
        # 单阶模态允许传一维全局振型，补成列向量，避免后续按 (n, m) 广播出错
        glob = glob.reshape(-1, 1)
    if glob.ndim != 2:
        raise ValueError("mode_shape 必须是一维向量或二维矩阵")
    if glob.shape[0] != n_dof:
        raise ValueError("mode_shape 的行数必须等于 n_dof")
    out = []
    for idx, spec in enumerate(setups):
        mat = selection_matrix(spec["dofs"], n_dof)
        local = mat @ glob
        data = simulate_oma_data(
            freqs_hz=np.asarray(spec["f"], dtype=float),
            damping=np.asarray(spec["z"], dtype=float),
            phi=local,
            S=np.asarray(spec["S"], dtype=complex),
            Se=float(spec["Se"]),
            fs=float(spec["fs"]),
            duration=float(spec["duration"]),
            seed=int(rng.integers(0, 2 ** 31 - 1)),
            q=q,
        )
        LOG.info(
            "测组 %d：%d 通道、%.1f s、fs=%.4g Hz", idx + 1, data.shape[1], spec["duration"], spec["fs"]
        )
        out.append(data)
    return out
