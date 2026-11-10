"""单模态贝叶斯识别（课本第 12 章）。

本模块实现课本第 12 章的负对数似然函数改写、最可能值（MPV）迭代算法与
后验协方差矩阵解析计算，对应式 (12.10)–(12.27) 与表 12.1–12.6。

记号
----
- ``n``   ：测点数
- ``N_f`` ：频带内 FFT 点数
- ``D_k`` ：动力放大因子，式 (12.4)
- ``B_k`` ：``(1 + S_e/(S D_k))^{-1}``，表 12.6
- ``A``   ：``sum_k B_k D_k``，式 (12.11)
- ``d``   ：``sum_k F_k^* F_k``，式 (12.12)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ._util import LOG

# 允许的最小阻尼比与频率，防止优化过程中越界到非物理区域。
_ZMIN = 1.0e-6
_ZMAX = 0.5
_SMIN = 1.0e-30


@dataclass
class SingleModeResult:
    """单模态识别的输出。

    Attributes
    ----------
    f : float
        固有频率（Hz）。
    z : float
        阻尼比。
    S : float
        模态力谱密度。
    Se : float
        预测误差（通道噪声）谱密度。
    phi : ndarray, shape (n,)
        模态形状，欧氏范数为 1。
    sn : float
        模态信噪比 ``S/(4 S_e z^2)``。
    nllf : float
        极小化后的负对数似然函数值。
    n_iter : int
        外层迭代次数。
    converged : bool
        是否满足收敛判据。
    """

    f: float
    z: float
    S: float
    Se: float
    phi: np.ndarray
    sn: float
    nllf: float
    n_iter: int
    converged: bool


def dynamic_factor(f: float, z: float, freqs: np.ndarray, q: int) -> np.ndarray:
    """动力放大因子 ``D_k``，式 (12.4)。

    ``q=0`` 对应加速度数据，``q=1`` 速度，``q=2`` 位移。
    """
    beta = f / freqs
    denom = (1.0 - beta ** 2) ** 2 + (2.0 * z * beta) ** 2
    return (2.0 * np.pi * freqs) ** (-2.0 * q) / denom


def dynamic_factor_derivatives(
    f: float, z: float, freqs: np.ndarray, q: int
) -> dict[str, np.ndarray]:
    """``D_k`` 对 ``f`` 与 ``z`` 的一、二阶导数，对应表 12.2。

    表 12.2 给出 ``D_k^{-1}`` 的导数形式；此处直接对 ``D_k`` 求导，
    两者在数学上等价，实现后以有限差分交叉校核。
    """
    beta = f / freqs
    om = (2.0 * np.pi * freqs) ** (-2.0 * q)
    b2 = beta ** 2
    re = (1.0 - b2) ** 2 + (2.0 * z * beta) ** 2  # D^{-1} / om
    dk = om / re

    # re 对 f、z 的导数
    re_f = 4.0 * beta * (b2 - 1.0 + 2.0 * z ** 2) / f
    re_z = 8.0 * z * b2

    # re 的二阶导数
    re_ff = 4.0 * (3.0 * b2 - 1.0 + 2.0 * z ** 2) / (f * f)
    re_fz = 16.0 * z * beta / f
    re_zz = 8.0 * b2

    dk_f = -om * re_f / re ** 2
    dk_z = -om * re_z / re ** 2
    dk_ff = 2.0 * om * re_f ** 2 / re ** 3 - om * re_ff / re ** 2
    dk_fz = 2.0 * om * re_f * re_z / re ** 3 - om * re_fz / re ** 2
    dk_zz = 2.0 * om * re_z ** 2 / re ** 3 - om * re_zz / re ** 2

    return {
        "D": dk,
        "f": dk_f,
        "z": dk_z,
        "ff": dk_ff,
        "fz": dk_fz,
        "zz": dk_zz,
    }


def nllf_reduced(
    f: float,
    z: float,
    S: float,
    Se: float,
    freqs: np.ndarray,
    dhat: np.ndarray,
    d_total: float,
    q: int,
) -> float:
    """降维负对数似然函数 ``L~``，式 (12.13)。

    ``dhat`` 为 ``D_k = Re(F_k F_k^*)`` 组成的数组，形状 ``(n, n, N_f)``。
    模态形状已被解析消去，``u^T A u`` 取 ``A`` 的最大特征值。
    """
    nch = dhat.shape[0]
    nf = dhat.shape[2]
    dk = dynamic_factor(f, z, freqs, q)
    p = S * dk + Se
    if np.any(p <= 0.0):
        return np.inf
    weights = S * dk / p  # B_k
    amat = np.einsum("k,ijk->ij", weights, dhat)
    amat = 0.5 * (amat + amat.T)
    lam = float(np.linalg.eigvalsh(amat)[-1])
    return float(
        nch * nf * np.log(np.pi)
        + np.sum(np.log(p))
        + (nch - 1) * nf * np.log(Se)
        + (d_total - lam) / Se
    )


def _initial_guess(
    freqs: np.ndarray, dhat: np.ndarray, d_total: float, q: int
) -> tuple[float, float, float, float, np.ndarray]:
    """按式 (12.20) 与第 12.3.1 节构造初值。"""
    nch = dhat.shape[0]
    nf = dhat.shape[2]
    a0 = np.einsum("ijk->ij", dhat)
    a0 = 0.5 * (a0 + a0.T)
    vals, vecs = np.linalg.eigh(a0)
    phi = vecs[:, -1]
    lam0 = float(vals[-1])

    # 频率初值取奇异值谱峰值
    sv = np.empty((nch, nf))
    for k in range(nf):
        sv[:, k] = np.linalg.eigvalsh(dhat[:, :, k])[::-1]
    f0 = float(freqs[int(np.argmax(sv[0]))])
    z0 = 0.01

    dk = dynamic_factor(f0, z0, freqs, q)
    dhat_k = np.einsum("i,ijk,j->k", phi, dhat, phi)
    Se0 = max((d_total - lam0) / ((nch - 1) * nf), 1.0e-30)
    S0 = max(float(np.mean(dhat_k / dk)), 1.0e-30)
    return f0, z0, S0, Se0, phi


def mpv(
    freqs: np.ndarray,
    dhat: np.ndarray,
    d_total: float,
    q: int = 0,
    f0: float | None = None,
    z0: float = 0.01,
    max_iter: int = 500,
    tol: float = 1.0e-6,
    n_iter_inner: int = 200,
) -> SingleModeResult:
    """单模态最可能值（MPV）迭代算法，第 12.2–12.3 节。

    采用课本 Algorithm 2 的交替格式：给定模态形状时用式 (12.13) 的降维目标
    函数更新 ``{f, z, S, S_e}``；给定谱参数时取 ``A`` 的最大特征向量更新模态
    形状。两者交替至收敛。

    Parameters
    ----------
    freqs : ndarray, shape (N_f,)
        频带内频率坐标。
    dhat : ndarray, shape (n, n, N_f)
        ``D_k = Re(F_k F_k^*)``。
    d_total : float
        ``d = sum_k F_k^* F_k``。
    q : int, optional
        数据类型指数，``0`` 加速度、``1`` 速度、``2`` 位移。
    f0 : float, optional
        频率初值；缺省取奇异值谱峰值。
    z0 : float, optional
        阻尼比初值，默认 0.01。
    max_iter : int, optional
        外层迭代上限。
    tol : float, optional
        收敛容差，用于判断参数相对变化。
    n_iter_inner : int, optional
        内层谱参数优化的迭代上限。

    Returns
    -------
    SingleModeResult
    """
    nch = dhat.shape[0]
    nf = dhat.shape[2]
    freqs = np.asarray(freqs, dtype=float)

    f_ini, z_ini, S_ini, Se_ini, phi = _initial_guess(freqs, dhat, d_total, q)
    f_cur = float(f0) if f0 is not None else f_ini
    z_cur = float(z0) if z0 is not None else z_ini

    # 对数参数化，保证谱参数与频率、阻尼比始终为正
    x = np.array([np.log(f_cur), np.log(z_cur), np.log(S_ini), np.log(Se_ini)])
    lb = np.array([np.log(freqs[0] * 0.5), np.log(_ZMIN), np.log(_SMIN), np.log(_SMIN)])
    ub = np.array([np.log(freqs[-1] * 2.0), np.log(_ZMAX), np.log(1.0e30), np.log(1.0e30)])

    def objective(xv: np.ndarray) -> float:
        fv, zv, Sv, Sev = np.exp(xv)
        return nllf_reduced(fv, zv, Sv, Sev, freqs, dhat, d_total, q)

    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        res = minimize(
            objective,
            x,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxiter": n_iter_inner, "ftol": 1.0e-12, "gtol": 1.0e-10},
        )
        x_new = res.x if np.all(np.isfinite(res.x)) else x
        f_new, z_new, S_new, Se_new = np.exp(x_new)

        # 给定谱参数，取 A 的最大特征向量更新模态形状（式 12.11）
        dk = dynamic_factor(f_new, z_new, freqs, q)
        p = S_new * dk + Se_new
        weights = S_new * dk / p
        amat = np.einsum("k,ijk->ij", weights, dhat)
        amat = 0.5 * (amat + amat.T)
        vals, vecs = np.linalg.eigh(amat)
        phi_new = vecs[:, -1]

        rel = np.max(
            np.abs(
                [
                    (f_new - np.exp(x[0])) / max(f_new, 1.0e-30),
                    (z_new - np.exp(x[1])) / max(z_new, 1.0e-30),
                    (S_new - np.exp(x[2])) / max(S_new, 1.0e-30),
                    (Se_new - np.exp(x[3])) / max(Se_new, 1.0e-30),
                ]
            )
        )
        x = x_new
        phi = phi_new
        if rel < tol:
            converged = True
            break

    f_mpv, z_mpv, S_mpv, Se_mpv = np.exp(x)
    phi = phi / np.linalg.norm(phi)
    # 固定符号，使最大绝对值分量恒为正
    if phi[int(np.argmax(np.abs(phi)))] < 0.0:
        phi = -phi

    nllf_val = nllf_reduced(f_mpv, z_mpv, S_mpv, Se_mpv, freqs, dhat, d_total, q)
    sn = float(S_mpv / (4.0 * Se_mpv * z_mpv ** 2)) if Se_mpv > 0.0 else np.inf

    return SingleModeResult(
        f=float(f_mpv),
        z=float(z_mpv),
        S=float(S_mpv),
        Se=float(Se_mpv),
        phi=phi,
        sn=sn,
        nllf=float(nllf_val),
        n_iter=int(n_iter),
        converged=bool(converged),
    )


def posterior_covariance(
    result: SingleModeResult,
    freqs: np.ndarray,
    dhat: np.ndarray,
    d_total: float,
    q: int = 0,
) -> np.ndarray:
    """单模态后验协方差矩阵，第 12.4 节与式 (12.21)–(12.31)。

    参数顺序为 ``[f, z, S, S_e, phi_1, ..., phi_n]``，返回 ``(n+4, n+4)``
    的协方差矩阵。模态形状的范数约束通过约束映射 ``v_c`` 与拉格朗日乘子
    ``lambda_1 = -S_e^{-1} lambda`` 处理。

    Notes
    -----
    返回值在模态形状分量的符号上与 ``result.phi`` 一致；模态形状的协方差
    只在 ``(I - phi phi^T)`` 方向上有意义，沿 ``phi`` 方向的分量为零。
    """
    f = result.f
    z = result.z
    S = result.S
    Se = result.Se
    phi = np.asarray(result.phi, dtype=float)
    nch = phi.size
    nf = freqs.size
    freqs = np.asarray(freqs, dtype=float)

    der = dynamic_factor_derivatives(f, z, freqs, q)
    dk = der["D"]
    p = S * dk + Se
    dhat_k = np.einsum("i,ijk,j->k", phi, dhat, phi)
    qsum = np.einsum("k,k->", S * dk / p, dhat_k)  # sum_k B_k q_k

    # ---- L_D 的二阶导数（表 12.4） ----
    def sum_(a, b):
        return float(np.sum(a * b))

    ld_ff = sum_(S * der["ff"] / p, np.ones(nf)) - sum_((S * der["f"]) ** 2 / p ** 2, np.ones(nf))
    ld_fz = sum_(S * der["fz"] / p, np.ones(nf)) - sum_(S ** 2 * der["f"] * der["z"] / p ** 2, np.ones(nf))
    ld_fS = sum_(der["f"] * Se / p ** 2, np.ones(nf))
    ld_fSe = sum_(-S * der["f"] / p ** 2, np.ones(nf))
    ld_zz = sum_(S * der["zz"] / p, np.ones(nf)) - sum_((S * der["z"]) ** 2 / p ** 2, np.ones(nf))
    ld_zS = sum_(der["z"] * Se / p ** 2, np.ones(nf))
    ld_zSe = sum_(-S * der["z"] / p ** 2, np.ones(nf))
    ld_SS = -sum_(dk ** 2 / p ** 2, np.ones(nf))
    ld_SSe = -sum_(dk / p ** 2, np.ones(nf))
    ld_SeSe = -sum_(1.0 / p ** 2, np.ones(nf)) - (nch - 1) * nf / Se ** 2

    # ---- B_k 的导数（表 12.6 的等价形式） ----
    b_d = Se * S * der["f"] / p ** 2
    b_z = Se * S * der["z"] / p ** 2
    b_S = Se * dk / p ** 2
    b_Se = -S * dk / p ** 2

    b_ff = Se * S * der["ff"] / p ** 2 - 2.0 * Se * S ** 2 * der["f"] ** 2 / p ** 3
    b_fz = Se * S * der["fz"] / p ** 2 - 2.0 * Se * S ** 2 * der["f"] * der["z"] / p ** 3
    b_zz = Se * S * der["zz"] / p ** 2 - 2.0 * Se * S ** 2 * der["z"] ** 2 / p ** 3
    b_fS = Se * der["f"] * (Se - S * dk) / p ** 3
    b_fSe = S * der["f"] * (S * dk - Se) / p ** 3
    b_zS = Se * der["z"] * (Se - S * dk) / p ** 3
    b_zSe = S * der["z"] * (S * dk - Se) / p ** 3
    b_SS = -2.0 * Se * dk ** 2 / p ** 3
    b_SSe = dk * (S * dk - Se) / p ** 3
    b_SeSe = 2.0 * S * dk / p ** 3

    # ---- L_Q 的二阶导数（表 12.5） ----
    lq_ff = -sum_(b_ff, dhat_k) / Se
    lq_fz = -sum_(b_fz, dhat_k) / Se
    lq_fS = -sum_(b_fS, dhat_k) / Se
    lq_fSe = sum_(b_d, dhat_k) / Se ** 2 - sum_(b_fSe, dhat_k) / Se
    lq_zz = -sum_(b_zz, dhat_k) / Se
    lq_zS = -sum_(b_zS, dhat_k) / Se
    lq_zSe = sum_(b_z, dhat_k) / Se ** 2 - sum_(b_zSe, dhat_k) / Se
    lq_SS = -sum_(b_SS, dhat_k) / Se
    lq_SSe = sum_(b_S, dhat_k) / Se ** 2 - sum_(b_SSe, dhat_k) / Se
    lq_SeSe = (
        2.0 * (d_total - qsum) / Se ** 3
        + 2.0 * sum_(b_Se, dhat_k) / Se ** 2
        - sum_(b_SeSe, dhat_k) / Se
    )

    h_pp = np.array(
        [
            [ld_ff + lq_ff, ld_fz + lq_fz, ld_fS + lq_fS, ld_fSe + lq_fSe],
            [ld_fz + lq_fz, ld_zz + lq_zz, ld_zS + lq_zS, ld_zSe + lq_zSe],
            [ld_fS + lq_fS, ld_zS + lq_zS, ld_SS + lq_SS, ld_SSe + lq_SSe],
            [ld_fSe + lq_fSe, ld_zSe + lq_zSe, ld_SSe + lq_SSe, ld_SeSe + lq_SeSe],
        ]
    )

    # ---- 混合块 L^(x u)（表 12.3）与 H_uu ----
    dhat_phi = np.einsum("ijk,j->ik", dhat, phi)  # (n, Nf) 即 D_k phi
    a_mat = np.einsum("k,ijk->ij", S * dk / p, dhat)
    a_mat = 0.5 * (a_mat + a_mat.T)

    row_f = -2.0 * np.sum(b_d[None, :] * dhat_phi, axis=1) / Se
    row_z = -2.0 * np.sum(b_z[None, :] * dhat_phi, axis=1) / Se
    row_S = -2.0 * np.sum(b_S[None, :] * dhat_phi, axis=1) / Se
    row_Se = -2.0 * np.sum(b_Se[None, :] * dhat_phi, axis=1) / Se + 2.0 * (a_mat @ phi) / Se ** 2
    h_pu = np.vstack([row_f, row_z, row_S, row_Se])  # (4, n)

    h_uu = -2.0 * a_mat / Se

    nh = 4 + nch
    hess = np.zeros((nh, nh))
    hess[:4, :4] = h_pp
    hess[:4, 4:] = h_pu
    hess[4:, :4] = h_pu.T
    hess[4:, 4:] = h_uu

    # ---- 约束处理：G = 1 - phi^T phi，lambda_1 = -Se^{-1} lambda ----
    lam = float(np.linalg.eigvalsh(a_mat)[-1])
    lam1 = -lam / Se

    proj = np.eye(nch) - np.outer(phi, phi)
    vmap = np.zeros((nh, nh))
    vmap[:4, :4] = np.eye(4)
    vmap[4:, 4:] = proj

    g_hess = np.zeros((nh, nh))
    g_hess[4:, 4:] = -2.0 * np.eye(nch)

    hc = vmap.T @ (hess + lam1 * g_hess) @ vmap
    hc = 0.5 * (hc + hc.T)

    # 谱参数（S、Se 的量级随数据单位变化，可从 1e-12 到 1e6）与振型分量（无量纲）相差
    # 若干数量级，直接求伪逆会把小量级方向判成零曲率，使对应参数的方差退化为零。
    # 先用对角元做雅可比预条件，令各自由参数在对角尺度上同量级，求逆后再还原。
    scale = np.sqrt(np.abs(np.diag(hc)))
    scale = np.where(scale > 0.0, scale, 1.0)
    hcs = hc / np.outer(scale, scale)
    cov_scaled = np.linalg.pinv(hcs, rcond=1.0e-12)
    cov_free = cov_scaled / np.outer(scale, scale)
    cov = vmap @ cov_free @ vmap.T
    return 0.5 * (cov + cov.T)


def coefficients_of_variation(
    result: SingleModeResult, cov: np.ndarray
) -> dict[str, float]:
    """由协方差矩阵提取各参数的后验变异系数。"""
    diag = np.diag(cov)
    out = {
        "f": float(np.sqrt(max(diag[0], 0.0)) / result.f),
        "z": float(np.sqrt(max(diag[1], 0.0)) / result.z),
        "S": float(np.sqrt(max(diag[2], 0.0)) / result.S),
        "Se": float(np.sqrt(max(diag[3], 0.0)) / result.Se),
        "phi": float(np.sqrt(max(np.sum(diag[4:]), 0.0))),
    }
    # 模态加速度 rms 的变异系数由 f、z、S 及其相关性按一阶小扰动传播
    rms = np.sqrt(np.pi * result.S * result.f / (4.0 * result.z))
    grad = np.array(
        [
            0.5 * rms / result.f,
            -0.5 * rms / result.z,
            0.5 * rms / result.S,
        ]
    )
    sub = cov[:3, :3]
    var_rms = float(grad @ sub @ grad)
    out["rms"] = float(np.sqrt(max(var_rms, 0.0)) / rms)
    return out
