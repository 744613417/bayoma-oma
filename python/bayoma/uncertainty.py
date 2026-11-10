"""第 16 章不确定度定律的完整形式：Fisher 信息矩阵与参数间相关性。

第 15 章表 15.1 只给出各参数的边际变异系数，且已把阻尼比与模态力谱之间的相关性
（``q_{zeta S}``）解析地吸收进 ``B_zeta``。第 16 章表 16.2 直接给出长数据渐近下
NLLF 二阶导矩阵（即 Fisher 信息矩阵）的闭式，其逆同时给出边际方差与参数间
相关系数。本模块实现表 16.2 的闭式矩阵，用于：

* 给出 (f, ζ, S, S_e) 四参数的**相关矩阵**，供误差传递使用；
* 与 :func:`bayoma.auto_band.uncertainty_law_cov` 的边际结果互相校核；
* 检验表 16.4 的解析结果 ``σ²_ζ = 1/(L^{ζζ}(1 - q²_{ζS}))``。

约定与课本一致：频带取 ``f(1 ± κζ)``，``κ`` 为带宽因子；``N_f = 2κζfT_d``；
``N_c = N_f/(2κζ)``；``q = 0, 1, 2`` 分别对应加速度、速度、位移数据。表 16.2 的
各式对任意 ``q`` 成立，只需把噪声/环境比 ``ν = (2πf)^{2q} S_e/S`` 按 ``q`` 折算
（课本 16.6 节）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fisher_block",
    "posterior_matrix",
    "mode_shape_cov",
    "check_table_16_4",
]

# 表 16.2 / 表 16.4 中的参数顺序。
_PARAMETERS = ("f", "z", "S", "Se")


def fisher_block(
    f: float,
    z: float,
    S: float,
    Se: float,
    kappa: float,
    duration: float,
    nch: int,
    q: int = 0,
) -> np.ndarray:
    """表 16.2 的四参数 Fisher 信息矩阵（物理单位）。

    Parameters
    ----------
    f, z : float
        固有频率（Hz）与阻尼比。
    S, Se : float
        模态力谱密度与预测误差谱密度。
    kappa : float
        带宽因子。
    duration : float
        记录时长（s）。
    nch : int
        测点数。
    q : int, optional
        数据类型指数。

    Returns
    -------
    numpy.ndarray
        4×4 对称正定矩阵，顺序为 ``(f, z, S, Se)``；其逆为后验协方差矩阵。
    """
    f = float(f)
    z = float(z)
    S = float(S)
    Se = float(Se)
    nf = 2.0 * kappa * z * f * float(duration)
    nc = nf / (2.0 * kappa * z)
    at = np.arctan(kappa)
    k = float(kappa)
    k2 = k * k

    mat = np.zeros((4, 4))
    # (f, f)、(f, ζ)
    mat[0, 0] = 4.0 * nc / (f * f * z) * (at - k / (k2 + 1.0))
    mat[0, 1] = 4.0 * nc / f * (at + (2.0 * k ** 3 + k) / (k2 + 1.0) ** 2)
    # (f, S)、(f, Se)
    mat[0, 2] = -2.0 * nc * z / (f * S) * (2.0 * at + k ** 3 / (k2 + 1.0))
    mat[0, 3] = -8.0 * nc * z ** 3 / (3.0 * f * S) * (7.0 * k ** 3 + 6.0 * k)
    # (ζ, ζ)、(ζ, S)、(ζ, Se)
    mat[1, 1] = 4.0 * nc / z * (at + k / (k2 + 1.0))
    mat[1, 2] = -4.0 * nc / S * at
    mat[1, 3] = -16.0 * nc * z * z * k / S
    # (S, S)、(S, Se)、(Se, Se)
    mat[2, 2] = nf / (S * S)
    mat[2, 3] = 8.0 * nc * z ** 3 / (3.0 * S * S) * (k ** 3 + 3.0 * k)
    mat[3, 3] = (nch - 1) * nf / (Se * Se)
    return mat + np.triu(mat, 1).T


def posterior_matrix(
    f: float,
    z: float,
    S: float,
    Se: float,
    kappa: float,
    duration: float,
    nch: int,
    q: int = 0,
) -> dict[str, object]:
    """四参数后验协方差、相关系数与变异系数。

    Returns
    -------
    dict
        ``cov``（4×4，物理单位）、``coefv``（4 个变异系数）、``corr``（4×4 相关
        矩阵）、``fisher``（4×4 Fisher 信息矩阵）、``names``（参数名顺序）。
    """
    info = fisher_block(f, z, S, Se, kappa, duration, nch, q)
    cov = np.linalg.inv(info)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    base = np.array([float(f), float(z), float(S), float(Se)])
    return {
        "names": _PARAMETERS,
        "fisher": info,
        "cov": cov,
        "coefv": {name: float(sd[i] / base[i]) for i, name in enumerate(_PARAMETERS)},
        "corr": corr,
    }


def mode_shape_cov(
    f: float, z: float, S: float, Se: float, kappa: float, duration: float, nch: int, q: int = 0
) -> float:
    """表 16.4 的振型变异系数（振型协方差矩阵迹的平方根）。

    表 16.4 给出 ``C_phi = νζ/(N_c tan⁻¹κ) (I_n - φ̄φ̄ᵀ)``，其迹为
    ``νζ(n-1)/(N_c tan⁻¹κ)``，与表 15.1 的 ``2νζ/(πN_c B_phi)`` 在
    ``B_phi = (2/π)tan⁻¹κ`` 时完全一致。
    """
    nf = 2.0 * kappa * z * float(f) * float(duration)
    nc = nf / (2.0 * kappa * z)
    at = np.arctan(float(kappa))
    nu = (2.0 * np.pi * float(f)) ** (2 * int(q)) * float(Se) / float(S)
    return float(np.sqrt(nu * float(z) * (int(nch) - 1) / (nc * at)))


def check_table_16_4(
    f: float,
    z: float,
    S: float,
    Se: float,
    kappa: float,
    duration: float,
    nch: int,
    q: int = 0,
) -> dict[str, float]:
    """核对表 16.4 的两个解析结论。

    * 阻尼比方差应等于 ``1/(L^{ζζ}(1 - q²_{ζS}))``，其中
      ``q_{ζS} = L^{ζS}/sqrt(L^{ζζ} L^{SS})``；
    * 模态力谱方差应等于 ``1/(L^{SS}(1 - q²_{ζS}))``，即表 16.4 的
      ``S²/N_f [1 - 2(tan⁻¹κ)²/(κ(tan⁻¹κ + κ/(κ²+1)))]⁻¹`` 已含该修正；
      频率方差按表 16.4 只取 ``1/L^{ff}``，因 ``ρ_{fζ}`` 与 ``ρ_{fS}`` 均小。

    Returns
    -------
    dict[str, float]
        ``q_zeta_S``、``var_z_exact``、``var_z_table``、``var_S_exact``、
        ``var_S_table``、``rel_z``、``rel_S``。
    """
    info = fisher_block(f, z, S, Se, kappa, duration, nch, q)
    l_zz = info[1, 1]
    l_ss = info[2, 2]
    l_zs = info[1, 2]
    q_zs = l_zs / np.sqrt(l_zz * l_ss)
    one_minus = 1.0 - q_zs * q_zs
    var_z_exact = 1.0 / (l_zz * one_minus)
    var_s_exact = 1.0 / (l_ss * one_minus)

    nf = 2.0 * kappa * z * f * duration
    nc = nf / (2.0 * kappa * z)
    at = np.arctan(kappa)
    b_par = at + kappa / (kappa * kappa + 1.0)
    var_z_table = z / (4.0 * nc * (b_par - 2.0 * at * at / kappa))
    var_s_table = S * S / nf * 1.0 / (1.0 - 2.0 * at * at / (kappa * b_par))
    return {
        "q_zeta_S": float(q_zs),
        "var_z_exact": float(var_z_exact),
        "var_z_table": float(var_z_table),
        "var_S_exact": float(var_s_exact),
        "var_S_table": float(var_s_table),
        "rel_z": float(abs(var_z_exact - var_z_table) / var_z_table),
        "rel_S": float(abs(var_s_exact - var_s_table) / var_s_table),
    }
