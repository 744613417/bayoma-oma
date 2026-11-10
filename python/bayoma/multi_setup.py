"""多测组贝叶斯识别（课本第 14 章）。

同一结构在不同时段、以不同测点组合采集若干组环境振动数据时，各组测得的
局部模态形状是同一个全局模态形状在该组测点上的取值：

    t_r = L_r u,   u^T u = 1

其中 ``L_r`` 为第 r 组的测点选择矩阵（0/1 元素）。第 14.3 节把各组的模态力
谱密度与预测误差谱密度视为独立参数，仅共享全局模态形状 ``u``，由此构造的
负对数似然函数可写为各组贡献之和。

实现内容
--------
1. 各组缩放 FFT 与样本谱矩阵（式 14.21）。
2. 每组负对数似然函数（式 14.31），用 ``S0 = S ||t_r||^2`` 换元使每组子问题与
   单模态问题同形（式 14.26–14.33）。
3. 全局模态形状的部分最可能值：``B u = lambda_0 u``，``B`` 由式 14.40 给出，
   ``u`` 取 ``B`` 最小特征值的特征向量。
4. 第 14.3.3 节的三步迭代：先逐组更新 ``{f, z, S0, S_e}``，再更新 ``u``，
   再更新辅助参数 ``{lambda_r, c_r}``，循环至收敛。
5. 第 14.3.4 节高信噪比渐近结果用作初值与权重，第 14.3.6 节的渐近权重
   ``w_r`` 供全局最小二乘使用。

记号
----
- ``q``：测组数；``n_r``：第 r 组测点数；``n``：全局测点数
- ``N_f^{(r)}``：第 r 组频带内频率点数
- ``t_r = L_r u``：第 r 组局部模态形状；``t~_r = t_r / ||t_r||``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .condense import condensation_matrix, project_condensed
from scipy.optimize import minimize

from .single_mode import _SMIN, _ZMAX, _ZMIN, dynamic_factor, dynamic_factor_derivatives

_LOG_PI = float(np.log(np.pi))


@dataclass
class SetupData:
    """一个测组的频带数据。

    Attributes
    ----------
    freqs : ndarray, shape (Nf,)
        该组频带内的频率坐标（Hz）。
    fft : ndarray, shape (n_r, Nf), complex
        该组频带内的缩放 FFT。
    dofs : ndarray, shape (n_r,), int
        该组各通道对应的全局自由度编号（0 起）。
    """

    freqs: np.ndarray
    fft: np.ndarray
    dofs: np.ndarray

    @property
    def nch(self) -> int:
        return int(np.asarray(self.fft).shape[0])

    @property
    def nf(self) -> int:
        return int(np.asarray(self.freqs).size)

    @property
    def dhat(self) -> np.ndarray:
        fft = np.asarray(self.fft)
        return np.einsum("ak,bk->abk", fft, fft.conj()).real

    @property
    def d_total(self) -> float:
        return float(np.sum(np.abs(np.asarray(self.fft)) ** 2))


@dataclass
class MultiSetupResult:
    """多测组识别的输出。

    Attributes
    ----------
    f : ndarray, shape (q,)
        各测组的固有频率（Hz）。
    z : ndarray, shape (q,)
        各测组的阻尼比。
    S : ndarray, shape (q,)
        各测组的模态力谱密度（式 14.27 的 ``S``，已由 ``S0`` 换算）。
    Se : ndarray, shape (q,)
        各测组的预测误差谱密度。
    phi : ndarray, shape (n,)
        全局模态形状，欧氏范数为 1。
    local : list of ndarray
        各测组的局部模态形状 ``t_r = L_r phi``。
    nllf : float
        负对数似然函数值。
    n_iter : int
        外层迭代次数。
    converged : bool
        是否满足收敛判据。
    lam : ndarray, shape (q,)
        式 (14.41) 的拉格朗日乘子。
    c : ndarray, shape (q,)
        式 (14.37) 的辅助参数 ``c_r = u^T L_r^T L_r u``。
    """

    f: np.ndarray
    z: np.ndarray
    S: np.ndarray
    Se: np.ndarray
    phi: np.ndarray
    local: list = field(default_factory=list)
    nllf: float = np.nan
    n_iter: int = 0
    converged: bool = False
    lam: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c: np.ndarray = field(default_factory=lambda: np.zeros(0))


def selection_matrix(dofs: np.ndarray, n_dof: int) -> np.ndarray:
    """构造测点选择矩阵 ``L_r``，形状 ``(n_r, n)``。"""
    idx = np.asarray(dofs, dtype=int).reshape(-1)
    nch = idx.size
    mat = np.zeros((nch, int(n_dof)), dtype=float)
    mat[np.arange(nch), idx] = 1.0
    return mat


def setup_nllf(
    f: float,
    z: float,
    s0: float,
    se: float,
    freqs: np.ndarray,
    dhat: np.ndarray,
    d_total: float,
    tshape: np.ndarray,
    q: int,
) -> float:
    """单测组负对数似然函数，式 (14.31)。

    ``t̃_r`` 取单位范数，``S0`` 为式 (14.27) 换元后的模态力谱密度。该式与单模态
    问题的降维 NLLF 同形，区别在于振型固定、不再对 ``u`` 取最大特征值。

    Parameters
    ----------
    tshape : ndarray, shape (n_r,)
        该组局部模态形状（单位范数）。
    """
    nch = dhat.shape[0]
    nf = freqs.size
    if not np.isfinite(se) or se <= 0.0 or s0 <= 0.0 or f <= 0.0 or not (0.0 < z < 1.0):
        return np.inf
    dk = dynamic_factor(float(f), float(z), freqs, q)
    if np.any(~np.isfinite(dk)):
        return np.inf
    coeff = s0 * dk
    denom = coeff + se
    if np.any(denom <= 0.0):
        return np.inf
    weight = 1.0 / (1.0 + se / np.maximum(coeff, 1.0e-300))
    quad = np.einsum("a,abk,b->k", tshape, dhat, tshape)
    total = nch * nf * _LOG_PI
    total += float(np.sum(np.log(denom)))
    total += (nch - 1) * nf * np.log(se)
    # 式 (14.31) 的最后一项：d̂ 只出现一次，Σ_k B_k q_k 与之相减后再除以 S_e
    total += (float(d_total) - float(np.sum(weight * quad))) / se
    return float(total)


def _spectrum_subproblem(
    freqs: np.ndarray,
    dhat: np.ndarray,
    d_total: float,
    tshape: np.ndarray,
    q: int,
    f_init: float,
    z_init: float,
    s0_init: float,
    se_init: float,
    max_iter: int = 200,
) -> tuple[float, float, float, float, float]:
    """固定局部模态形状，极小化式 (14.31) 求该组的谱参数。

    返回 ``(f, z, S0, S_e, nllf)``。在无量纲参数空间（相对初值的倍率）作有界
    拟牛顿极小化，避免频率与谱密度的量级差异引起的病态。
    """

    def to_x(f, z, s0, se):
        return np.array([np.log(f), np.log(z), np.log(s0), np.log(se)])

    def unpack(x):
        return float(np.exp(x[0])), float(np.exp(x[1])), float(np.exp(x[2])), float(np.exp(x[3]))

    x0 = to_x(f_init, z_init, s0_init, se_init)
    lower = np.array(
        [
            np.log(max(freqs[0] * 0.5, 1.0e-12)),
            np.log(_ZMIN),
            np.log(_SMIN),
            np.log(_SMIN),
        ]
    )
    upper = np.array([np.log(freqs[-1] * 2.0), np.log(_ZMAX), np.log(1.0e30), np.log(1.0e30)])

    def obj(x):
        f, z, s0, se = unpack(x)
        return setup_nllf(f, z, s0, se, freqs, dhat, d_total, tshape, q)

    res = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": int(max_iter), "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
    )
    x = res.x if np.all(np.isfinite(res.x)) else x0
    f, z, s0, se = unpack(x)
    return f, z, s0, se, float(obj(x))


def mpv(
    setups: list,
    n_dof: int,
    q: int = 0,
    max_iter: int = 100,
    tol: float = 1.0e-8,
    condense: bool = True,
) -> MultiSetupResult:
    """多测组最可能值，第 14.3.3 节。

    Parameters
    ----------
    setups : list of SetupData
        各测组的频带数据，频带与模态阶次应对应同一阶模态。
    n_dof : int
        全局自由度数 ``n``。
    q : int, optional
        数据类型指数。
    max_iter : int, optional
        外层迭代上限，默认 100。
    tol : float, optional
        收敛容差，默认 ``1e-8``。
    condense : bool, optional
        是否启用参考自由度凝聚（式 14.17--14.20），默认启用。启用后
        Step 2 的特征值问题在压缩后的 ``n00 + q`` 维子空间内求解，
        对多测组且测点重叠较多的情形可显著降低计算量。

    Returns
    -------
    MultiSetupResult

    Notes
    -----
    迭代按课本分三步：Step 1 逐组更新 ``{f_r, z_r, S0_r, S_e_r}``（式 14.31）；
    Step 2 取式 (14.40) 中 ``B`` 最小特征值的特征向量作为全局模态形状；
    Step 3 用式 (14.41) 与式 (14.37) 更新 ``{lambda_r, c_r}``。

    Step 2 的特征向量确定到一个复相位；此处取绝对值最大的分量成为正实数，
    并把该相位差吸收进 ``S``（``E_k`` 的二次型不变），使输出可比。
    """
    qn = len(setups)
    if qn < 1:
        raise ValueError("至少需要一个测组")
    n = int(n_dof)

    mats = [selection_matrix(s.dofs, n) for s in setups]
    basis = None
    if condense:
        dofs_per_setup = [np.asarray(s.dofs, dtype=int).ravel() for s in setups]
        basis, _ = condensation_matrix(dofs_per_setup, n)
    dhats = [s.dhat for s in setups]
    dtot = [s.d_total for s in setups]
    freqs = [np.asarray(s.freqs, dtype=float) for s in setups]

    # 初值：各组按第一条奇异值谱曲线的谱峰取频率，阻尼比 1%，谱密度按式 (14.43)
    fs = np.empty(qn)
    zs = np.full(qn, 0.01)
    for r, (sr, fr) in enumerate(zip(setups, freqs)):
        sv = np.linalg.eigvalsh(dhats[r].transpose(2, 0, 1))
        main = sv[:, -1]
        fs[r] = float(fr[int(np.argmax(main))])

    # 全局模态形状初值：式 (14.48) 的最小特征值特征向量
    a0_list = []
    peak_list = []
    for r in range(qn):
        a0 = np.einsum("ijk->ij", dhats[r])
        a0 = 0.5 * (a0 + a0.T)
        vals = np.linalg.eigvalsh(a0)
        peak_list.append(float(vals[-1]))
        a0_list.append(a0)
    b0 = np.zeros((n, n))
    for r in range(qn):
        lr = mats[r]
        red = peak_list[r] * np.eye(lr.shape[0]) - a0_list[r]
        b0 += lr.T @ red @ lr
    if basis is not None:
        u, _ = project_condensed(0.5 * (b0 + b0.T), basis)
    else:
        vals, vecs = np.linalg.eigh(0.5 * (b0 + b0.T))
        u = vecs[:, 0]
        u = u / np.linalg.norm(u)

    # 谱密度初值：式 (14.43)
    s0 = np.empty(qn)
    se = np.empty(qn)
    for r in range(qn):
        tr = mats[r] @ u
        tr = tr / max(np.linalg.norm(tr), 1.0e-300)
        dk = dynamic_factor(fs[r], zs[r], freqs[r], q)
        quad = np.einsum("a,abk,b->k", tr, dhats[r], tr)
        nr, nf = dhats[r].shape[0], freqs[r].size
        se[r] = max((dtot[r] - float(np.sum(quad))) / max((nr - 1) * nf, 1), 1.0e-30)
        s0[r] = max(float(np.sum(quad / np.maximum(dk, 1.0e-300))) / nf, 1.0e-30)

    lam = np.zeros(qn)
    cval = np.empty(qn)
    nllf = np.inf
    converged = False
    n_iter = 0

    for n_iter in range(1, int(max_iter) + 1):
        u_old = u.copy()
        spec_old = np.concatenate([fs, zs, np.log(s0), np.log(se)])

        # Step 1：逐组更新谱参数
        for r in range(qn):
            tr = mats[r] @ u
            nrm = np.linalg.norm(tr)
            if nrm <= 0.0:
                raise ValueError(f"第 {r + 1} 组测点上的全局模态形状为零")
            tr = tr / nrm
            fs[r], zs[r], s0[r], se[r], _ = _spectrum_subproblem(
                freqs[r], dhats[r], dtot[r], tr, q, fs[r], zs[r], s0[r], se[r]
            )

        # 各组 A^{(r)} 与辅助量
        avec = []
        tvec = []
        for r in range(qn):
            tr = mats[r] @ u
            nrm = np.linalg.norm(tr)
            tr = tr / nrm
            tvec.append(tr)
            dk = dynamic_factor(fs[r], zs[r], freqs[r], q)
            w = 1.0 / (1.0 + se[r] / np.maximum(s0[r] * dk, 1.0e-300))
            avec.append(np.einsum("k,abk->ab", w, dhats[r]))

        # Step 2：式 (14.40)-(14.41)，u 取 B 最小特征值的特征向量
        for r in range(qn):
            lam[r] = float(tvec[r] @ avec[r] @ tvec[r] / (se[r] * (tvec[r] @ tvec[r]) ** 2))
        bmat = np.zeros((n, n))
        for r in range(qn):
            lr = mats[r]
            bmat += lam[r] * (lr.T @ lr)
        amat = np.zeros((n, n))
        for r in range(qn):
            lr = mats[r]
            nrm2 = float(tvec[r] @ tvec[r])
            amat += (lr.T @ avec[r] @ lr) / (nrm2 * se[r])
        bmat = bmat - amat
        bmat = 0.5 * (bmat + bmat.T)
        if basis is not None:
            u, _ = project_condensed(bmat, basis)
        else:
            vals, vecs = np.linalg.eigh(bmat)
            u = vecs[:, 0]
            u = u / np.linalg.norm(u)

        # Step 3：式 (14.37) 更新 c_r
        for r in range(qn):
            lr = mats[r]
            cval[r] = float(u @ (lr.T @ lr) @ u)

        # 总负对数似然函数，式 (14.34) 中的可加部分
        nllf = 0.0
        for r in range(qn):
            tr = mats[r] @ u
            tr = tr / np.linalg.norm(tr)
            nllf += setup_nllf(fs[r], zs[r], s0[r], se[r], freqs[r], dhats[r], dtot[r], tr, q)

        spec_new = np.concatenate([fs, zs, np.log(s0), np.log(se)])
        step_u = float(np.max(np.abs(u - u_old)))
        step_s = float(np.max(np.abs(spec_new / np.where(np.abs(spec_old) > 0, spec_old, 1.0) - 1.0)))
        if step_u < tol and step_s < tol:
            converged = True
            break

    # 相位基准：使全局模态形状绝对值最大的分量为正实数
    idx = int(np.argmax(np.abs(u)))
    theta = float(np.angle(u[idx])) if abs(u[idx]) > 0.0 else 0.0
    if theta != 0.0:
        u = u * np.exp(-1j * theta)

    local = []
    s_phys = np.empty(qn)
    for r in range(qn):
        tr = mats[r] @ u
        local.append(tr)
        # 式 (14.27) 反算：S = S0 / ||t_r||^2
        s_phys[r] = s0[r] / max(float(np.vdot(tr, tr).real), 1.0e-300)

    return MultiSetupResult(
        f=np.asarray(fs, dtype=float),
        z=np.asarray(zs, dtype=float),
        S=s_phys,
        Se=np.asarray(se, dtype=float),
        phi=u,
        local=local,
        nllf=float(nllf),
        n_iter=int(n_iter),
        converged=bool(converged),
        lam=np.asarray(lam, dtype=float),
        c=np.asarray(cval, dtype=float),
    )


def asymptotic_weights(result: MultiSetupResult, setups: list, q: int = 0) -> np.ndarray:
    """全局最小二乘的渐近权重，式 (14.50)。

    ``w_r ∝ a_{n_r}^{(r)} / (S_e^{(r)} ||t̂_r||^2)``，其中 ``a_{n_r}^{(r)}`` 为
    ``A0^{(r)} = sum_k D_k`` 的最大特征值。该权重使全局最小二乘解渐近等于贝叶斯
    最可能值。
    """
    ws = np.empty(len(setups))
    for r, sr in enumerate(setups):
        a0 = np.einsum("ijk->ij", np.asarray(sr.dhat))
        a0 = 0.5 * (a0 + a0.T)
        vals = np.linalg.eigvalsh(a0)
        tr = result.local[r]
        ws[r] = float(vals[-1]) / (result.Se[r] * max(float(np.vdot(tr, tr).real), 1.0e-300))
    total = float(np.sum(ws))
    return ws / total if total > 0.0 else np.full(ws.size, 1.0 / ws.size)
