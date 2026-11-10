"""后验协方差计算（课本 12.4、13.7、14.4 节）。

后验协方差由负对数似然函数在极大似然点处的 Hessian 求得：

    C = H^{-1},  H = ∇²L

模态形状受单位范数约束 ``u_i^H u_i = 1``，其协方差只在与 ``u_i`` 正交的切空间内
有定义，沿 ``u_i`` 方向的分量恒为零。本模块用切空间参数化处理该约束：为每阶模态
形状取一组与 ``u_i`` 正交的归一化基 ``Z_i``，令

    u_i(θ) = (u_i + Z_i θ) / ||u_i + Z_i θ||

则 ``θ = 0`` 处的导数 ``∂u_i/∂θ = Z_i``，自由参数向量取
``[谱参数, θ_1, ..., θ_m]``。在该坐标系中直接对负对数似然做数值微分，无需引入
拉格朗日乘子，也自然满足单位范数约束；模态形状的协方差由 ``C_u = Z C_θ Z^T``
还原，其对角元之和等于 ``trace(C_θ)``。

数值微分前先把谱参数按其极大似然值的绝对值无量纲化：谱参数量级可达 1e2–1e6，
模态形状分量量级约 1e-1，若在物理单位下求 Hessian，逆矩阵的条件数过大，方差
估计会被舍入误差主导。

复数模态形状按实部与虚部分别展开为实向量处理，协方差为 ``2n × 2n`` 实矩阵；
变异系数取全部 ``2n`` 个分量方差之和的平方根，实数情形下退化为课本的
``sqrt(sum(diag(C_u)))``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PosteriorResult:
    """后验协方差结果。

    Attributes
    ----------
    cov : ndarray, shape (p, p)
        无量纲参数协方差矩阵，参数按 ``[谱参数, 切空间坐标]`` 排序。谱参数按其
        极大似然值归一，故对角线元素即变异系数的平方。
    coefv : ndarray
        各自由参数的后验变异系数，顺序与 ``cov`` 一致。
    coefv_named : dict
        按参数名分组的变异系数，键为 ``f``、``z``、``S``、``Se``、``phi``。
    cov_phi : list of ndarray
        各阶模态形状在物理通道空间中的协方差矩阵；实数形状为 ``n × n``，
        复数形状为 ``2n × 2n``（实部在前、虚部在后）。
    """

    cov: np.ndarray
    coefv: np.ndarray
    coefv_named: dict = field(default_factory=dict)
    cov_phi: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _hessian(func, y: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """中心差分 Hessian。"""
    n = y.size
    hess = np.zeros((n, n))
    f0 = func(y)
    for i in range(n):
        yp = y.copy()
        ym = y.copy()
        yp[i] += steps[i]
        ym[i] -= steps[i]
        hess[i, i] = (func(yp) - 2.0 * f0 + func(ym)) / (steps[i] * steps[i])
    for i in range(n):
        for j in range(i + 1, n):
            ypp = y.copy()
            ypm = y.copy()
            ymp = y.copy()
            ymm = y.copy()
            ypp[i] += steps[i]
            ypp[j] += steps[j]
            ypm[i] += steps[i]
            ypm[j] -= steps[j]
            ymp[i] -= steps[i]
            ymp[j] += steps[j]
            ymm[i] -= steps[i]
            ymm[j] -= steps[j]
            val = (func(ypp) - func(ypm) - func(ymp) + func(ymm)) / (4.0 * steps[i] * steps[j])
            hess[i, j] = val
            hess[j, i] = val
    return hess


def _regularised_inverse(hess: np.ndarray) -> np.ndarray:
    """对称矩阵的正则化逆。

    数值微分得到的 Hessian 在弱可辨识方向上的特征值接近零，直接求逆会把舍入误差
    放大成虚假的巨大方差；带截断阈值的伪逆则把这类方向判为零方差，给出参数精确
    已知的错误结论。这里按最大特征值取相对下限，只把非正与近零特征值抬到该下限，
    使弱可辨识方向得到有限的、量级合理的大方差。
    """
    hess = 0.5 * (hess + hess.T)
    w, v = np.linalg.eigh(hess)
    wmax = float(np.max(np.abs(w))) if w.size else 0.0
    if not np.isfinite(wmax) or wmax <= 0.0:
        raise ValueError("负对数似然的 Hessian 非正定，无法计算后验协方差")
    floor = wmax * 1.0e-12
    w = np.where(w < floor, floor, w)
    cov = (v / w) @ v.T
    return 0.5 * (cov + cov.T)


def _tangent_real(u: np.ndarray) -> np.ndarray:
    """与实向量 ``u`` 正交的一组归一化实基，形状 ``(n, n-1)``。

    用 Householder 变换构造，数值稳定且不需要随机初值。
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    u = u / max(float(np.linalg.norm(u)), 1.0e-300)
    n = u.size
    if n < 2:
        return np.zeros((n, 0))
    e1 = np.zeros(n)
    e1[0] = 1.0
    v = u - e1 if u[0] >= 0.0 else u + e1
    nv = float(np.linalg.norm(v))
    if nv < 1.0e-12:
        v = np.zeros(n)
        v[1] = 1.0
        nv = 1.0
    v = v / nv
    house = np.eye(n) - 2.0 * np.outer(v, v)
    basis = house[:, 1:]
    q, _ = np.linalg.qr(basis - np.outer(u, u @ basis))
    return q[:, : n - 1]


def _tangent_complex(a: np.ndarray) -> np.ndarray:
    """与复向量 ``a`` 正交（按实内积）的一组归一化复基，形状 ``(m, 2m-2)``。

    复向量空间 ``C^m`` 按实内积视为 ``2m`` 维实空间：把 ``a`` 展开为
    ``[Re a; Im a]`` 后取其实切空间的正交基，再合并回复数。

    单位范数约束把实维数降为 ``2m-1``；此外整体复相位 ``a -> a e^{iθ}`` 不改变
    似然（相位差可由模态力谱矩阵 ``S`` 吸收），对应的方向 ``i a`` 是零曲率方向，
    必须从切空间中剔除，否则其方差无界并污染模态形状的变异系数。
    """
    a = np.asarray(a, dtype=complex).reshape(-1)
    a = a / max(float(np.linalg.norm(a)), 1.0e-300)
    m = a.size
    if m < 1:
        return np.zeros((m, 0))
    stacked = np.concatenate([np.real(a), np.imag(a)])
    zr = _tangent_real(stacked)
    # 相位方向：i a，按实内积与 a 正交，属于单位球切空间但不可辨识
    phase = np.concatenate([-np.imag(a), np.real(a)])
    nph = float(np.linalg.norm(phase))
    if nph > 1.0e-12 and zr.shape[1] > 0:
        phase = phase / nph
        reduced = zr - np.outer(phase, phase @ zr)
        q, _ = np.linalg.qr(reduced)
        keep = min(zr.shape[1], 2 * m - 2)
        zr = q[:, :keep]
    if zr.shape[1] == 0:
        return np.zeros((m, 0), dtype=complex)
    return zr[:m, :] + 1j * zr[m:, :]


def tangent_basis(u: np.ndarray) -> np.ndarray:
    """模态形状切空间的正交基，自动区分实数与复数。

    Parameters
    ----------
    u : ndarray
        单位范数的模态形状向量，可为实或复。

    Returns
    -------
    ndarray
        实向量返回 ``(n, n-1)``，复向量返回 ``(n, 2n-2)``（已剔除整体相位方向）。
    """
    u_arr = np.asarray(u)
    if np.iscomplexobj(u_arr):
        return _tangent_complex(u_arr)
    return _tangent_real(u_arr)


def _channel_map(base: np.ndarray, channel_basis) -> np.ndarray:
    """切空间坐标到通道空间实向量的线性映射矩阵。

    Parameters
    ----------
    base : ndarray
        切空间正交基。实向量为 ``(n, n-1)``，复向量为 ``(m0, 2*m0-2)``。
    channel_basis : ndarray or None
        ``None`` 表示形状向量本身就是通道空间向量；否则为把形状坐标映射到通道
        空间的正交基 ``B0``（形状 ``(n, m0)``）。

    Returns
    -------
    ndarray
        形状 ``(n_out, base.shape[1])``；复情形下 ``n_out = 2n``，实部在前、
        虚部在后。
    """
    if channel_basis is None:
        return np.asarray(base, dtype=float)
    b = np.asarray(channel_basis, dtype=float)
    n, m0 = b.shape
    if np.iscomplexobj(base):
        # 复基：先把实坐标展开为 [Re dalpha; Im dalpha]，再经 B0 映射到通道空间
        stacked = np.vstack([np.real(base), np.imag(base)])
        blk = np.zeros((2 * n, 2 * m0))
        blk[:n, :m0] = b
        blk[n:, m0:] = b
        return blk @ stacked
    return b @ np.asarray(base, dtype=float)


def pack_spectrum(f, z, S, Se: float) -> tuple[np.ndarray, int, int]:
    """把谱参数打包为实数向量 ``[f, z, diag(S), Re(offdiag S), Im(offdiag S), Se]``。

    Returns
    -------
    vec : ndarray
        实数向量。
    n_off : int
        非对角元个数 ``m(m-1)/2``。
    m : int
        模态阶数。
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    s_mat = np.asarray(S, dtype=complex)
    m = f.size
    iu = np.triu_indices(m, 1)
    vec = np.concatenate(
        [f, z, np.real(np.diag(s_mat)), np.real(s_mat[iu]), np.imag(s_mat[iu]), [float(Se)]]
    )
    return vec, iu[0].size, m


def unpack_spectrum(vec: np.ndarray, m: int, n_off: int):
    """谱参数向量还原为 ``f, z, S, Se``。"""
    f = np.asarray(vec[:m], dtype=float)
    z = np.asarray(vec[m : 2 * m], dtype=float)
    diag = np.asarray(vec[2 * m : 3 * m], dtype=float)
    re_off = np.asarray(vec[3 * m : 3 * m + n_off], dtype=float)
    im_off = np.asarray(vec[3 * m + n_off : 3 * m + 2 * n_off], dtype=float)
    se = float(vec[3 * m + 2 * n_off])
    s_mat = np.diag(diag.astype(complex))
    if n_off:
        iu = np.triu_indices(m, 1)
        s_mat[iu] = re_off + 1j * im_off
        s_mat[(iu[1], iu[0])] = re_off - 1j * im_off
    return f, z, s_mat, se


def _covariance_core(
    spec_vec: np.ndarray,
    shapes: list,
    evaluate,
    *,
    channel_basis: list,
    rel_step: float,
) -> PosteriorResult:
    """切空间参数化下的后验协方差核心计算。

    Parameters
    ----------
    spec_vec : ndarray
        谱参数的极大似然值，排列见 :func:`pack_spectrum`。
    shapes : list of ndarray
        各阶模态形状的单位范数向量，可为实或复。
    evaluate : callable
        入参为 ``(spec_vec, shapes)``、返回负对数似然值的函数。
    channel_basis : list
        各阶形状到通道空间的映射基；``None`` 表示形状向量本身即通道空间向量。
    rel_step : float
        无量纲坐标中的数值微分步长。
    """
    n_spec = int(spec_vec.size)
    scale_spec = np.maximum(np.abs(spec_vec), 1.0e-300)
    bases = [tangent_basis(s) for s in shapes]
    sizes = [int(b.shape[1]) for b in bases]
    n_free = n_spec + int(np.sum(sizes))

    def expand(y: np.ndarray) -> tuple[np.ndarray, list]:
        spec = y[:n_spec] * scale_spec
        out = []
        pos = n_spec
        for i, base in enumerate(bases):
            theta = y[pos : pos + sizes[i]]
            pos += sizes[i]
            vec = shapes[i] + base @ theta
            vec = vec / max(float(np.linalg.norm(vec)), 1.0e-300)
            out.append(vec)
        return spec, out

    def nllf_free(y: np.ndarray) -> float:
        spec, vecs = expand(y)
        return float(evaluate(spec, vecs))

    y0 = np.concatenate([spec_vec / scale_spec, np.zeros(n_free - n_spec)])
    steps = np.full(n_free, float(rel_step))
    hess = _hessian(nllf_free, y0, steps)
    cov = _regularised_inverse(hess)

    diag = np.maximum(np.diag(cov), 0.0)
    coefv = np.sqrt(diag)

    cov_phi = []
    for i, base in enumerate(bases):
        pos = n_spec + int(np.sum(sizes[:i]))
        block = cov[pos : pos + sizes[i], pos : pos + sizes[i]]
        mat = _channel_map(base, channel_basis[i])
        cov_phi.append(mat @ block @ mat.T)

    named = {"phi": float(np.sqrt(max(float(np.sum(diag[n_spec:])), 0.0)))}
    eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
    wmax = float(np.max(np.abs(eig))) if eig.size else 0.0
    diagnostics = {
        "n_free": n_free,
        "hessian_eigenvalues": eig,
        "hessian_condition": float(np.max(np.abs(eig)) / max(float(np.min(np.abs(eig))), 1.0e-300)),
        "n_negative": int(np.sum(eig < -1.0e-8 * max(wmax, 1.0))),
        "min_over_max": float(np.min(eig) / wmax) if wmax > 0.0 else 0.0,
    }
    return PosteriorResult(
        cov=cov, coefv=coefv, coefv_named=named, cov_phi=cov_phi, diagnostics=diagnostics
    )


def _name_spectrum(named: dict, coefv: np.ndarray, m: int, n_off: int) -> None:
    """把谱参数的变异系数写入 ``named``。"""
    named["f"] = coefv[:m]
    named["z"] = coefv[m : 2 * m]
    named["S"] = coefv[2 * m : 3 * m]
    named["Se"] = float(coefv[3 * m + 2 * n_off])


def covariance_from_nllf(
    nllf,
    f,
    z,
    S,
    Se: float,
    phi: np.ndarray,
    *,
    rel_step: float = 1.0e-4,
) -> PosteriorResult:
    """由返回负对数似然值的函数计算后验协方差。

    Parameters
    ----------
    nllf : callable
        入参为全参数向量、返回负对数似然值的函数。全参数向量的排列为
        ``[f (m), z (m), diag(S) (m), Re(offdiag S) (m(m-1)/2),
        Im(offdiag S) (m(m-1)/2), Se (1), phi (n*m，按列展开，可为复)``。模态形状
        在入参中不要求单位化，本函数按单位范数规范化后代入。
    f, z : array_like, shape (m,)
        极大似然估计的固有频率（Hz）与阻尼比。
    S : ndarray, shape (m, m), complex
        模态力谱密度矩阵。
    Se : float
        预测误差谱密度。
    phi : ndarray, shape (n, m)
        模态形状矩阵。
    rel_step : float, optional
        无量纲坐标中的数值微分步长，默认 ``1e-4``。

    Returns
    -------
    PosteriorResult
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    phi = np.asarray(phi)
    if phi.ndim != 2:
        raise ValueError("phi 必须是二维数组")
    m = f.size
    n = phi.shape[0]
    if phi.shape[1] != m:
        raise ValueError("phi 的列数必须等于模态阶数")

    spec_vec, n_off, _ = pack_spectrum(f, z, S, Se)
    phi0 = phi / np.maximum(np.linalg.norm(phi, axis=0, keepdims=True), 1.0e-300)
    shapes = [phi0[:, i] for i in range(m)]

    def evaluate(spec: np.ndarray, vecs: list) -> float:
        full = np.concatenate([spec, np.column_stack(vecs).reshape(-1, order="F")])
        return float(nllf(full))

    out = _covariance_core(
        spec_vec,
        shapes,
        evaluate,
        channel_basis=[None] * m,
        rel_step=rel_step,
    )
    _name_spectrum(out.coefv_named, out.coefv, m, n_off)
    return out


def multi_mode_covariance(
    result,
    freqs: np.ndarray,
    fft: np.ndarray,
    q: int = 0,
    *,
    rel_step: float = 1.0e-4,
) -> PosteriorResult:
    """多模态问题的后验协方差（式 13.61–13.67）。

    参数集取 ``{f_i}, {zeta_i}, {S_ii}, {Re S_ij}, {Im S_ij}, S_e, alpha``，其中
    ``alpha`` 为模态形状在子空间基 ``B0`` 下的复坐标，``U = B0 alpha``。由于
    ``B0`` 的列正交归一，``U`` 的单位范数约束等价于 ``alpha`` 各列的欧氏范数为 1，
    故可直接在 ``alpha`` 空间做切空间参数化。

    模态形状的变异系数按 ``C_U = B0 C_alpha B0^H`` 映射回通道空间后取对角和，
    与课本第 11.3.3 节对 ``epsilon_u`` 的定义一致；直接在 ``alpha`` 空间取迹会
    低估通道数大于模态数时的形状不确定性。

    Parameters
    ----------
    result : MultiModeResult
        ``multi_mode.mpv`` 的返回结果。
    freqs : ndarray, shape (Nf,)
        频带内的离散频率。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT 系数。
    q : int, optional
        派生数据阶次，默认 0（加速度）。
    rel_step : float, optional
        无量纲坐标中的数值微分步长，默认 ``1e-4``。

    Returns
    -------
    PosteriorResult
    """
    from .multi_mode import _project_fft, nllf_multi

    basis = np.asarray(result.basis, dtype=float)
    fproj = _project_fft(np.asarray(fft, dtype=complex), basis)
    d_total = float(np.sum(np.abs(fft) ** 2))
    nch = int(np.asarray(fft).shape[0])
    m = int(np.asarray(result.f).size)

    spec_vec, n_off, _ = pack_spectrum(result.f, result.z, result.S, float(result.Se))
    alpha0 = np.asarray(result.alpha, dtype=complex)
    alpha0 = alpha0 / np.maximum(np.linalg.norm(alpha0, axis=0, keepdims=True), 1.0e-300)
    shapes = [alpha0[:, i] for i in range(m)]

    def evaluate(spec: np.ndarray, vecs: list) -> float:
        f, z, s_mat, se = unpack_spectrum(spec, m, n_off)
        alpha = np.column_stack(vecs)
        return nllf_multi(f, z, s_mat, se, alpha, freqs, fproj, d_total, q, nch)

    out = _covariance_core(
        spec_vec,
        shapes,
        evaluate,
        channel_basis=[basis] * m,
        rel_step=rel_step,
    )
    _name_spectrum(out.coefv_named, out.coefv, m, n_off)
    # 形状的变异系数按通道空间协方差的迹计算（课本 11.3.3 的定义）。
    # 复形状展开为 [Re; Im] 后，实协方差的迹等于 E[||du||^2]。
    trace_u = sum(float(np.trace(np.asarray(mat))) for mat in out.cov_phi)
    out.coefv_named["phi"] = float(np.sqrt(max(trace_u, 0.0)))
    return out


def multi_setup_covariance(
    result,
    setups,
    q: int = 0,
    *,
    rel_step: float = 1.0e-4,
) -> PosteriorResult:
    """多测组问题的后验协方差（式 14.51–14.57）。

    参数集取 ``{f_r}, {zeta_r}, {S0_r}, {Se_r}`` 与全局模态形状 ``u``。全局模态
    形状经切空间参数化满足单位范数约束，各测组的局部模态形状由
    ``t_r = L_r u / ||L_r u||`` 确定，其单位范数约束随之自动满足。负对数似然按
    式 (14.22) 对各测组求和。

    Parameters
    ----------
    result : MultiSetupResult
        ``multi_setup.mpv`` 的返回结果。
    setups : list of SetupData
        各测组的频谱数据。
    q : int, optional
        派生数据阶次，默认 0（加速度）。
    rel_step : float, optional
        无量纲坐标中的数值微分步长，默认 ``1e-4``。

    Returns
    -------
    PosteriorResult
        参数顺序为 ``[f_1..f_ns, z_1..z_ns, S0_1..S0_ns, Se_1..Se_ns, theta]``。
    """
    from .multi_setup import selection_matrix, setup_nllf

    ns = len(setups)
    u0 = np.asarray(result.phi, dtype=float).reshape(-1)
    u0 = u0 / max(float(np.linalg.norm(u0)), 1.0e-300)
    n_dof = int(u0.size)
    mats = [selection_matrix(np.asarray(s.dofs, dtype=int), n_dof) for s in setups]

    f0 = np.asarray(result.f, dtype=float).reshape(-1)
    z0 = np.asarray(result.z, dtype=float).reshape(-1)
    s0 = np.asarray(result.S, dtype=float).reshape(-1)
    se0 = np.asarray(result.Se, dtype=float).reshape(-1)

    # S0 = S * ||L_r u||^2，按式 (14.27) 换算
    s00 = np.empty(ns)
    for i in range(ns):
        t = mats[i] @ u0
        s00[i] = s0[i] * float(t @ t)

    spec_vec = np.concatenate([f0, z0, s00, se0])
    n_spec = int(spec_vec.size)
    base = tangent_basis(u0)
    n_tan = base.shape[1]
    scale_spec = np.maximum(np.abs(spec_vec), 1.0e-300)

    def nllf_free(y: np.ndarray) -> float:
        spec = y[:n_spec] * scale_spec
        theta = y[n_spec:]
        u = u0 + base @ theta
        u = u / max(float(np.linalg.norm(u)), 1.0e-300)
        f = spec[:ns]
        z = spec[ns : 2 * ns]
        s0v = spec[2 * ns : 3 * ns]
        sev = spec[3 * ns : 4 * ns]
        total = 0.0
        for i in range(ns):
            t = mats[i] @ u
            tn = max(float(np.linalg.norm(t)), 1.0e-300)
            total += setup_nllf(
                float(f[i]),
                float(z[i]),
                float(s0v[i]),
                float(sev[i]),
                setups[i].freqs,
                setups[i].dhat,
                setups[i].d_total,
                t / tn,
                q,
            )
        return float(total)

    y0 = np.concatenate([spec_vec / scale_spec, np.zeros(n_tan)])
    steps = np.full(y0.size, float(rel_step))
    hess = _hessian(nllf_free, y0, steps)
    cov = _regularised_inverse(hess)

    diag = np.maximum(np.diag(cov), 0.0)
    coefv = np.sqrt(diag)
    named = {
        "f": coefv[:ns],
        "z": coefv[ns : 2 * ns],
        "S": coefv[2 * ns : 3 * ns],
        "Se": coefv[3 * ns : 4 * ns],
        "phi": float(np.sqrt(max(float(np.sum(diag[n_spec:])), 0.0))),
    }
    block = cov[n_spec:, n_spec:]
    cov_phi = [base @ block @ base.T]
    eig = np.linalg.eigvalsh(0.5 * (hess + hess.T))
    wmax = float(np.max(np.abs(eig))) if eig.size else 0.0
    diagnostics = {
        "n_free": int(y0.size),
        "hessian_eigenvalues": eig,
        "hessian_condition": float(
            np.max(np.abs(eig)) / max(float(np.min(np.abs(eig))), 1.0e-300)
        ),
        "n_negative": int(np.sum(eig < -1.0e-8 * max(wmax, 1.0))),
        "min_over_max": float(np.min(eig) / wmax) if wmax > 0.0 else 0.0,
    }
    return PosteriorResult(
        cov=cov,
        coefv=coefv,
        coefv_named=named,
        cov_phi=cov_phi,
        diagnostics=diagnostics,
    )
