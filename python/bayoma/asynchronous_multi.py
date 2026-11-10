"""异步数据的多阶模态贝叶斯识别（课本第 10.4 节，式 10.43–10.49）。

同一测组内各通道共用时钟、不同测组之间不共用时钟时，组间互谱保留未知的相位，
第 10.1 节的同步模型不再适用。第 10.4 节把每一阶模态形状按测组切块

    u_i = [t_1i; ...; t_qi],   t_ri 为第 r 组 n_r 个测点的局部振型        (10.43)

并让同一阶模态在组 r 的模态响应跟随组 r 的时钟（式 10.44）；组间的不完全相干由
复相干因子表征

    E[eta_ik^(r) eta_jk^(s)* | theta] = h_ik h_jk* S_ij v_ijrs,           (10.45)

其中 ``|v_ijrs| <= 1``、``v_ijrr = 1``。把各阶模态叠加后谱矩阵为

    E_k = sum_i sum_j h_ik h_jk* S_ij (U_i V_ij U_j^T) + D0,             (10.47)
    D0  = diag{S_er I_{n_r}},  r = 1..q,

``U_i = block_operator(u_i, groups)`` 为式 (10.46) 的 ``n x q`` 分块振型算子，
第 r 列的第 r 组行块等于 ``t_ri``、其余为零。``V_ii = I_q``；``V_ij``（i < j）为
``q x q`` 的 Hermitian 相干矩阵，对角元为 1（式 10.51 的多阶推广）。

紧凑形式（式 10.48–10.49）::

    G = [U_1, ..., U_m]                       n x mq
    Sv[iq:(i+1)q, jq:(j+1)q] = S_ij V_ij      mq x mq，Hermitian
    W_k = diag(h_k) (x) I_q                   mq x mq
    E_k = G W_k Sv W_k^* G^T + D0

参数集为 ``{f_i, zeta_i}``、模态力谱密度矩阵 ``S``、``{S_er}``、单位范数的
模态形状 ``{u_i}``（``u_i^H u_i = 1``，跨组拼接、共用自由度上取值相同）与各阶
对的相干矩阵 ``{V_ij}``。``S`` 的取值形式按测组数分两支（第 2 条实现说明）：
单测组时取 Hermitian 复矩阵（对角元实正、非对角元实部与虚部均自由），多测组
时取实对称矩阵（非对角元允许负实数）。模态形状的取值形式同样按测组数分两支
（第 5 条实现说明）：单测组且 ``m > 1`` 时取复向量，多测组或 ``m = 1`` 时取
实向量。

实现说明
--------
1. **降维恒等式**。取 ``G`` 列空间的正交基 ``B``（``n x m0``，``m0 = rank(G)``），
   则 ``E_k = B E0_k B^T + D0``，``E0_k = (B^T G) W_k Sv W_k^* (G^T B)`` 为
   ``m0 x m0``。记 ``B~ = D0^{-1/2} B``、``C = B~^T B~``，由 Woodbury 恒等式
   ``(I + B~ E0 B~^T)^{-1} = I - B~ (E0^{-1} + C)^{-1} B~^T`` 与
   ``det(I + B~ E0 B~^T) = det(I + E0 C) = det(E0) det(E0^{-1} + C)``，两次求和
   全部化为 ``m0 x m0`` 上的行列式与二次型，单次似然评估为
   ``O(N_f m0^2 m q)``，与逐频率的 ``n x n`` 分解结果一致。
2. **模态力谱密度矩阵的规范按测组数分两支**。``S_ij`` 的整体相位与 ``V_ij`` 的
   整体相位是同一规范自由度（``S_ij -> S_ij e^{i alpha}``、``V_ij -> V_ij e^{-i
   alpha}`` 时 ``E_k`` 不变）。多测组（``q >= 2``）时 ``V_ij`` 的非对角元承担该
   相位，把 ``S`` 取实即固定规范；若允许 ``S_ij`` 复值，Hessian 在该方向上的
   曲率为零，后验方差发散。单测组（``q = 1``）时 ``V`` 不存在（``npair = 0``，
   :func:`_coherence_pairs` 返回空表），``S_ij`` 的复相位是可辨识参数，取
   Hermitian 复矩阵（对角元实正，非对角元的实部与虚部均自由），式 (10.45) 的
   复模态力相干因而可以表达；此时无相位规范退化。参数向量中非对角元的长度
   相应由 ``ntri`` 变作 ``2 ntri``（实部、虚部各一段），由 ``complex_s`` 标志控制。
3. **符号规范**。``m = 1`` 时 ``t_r -> -t_r`` 同时把 ``v`` 的第 r 行与第 r 列反号
   （第 10.4 节）；``m >= 2`` 时 ``V_ij`` 的对角元固定为 1，逐组反号不再可行，
   剩下的是整阶反号 ``u_i -> -u_i``，对应 ``S_ij -> -S_ij``（j 不等于 i），
   式 (10.47) 的谱矩阵不变。
4. 谱参数在自然对数空间优化，上下界按数据量纲取（``d_scale = mean|F_k|^2``，
   界 ``d_scale * 1e+-8``）。改用与数据无关的 ``1e-30~1e30`` 会使有界拟牛顿法的
   线搜索在初值处即报告相对下降收敛。
5. **模态形状的参数化（单测组）**。``q = 1`` 时组间不存在时钟相位，``V``
   退化，式 (10.47) 成为 ``E_k = U diag(h_k) S diag(h_k)^H U^H + S_e I_n``，与
   第 13 章式 (13.10) 同构，两个模型是同一问题。同步多阶模型把模态形状写成

       phi = B_0 alpha，B_0 为 n x m 的实正交基，alpha 为 m x m 复系数，  (13.6)

   ``B_0`` 取 ``A_0 = sum_k D_k`` 的前 ``m_0`` 个最大特征向量（第 13.6.1 节），
   ``alpha`` 的实部与虚部均自由，初值取带内平均的模态形状初值在 ``B_0`` 上的
   坐标。``(B_0 alpha) diag(h_k)`` 与
   ``B_0 diag(h_k) alpha`` 一般不相等，模态形状若限制为实向量，模型类会严格
   小于同步多阶模型，频率与阻尼比的最可能值随之系统偏离。故 ``q = 1`` 且
   ``m > 1`` 时沿用式 (13.6) 的参数化：形状由 ``alpha`` 的 ``2 m_0 m`` 个实坐标
   给出，逐列归一化后每阶的实维数为 ``2 m_0 - 2``（``m_0 = min(m, n)``），
   合计 ``2 m (m_0 - 1)``；子空间基 ``B_0`` 在极小化过程中保持不变，相邻两阶
   共用同一子空间基，密集模态的形状差异由 ``alpha`` 的列间差异表达。第 13 章
   另按式 (13.22)–(13.23) 用超角转动子空间，本实现的形状类是它的子集；在实测
   算例上两者的最优负对数似然相差 ``5e-7``，频率、阻尼比与振型的识别差异都在
   ``1e-4`` 以内。两者在同一参数点（形状落在 ``span(B_0)`` 内时）的似然取值
   一致到 ``2e-7``；同步解的振型含 ``1e-4`` 量级的子空间外分量，把该解直接代回
   本模型时似然偏低 ``3e-3``（相对 ``1.4e-7``），这是两套实现对该分量的处理
   不同，不影响上列的识别值。
   ``q >= 2`` 时 ``S_ij`` 的相位与 ``V_ij`` 的整体相位是同一规范
   自由度，形状必须取实向量（``n - 1`` 个实坐标）；``m = 1`` 时 ``S_11`` 为
   实正标量、相位无处吸收，形状同样取实向量。收敛后由 :func:`_fix_signs` 把
   每阶形状绝对值最大的分量规范为正实数，并把相位差并回 ``S``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ._util import LOG
from .asynchronous import _VMAX, _validate_groups, noise_diagonal
from .multi_mode import (
    _initial_basis,
    _initial_frequencies,
    transfer_matrix,
)

_LOG_PI = float(np.log(np.pi))

# 阻尼比与谱密度的允许范围，与第 12 章、第 10.4 节的既有实现保持一致。
_ZMIN = 1.0e-6
_ZMAX = 0.5
_SMIN = 1.0e-30
_SMAX = 1.0e30

# 模态形状切空间坐标的允许范围。切空间参数化 ``u = (u0 + Z theta)/||...||`` 中
# ``theta`` 没有上界约束（归一化自动保证单位范数），但过大的搜索区间会使有界
# 拟牛顿法在初期沿坐标轴方向试探时步长失配。十是足够宽的界：``|theta| = 10``
# 对应偏离初值约 84 度。
_TAN_MAX = 10.0


@dataclass
class AsynchronousMultiSetup:
    """一个频带内的异步多阶数据集合。

    Attributes
    ----------
    freqs : ndarray, shape (Nf,)
        频带内各 FFT 点的频率（Hz）。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT（式 10.1），按列存放各频率点的 ``F_k``。异步数据保留复值本身的
        组间相位信息，不做共轭相乘或取实部。
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
        if self.fft.ndim != 2:
            raise ValueError("fft 必须是二维数组，形状 (n, Nf)")
        self.groups = _validate_groups(self.groups, int(self.fft.shape[0]))
        if self.fft.shape[1] != self.freqs.size:
            raise ValueError("fft 的列数必须等于频率点数")

    @classmethod
    def from_spectral_data(cls, data, groups) -> "AsynchronousMultiSetup":
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
class AsynchronousMultiResult:
    """异步多阶模态识别的输出。

    Attributes
    ----------
    f : ndarray, shape (m,)
        各阶固有频率（Hz），按升序排列。
    z : ndarray, shape (m,)
        各阶阻尼比。
    S : ndarray, shape (m, m), complex
        模态力谱密度矩阵。单测组时为 Hermitian（对角元实数正、非对角元复值），
        多测组时为实对称（非对角元允许负实数），与 ``phi`` 的符号约定配套。
    Se : ndarray, shape (q,)
        各测组的预测误差（通道噪声）谱密度 ``S_er``。
    phi : ndarray, shape (n, m)
        模态形状矩阵，第 i 列即 ``u_i``，欧氏范数为 1。单测组且多阶时为复向量，
        其余情形为实向量（模块说明第 5 条）。
    groups : list of list of int
        测组划分。
    coherence : dict
        各阶对的相干矩阵，键 ``(i, j)``（``i < j``，按 ``phi`` 列的次序）对应
        ``q x q`` 复矩阵，Hermitian 且对角元为 1。
    nllf : float
        极小化后的负对数似然函数值，式 (10.42)。
    n_iter : int
        交替极小化的外层迭代次数。
    converged : bool
        是否满足收敛判据。
    basis : ndarray or None
        仅单测组且多阶时给出，为式 (13.6) 的实正交基 ``B_0``，形状 ``(n, m_0)``，
        ``m_0 = min(m, n)``；此时 ``phi = basis @ alpha``。其余情形为 ``None``。
    alpha : ndarray or None
        仅单测组且多阶时给出，为式 (13.6) 的复系数矩阵，形状 ``(m_0, m)``；各列
        到通道空间按 ``basis`` 映射，列范数归一时 ``phi`` 的列范数为 1。
    """

    f: np.ndarray
    z: np.ndarray
    S: np.ndarray
    Se: np.ndarray
    phi: np.ndarray
    groups: list
    coherence: dict
    nllf: float
    n_iter: int
    converged: bool
    basis: np.ndarray | None = None
    alpha: np.ndarray | None = None

    @property
    def local(self) -> list:
        """各测组的局部振型矩阵，第 r 个元素形状为 ``(n_r, m)``。"""
        phi = np.asarray(self.phi)
        if not np.iscomplexobj(phi):
            phi = np.asarray(phi, dtype=float)
        return [phi[np.asarray(g, dtype=int), :] for g in self.groups]


def _as_square(mat, m: int) -> np.ndarray:
    """把模态力谱密度矩阵统一为 ``m x m`` 复数组。"""
    arr = np.asarray(mat, dtype=complex)
    if arr.ndim == 0:
        if m != 1:
            raise ValueError(f"模态力谱密度矩阵应为 {m} x {m}，收到标量")
        arr = arr.reshape(1, 1)
    if arr.shape != (m, m):
        raise ValueError(f"模态力谱密度矩阵应为 {m} x {m}，收到 {arr.shape}")
    return arr


def _as_shapes(phi, n: int) -> np.ndarray:
    """把模态形状统一为 ``n x m`` 数组，保留复值。

    单测组（``q = 1``）且 ``m > 1`` 时模态形状取复向量（模块说明第 5 条），
    其余情形为实向量；复值是否合法由调用方按测组数与模态数判断。
    """
    arr = np.asarray(phi)
    if np.iscomplexobj(arr):
        arr = np.asarray(arr, dtype=complex)
    else:
        arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError(f"模态形状的行数应为 {n}，收到 {np.shape(phi)}")
    return arr


def _complex_shapes(ng: int, m: int) -> bool:
    """模态形状是否取复向量，即单测组且多阶的情形（模块说明第 5 条）。

    ``q = 1`` 时组间无时钟相位，式 (10.47) 与第 13 章式 (13.10) 同构，模态形状
    可以是复向量；``m = 1`` 时 ``S_11`` 为实正标量、形状相位无处吸收，形状必须
    取实，故该分支只在 ``m > 1`` 时启用。``q >= 2`` 时 ``S_ij`` 的相位与 ``V_ij``
    的整体相位构成同一规范自由度，形状必须取实。
    """
    return int(ng) == 1 and int(m) > 1


def _subspace_coordinates(phi, basis: np.ndarray) -> np.ndarray:
    """把通道空间的模态形状投影到子空间基上，得到式 (13.6) 的坐标 ``alpha``。

    ``alpha_i = B_0^T phi[:, i]`` 后逐列归一化为单位向量，于是 ``B_0 alpha`` 与
    ``phi`` 同向（``B_0`` 列正交归一，投影不改变各列的相对幅值），切空间坐标
    的原点严格对应待投影的形状。第 13.6.1 节的单频点初值取初始频率处样本谱
    矩阵的主特征向量，在共振峰顶部信噪比最高，但第 2 阶及以后各阶的初始频率
    与真实频率可能相差一个频率分辨率，该频点的主特征向量已明显偏离该阶形状；
    带内平均得到的 ``phi0``（:func:`_initial_guess`）对频率初值的偏差不敏感，
    实测与真值的 MAC 在 0.9999 以上，故以它为复坐标的起点。
    """
    proj = np.asarray(basis, dtype=float).T @ np.asarray(phi, dtype=complex)
    nrm = np.linalg.norm(proj, axis=0)
    nrm = np.where(nrm > 0.0, nrm, 1.0)
    return np.asarray(proj / nrm[None, :], dtype=complex)


def _transfer(f, z, freqs: np.ndarray, q: int) -> np.ndarray:
    """各阶模态的传递函数 ``h``，形状 ``(m, Nf)``，式 (10.8)。

    ``h_ik = (2 pi f_k)^(-q) / (1 - beta_ik^2 - 2 i zeta_i beta_ik)``，
    ``beta_ik = f_i / f_k``。与 :func:`bayoma.multi_mode.transfer_matrix` 同一
    实现：``q`` 取 0/1/2 表示派生数据为加速度/速度/位移。
    """
    return transfer_matrix(
        np.asarray(f, dtype=float).reshape(-1),
        np.asarray(z, dtype=float).reshape(-1),
        np.asarray(freqs, dtype=float),
        int(q),
    )


def _pair_matrix(vij, i: int, j: int, ng: int, m: int) -> np.ndarray:
    """取第 ``(i, j)`` 阶对的相干矩阵 ``V_ij``。

    ``i == j`` 时按式 (10.47) 取 ``V_ii = I_q``；唯一例外是 ``m = 1``，此时
    ``vij[(0, 0)]`` 即式 (10.51) 的单阶相干矩阵，直接采用，以便与
    :mod:`bayoma.asynchronous` 的 ``E_k = S D_k u~ v u~^T + D0`` 逐项对照。
    """
    if i == j:
        if m == 1 and (0, 0) in vij:
            return np.asarray(vij[(0, 0)], dtype=complex)
        return np.eye(ng, dtype=complex)
    key = (min(i, j), max(i, j))
    if key not in vij:
        if ng == 1:
            # 单测组时相干矩阵退化为 1 x 1 的 [1]，无需显式给出
            return np.eye(1, dtype=complex)
        raise ValueError(f"缺少模态阶对 {key} 的相干矩阵")
    arr = np.asarray(vij[key], dtype=complex)
    if arr.shape != (ng, ng):
        raise ValueError(f"阶对 {key} 的相干矩阵应为 {ng} x {ng}，收到 {arr.shape}")
    return arr if i < j else arr.conj().T


def _sv_matrix(S, vij, m: int, ng: int) -> np.ndarray:
    """式 (10.49) 的分块矩阵 ``Sv``，形状 ``(m q, m q)``，``(i, j)`` 块为
    ``S_ij V_ij``。"""
    smat = _as_square(S, m)
    sv = np.zeros((m * ng, m * ng), dtype=complex)
    for i in range(m):
        for j in range(m):
            sv[i * ng : (i + 1) * ng, j * ng : (j + 1) * ng] = smat[i, j] * _pair_matrix(
                vij, i, j, ng, m
            )
    return 0.5 * (sv + sv.conj().T)


def _block_operator(u: np.ndarray, groups) -> np.ndarray:
    """式 (10.46) 的分块振型算子 ``U``，形状 ``(n, q)``，保留复值。

    第 r 列的第 r 组行块等于局部振型 ``t_r``，其余为零。与
    :func:`bayoma.asynchronous.block_operator` 相同，只是不把输入强制转为实
    数组：单测组时模态形状取复向量（模块说明第 5 条）。
    """
    u = np.asarray(u)
    if not np.iscomplexobj(u):
        u = np.asarray(u, dtype=float)
    u = u.reshape(-1)
    grp = _validate_groups(groups, u.size)
    uop = np.zeros((u.size, len(grp)), dtype=u.dtype)
    for r, idx in enumerate(grp):
        uop[idx, r] = u[idx]
    return uop


def _shape_matrix(phi, groups) -> np.ndarray:
    """式 (10.46) 的按组分块振型算子拼接 ``G = [U_1, ..., U_m]``，``n x m q``。"""
    phi = np.asarray(phi)
    if phi.ndim != 2:
        raise ValueError("模态形状必须是二维数组，形状 (n, m)")
    m = phi.shape[1]
    return np.hstack([_block_operator(phi[:, i], groups) for i in range(m)])


def _column_basis(G: np.ndarray, rtol: float = 1.0e-10) -> np.ndarray:
    """``G`` 列空间的一组正交归一基 ``B``，维数 ``m0 = rank(G)``。

    奇异值按 ``> rtol * s_max`` 判定，``m0 <= min(n, m q)``。``G`` 为复矩阵时
    基按复内积正交归一（``B^H B = I``），式 (10.47) 的 ``E_k = B E0_k B^H + D0``
    对复基同样成立。
    """
    gmat = np.asarray(G)
    if gmat.ndim != 2:
        raise ValueError("G 必须是二维数组")
    u, s, _ = np.linalg.svd(gmat, full_matrices=False)
    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
        raise ValueError("模态形状算子为零矩阵，无法构造列空间基")
    m0 = int(np.sum(s > rtol * s[0]))
    m0 = max(m0, 1)
    return np.ascontiguousarray(u[:, :m0])


def theoretical_psd_block(
    f,
    z,
    S,
    Se_list,
    phi,
    vij,
    freqs: np.ndarray,
    groups,
    q: int = 0,
) -> np.ndarray:
    """按式 (10.47)–(10.49) 计算各频率点的理论谱矩阵 ``E_k``。

    Parameters
    ----------
    f, z : array_like, shape (m,)
        各阶固有频率（Hz）与阻尼比。
    S : ndarray, shape (m, m)
        实对称（推广情形可 Hermitian）模态力谱密度矩阵。
    Se_list : array_like, shape (q,)
        各测组的预测误差谱密度 ``S_er``。
    phi : ndarray, shape (n, m)
        模态形状矩阵，第 i 列为 ``u_i``。
    vij : dict
        各阶对的相干矩阵，键 ``(i, j)``（``i < j``）。``m = 1`` 时可给出键
        ``(0, 0)`` 表示式 (10.51) 的单阶相干矩阵；``m >= 2`` 时对角块固定取
        ``V_ii = I_q``。
    freqs : ndarray, shape (Nf,)
        频率坐标（Hz）。
    groups : list of list of int
        测组划分。
    q : int, optional
        数据类型指数，``0`` 加速度、``1`` 速度、``2`` 位移（式 10.8）。

    Returns
    -------
    ndarray, shape (n, n, Nf), complex
        Hermitian 谱矩阵序列 ``E_k = G W_k Sv W_k^* G^T + D0``。

    Notes
    -----
    ``W_k = diag(h_k) (x) I_q`` 作用在按「阶优先、组其次」排列的 ``m q`` 维向量
    上，故 ``G W_k`` 的每一列按 ``np.repeat(h[:, k], ng)`` 缩放。
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    phi_mat = np.asarray(phi)
    if phi_mat.ndim == 1:
        phi_mat = phi_mat.reshape(-1, 1)
    if not np.iscomplexobj(phi_mat):
        phi_mat = np.asarray(phi_mat, dtype=float)
    n, m = phi_mat.shape
    grp = _validate_groups(groups, n)
    ng = len(grp)
    if np.iscomplexobj(phi_mat) and not _complex_shapes(ng, m):
        raise ValueError(
            "复值模态形状只在单测组且多阶（q = 1、m > 1）时成立，"
            f"收到 q = {ng}、m = {m}"
        )
    fv = np.asarray(f, dtype=float).reshape(-1)
    zv = np.asarray(z, dtype=float).reshape(-1)
    if fv.size != m or zv.size != m:
        raise ValueError(f"频率与阻尼比应为 {m} 个，收到 {fv.size} 与 {zv.size}")
    smat = _as_square(S, m)
    se = np.asarray(Se_list, dtype=float).reshape(-1)
    d0 = noise_diagonal(se, grp)
    h = _transfer(fv, zv, freqs, q)  # (m, Nf)
    if not np.all(np.isfinite(h)):
        raise ValueError("传递函数出现非有限值，请检查频率范围")

    gmat = _shape_matrix(phi_mat, grp)  # (n, m q)
    sv = _sv_matrix(smat, vij, m, ng)
    rep = np.repeat(h, ng, axis=0).T  # (Nf, m q)
    aw = gmat[None, :, :] * rep[:, None, :]  # (Nf, n, m q)
    ek = aw @ sv @ np.conj(np.transpose(aw, (0, 2, 1)))
    ek = 0.5 * (ek + np.conj(np.transpose(ek, (0, 2, 1))))
    # 加上预测误差谱密度：D0 = diag{S_er I_{n_r}}（式 10.47 第二项）。
    # 此处按 (n, n) 广播相加，若写成 ek[arange(n), arange(n), :] += d0[:, None]
    # 得到的是花式索引的副本，加不到原数组上，对角噪声会被静默丢弃。
    ek += np.diag(d0)[None, :, :]
    return np.transpose(ek, (1, 2, 0))


def nllf(
    f,
    z,
    S,
    se_list,
    phi,
    vij,
    freqs: np.ndarray,
    fft: np.ndarray,
    groups,
    q: int = 0,
) -> float:
    """式 (10.42) 的负对数似然函数，按降维恒等式计算。

    Parameters
    ----------
    f, z : array_like, shape (m,)
        各阶固有频率（Hz）与阻尼比。
    S : ndarray, shape (m, m)
        模态力谱密度矩阵。
    se_list : array_like, shape (q,)
        各测组的预测误差谱密度。
    phi : ndarray, shape (n, m)
        模态形状矩阵。
    vij : dict
        各阶对的相干矩阵，见 :func:`theoretical_psd_block`。
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
        负对数似然函数值；参数越出可行域或谱矩阵不正定时返回 ``inf``。

    Notes
    -----
    降维恒等式的推导（两行）：

    1. 取 ``B`` 为 ``G`` 列空间的正交基、``B~ = D0^{-1/2} B``、``C = B~^T B~``，
       则 ``E_k = D0^{1/2} (I + B~ E0_k B~^T) D0^{1/2}``，由 Woodbury 恒等式
       ``(I + B~ E0 B~^T)^{-1} = I - B~ (E0^{-1} + C)^{-1} B~^T`` 得
       ``F_k^* E_k^{-1} F_k = |D0^{-1/2} F_k|^2 - (B~^T D0^{-1/2} F_k)^*
       (E0_k^{-1} + C)^{-1} (B~^T D0^{-1/2} F_k)``；
    2. 由 ``det(I + B~ E0 B~^T) = det(I + E0 C) = det(E0) det(E0^{-1} + C)`` 得
       ``ln|E_k| = sum_r n_r ln S_er + ln det E0_k + ln det(E0_k^{-1} + C)``。

    两式都只含 ``m0 x m0`` 的矩阵，故单次评估为 ``O(N_f m0^2 m q)``。
    """
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    fft = np.asarray(fft, dtype=complex)
    if fft.ndim != 2:
        raise ValueError("fft 必须是二维数组，形状 (n, Nf)")
    n, nf = fft.shape
    if freqs.size != nf:
        raise ValueError("频率点数必须等于 fft 的列数")
    phi_mat = _as_shapes(phi, n)
    m = phi_mat.shape[1]
    grp = _validate_groups(groups, n)
    ng = len(grp)
    if np.iscomplexobj(phi_mat) and not _complex_shapes(ng, m):
        raise ValueError(
            "复值模态形状只在单测组且多阶（q = 1、m > 1）时成立，"
            f"收到 q = {ng}、m = {m}"
        )
    fv = np.asarray(f, dtype=float).reshape(-1)
    zv = np.asarray(z, dtype=float).reshape(-1)
    if fv.size != m or zv.size != m:
        raise ValueError(f"频率与阻尼比应为 {m} 个，收到 {fv.size} 与 {zv.size}")
    smat = _as_square(S, m)
    se = np.asarray(se_list, dtype=float).reshape(-1)

    if not (np.all(np.isfinite(fv)) and np.all(fv > 0.0)):
        return np.inf
    if not (np.all(np.isfinite(zv)) and np.all(zv > 0.0) and np.all(zv < 1.0)):
        return np.inf
    if not np.all(np.isfinite(smat)):
        return np.inf
    if not (np.all(np.isfinite(se)) and np.all(se > 0.0)):
        return np.inf

    h = _transfer(fv, zv, freqs, q)
    if not np.all(np.isfinite(h)):
        return np.inf

    try:
        d0 = noise_diagonal(se, grp)
        gmat = _shape_matrix(phi_mat, grp)
        basis = _column_basis(gmat)
        m0 = basis.shape[1]
        # 复基按复内积取共轭转置（模块说明第 5 条）；实基时与转置相同。
        proj = basis.conj().T @ gmat  # (m0, m q)
        sv = _sv_matrix(smat, vij, m, ng)
        rep = np.repeat(h, ng, axis=0).T  # (Nf, m q)
        aw = proj[None, :, :] * rep[:, None, :]  # (Nf, m0, m q)
        e0 = aw @ sv @ np.conj(np.transpose(aw, (0, 2, 1)))
        e0 = 0.5 * (e0 + np.conj(np.transpose(e0, (0, 2, 1))))
        cho0 = np.linalg.cholesky(e0)
        ln_e0 = 2.0 * np.sum(
            np.log(np.real(np.diagonal(cho0, axis1=1, axis2=2))), axis=1
        )
        inv0 = np.linalg.inv(e0)
        # C = B~^H B~ = B^H D0^{-1} B
        cmat = basis.conj().T @ (basis / d0[:, None])
        mmat = 0.5 * (inv0 + np.conj(np.transpose(inv0, (0, 2, 1)))) + cmat[None, :, :]
        chom = np.linalg.cholesky(mmat)
        ln_m = 2.0 * np.sum(
            np.log(np.real(np.diagonal(chom, axis1=1, axis2=2))), axis=1
        )
        # x_k = B~^H D0^{-1/2} F_k = B^H D0^{-1} F_k
        xk = basis.conj().T @ (fft / d0[:, None])
        w = np.linalg.solve(chom, np.transpose(xk, (1, 0))[:, :, None])[:, :, 0]
        quad_corr = np.sum(np.abs(w) ** 2, axis=1)
    except np.linalg.LinAlgError:
        return np.inf

    if not (
        np.all(np.isfinite(ln_e0))
        and np.all(np.isfinite(ln_m))
        and np.all(np.isfinite(quad_corr))
    ):
        return np.inf

    n_grp = np.asarray([g.size for g in grp], dtype=float)
    d_grp = np.asarray([float(np.sum(np.abs(fft[g, :]) ** 2)) for g in grp])
    logdet = float(nf * np.sum(n_grp * np.log(se)))
    logdet += float(np.sum(ln_e0)) + float(np.sum(ln_m))
    quad = float(np.sum(d_grp / se)) - float(np.sum(quad_corr))
    val = n * nf * _LOG_PI + logdet + quad
    return float(val) if np.isfinite(val) else np.inf


def _normalise_columns(cols, dhat: np.ndarray, grp, m: int) -> np.ndarray:
    """把各阶初始模态形状归一化为单位范数，必要时补足线性无关的候选列。

    式 (10.47) 只要求各阶模态形状单位范数，并不要求不同阶之间正交：模态力谱
    密度矩阵 ``S`` 的非对角元一般不为零，``u_i`` 之间因而没有正交约束。故此处
    只做逐列归一化，不施加 Gram-Schmidt 正交化。若两阶初始列几乎共线（夹角
    余弦大于 0.99），后一阶改用「组内块对角样本谱矩阵之和」的高阶特征向量，
    逐个与已有列作正交化后再归一化，以保证 ``G`` 的列满秩。

    异步数据不存在跨组的样本谱矩阵，故补足向量取组内块对角矩阵之和
    ``A0 = sum_k blockdiag_r D_k[g_r, g_r]`` 的特征向量。测点全部落在同一测组时
    该矩阵即样本谱矩阵之和，与第 13 章同步多模态的子空间基选取一致。
    """
    n = np.asarray(cols[0]).size
    cand: list[np.ndarray] = []
    for col in cols:
        vec = np.asarray(col, dtype=float)
        nrm = float(np.linalg.norm(vec))
        if nrm > 0.0:
            cand.append(vec / nrm)
    pool: list[np.ndarray] = []

    def _add(vec):
        vec = np.asarray(vec, dtype=float)
        nrm = float(np.linalg.norm(vec))
        if nrm > 1.0e-12:
            pool.append(vec / nrm)

    for col in cand:
        if len(pool) >= m:
            break
        duplicate = any(abs(float(col @ prev)) > 0.99 for prev in pool)
        if duplicate:
            continue
        _add(col)

    if len(pool) < m:
        a0 = np.zeros((n, n))
        for idx in grp:
            a0[np.ix_(idx, idx)] += np.einsum("ijk->ij", dhat[np.ix_(idx, idx)])
        vals, vecs = np.linalg.eigh(0.5 * (a0 + a0.T))
        for j in np.argsort(vals)[::-1]:
            if len(pool) >= m:
                break
            vec = vecs[:, int(j)]
            # 与已选列作正交化，保证新增列与已有列线性无关
            for prev in pool:
                vec = vec - prev * float(prev @ vec)
            _add(vec)

    if len(pool) < m:
        raise ValueError("无法构造线性无关的初始模态形状，请检查频带内的模态数")
    return np.column_stack(pool[:m])


def _peak_frequencies(freqs, dhat, m, width, min_sep):
    """平滑奇异值曲线后按突出度取谱峰，作为频率初值。

    异步数据的第一条奇异值曲线在组间互谱起伏的叠加下，同一共振峰常裂成相距
    数个频率步长的双峰（两条曲线的相对符号在峰顶附近翻转）。直接取峰会把两阶
    模态判成同一阶的孪生峰。此处先以约 1% 频带宽度的滑动平均平滑曲线，再按
    突出度排序并强制最小间隔 ``min_sep``（密集模态按 ``3 df`` 拉开，与第 13.6.1
    节一致），峰数不足时由 :func:`bayoma.multi_mode._initial_frequencies` 补足。
    """
    from scipy.signal import find_peaks

    from .multi_mode import _sv_curves

    sv = _sv_curves(dhat)[0]
    ker = np.ones(int(width)) / float(width)
    smooth = np.convolve(sv, ker, mode="same")
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    dist = max(int(round(min_sep / df)), 1)
    pk, props = find_peaks(
        smooth, prominence=1.0e-6 * float(smooth.max()), distance=dist
    )
    sel: list[float] = []
    if pk.size:
        for idx in np.argsort(props["prominences"])[::-1]:
            cand = float(freqs[int(pk[idx])])
            if all(abs(cand - c) >= 0.999 * min_sep for c in sel):
                sel.append(cand)
            if len(sel) >= m:
                break
    if len(sel) < m:
        for cand in np.asarray(_initial_frequencies(freqs, dhat, m), dtype=float):
            if len(sel) >= m:
                break
            if all(abs(cand - c) >= 0.999 * min_sep for c in sel):
                sel.append(float(cand))
    while len(sel) < m:
        # 极端情形（频带内峰数少于模态数）按等间隔补齐，保证初值维数正确
        sel.append(float(freqs[min(len(sel), freqs.size - 1)]))
    return np.sort(np.asarray(sel[:m], dtype=float))


def _noise_floor(dhat, grp, width):
    """各测组预测误差谱密度 ``S_er`` 的初值（式 10.47 的 ``D0`` 块）。

    取「局部平滑后组内样本谱矩阵块的最小特征值谱」在带内的 10% 分位数。局部
    平滑（窗长约 1% 频带宽度）保持共振峰内的相干模态项，噪声项按窗长平均后
    回落到 ``S_er``；分位数取远离共振峰的静默段。频带全平均的组内秩估计
    ``(tr A_r - lambda_max)/((n_r - 1) N_f)`` 只在单模态频带成立，多模态时把
    相邻阶的模态能量计入噪声，可高估一个数量级以上。
    """
    nf = int(dhat.shape[2])
    se = np.empty(len(grp))
    for r, idx in enumerate(grp):
        if idx.size == 1:
            se[r] = max(float(np.percentile(np.real(dhat[idx[0], idx[0], :]), 10.0)), _SMIN)
            continue
        spec = np.empty(nf)
        for k in range(nf):
            lo, hi = max(k - width, 0), min(k + width + 1, nf)
            blk = np.einsum("ijk->ij", dhat[np.ix_(idx, idx, np.arange(lo, hi))])
            blk = 0.5 * (blk / float(hi - lo) + (blk / float(hi - lo)).T)
            spec[k] = float(np.linalg.eigvalsh(blk)[0])
        se[r] = max(float(np.percentile(spec, 10.0)), _SMIN)
    return se


def _band_columns(dhat, grp, freqs, f_init, half, se):
    """各阶模态形状的初值列与组内块特征值。

    组内同步使组内样本谱矩阵块在每个频率点近似秩一，故带内平均后的组内块仍以
    该阶局部振型为主特征向量。只取单个频率点会丢失组与组之间的相对幅值：各组块
    的最大特征值相差可达两个数量级，逐块归一化拼接后的振型与真值相差数十度。
    此处按 ``sqrt(lambda_max - S_er)`` 缩放各组块的主特征向量再拼接，使组间相对
    幅值由样本谱的模态能量给出（第 12.3.1 节的高信噪比渐近式）。
    """
    n = int(dhat.shape[0])
    nf = int(dhat.shape[2])
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    back = max(1, int(round(half / df)))
    m = len(f_init)
    cols = []
    lam = np.zeros((m, len(grp)))
    for i in range(m):
        kc = int(np.argmin(np.abs(freqs - f_init[i])))
        lo, hi = max(kc - back, 0), min(kc + back + 1, nf)
        u = np.zeros(n)
        for r, idx in enumerate(grp):
            blk = np.einsum("ijk->ij", dhat[np.ix_(idx, idx, np.arange(lo, hi))])
            blk = 0.5 * (blk / float(hi - lo) + (blk / float(hi - lo)).T)
            vals, vecs = np.linalg.eigh(blk)
            tvec = vecs[:, -1]
            if tvec[int(np.argmax(np.abs(tvec)))] < 0.0:
                tvec = -tvec
            lam[i, r] = float(vals[-1])
            u[idx] = np.sqrt(max(lam[i, r] - se[r], 1.0e-3 * max(lam[i, r], 1.0e-30))) * tvec
        cols.append(u)
    return cols, lam


def _select_sign_pattern(phi, f, z, sdiag, se, grp, freqs, fft, q, m, maxiter=15):
    """在候选组间符号图案中取谱预拟合后负对数似然最小的一个。

    逐块取组内主特征向量时，每个块的符号由特征向量的任意符号决定。式 (10.47) 的
    交叉项 ``S_ij h_ik h_jk* t^(i)_r t^(j)T_r`` 使同一测组内各阶形状的相对符号参与
    似然，取错符号会让第 2 阶及以后的初值落到镜像的局部极小（实测第 2 阶 MAC 由
    0.996 降到 0.180）。整阶反号是规范自由度，故固定第 0 阶全为正，只需比较
    ``2^((m-1) q)`` 个图案。

    初值处 ``S`` 的非对角元为零，交叉项消失，各图案的似然完全相同，无法直接排序；
    必须先让 ``S`` 的非对角元参与拟合，图案之间才分得开。为此对每个图案做一次
    限迭代的谱参数预拟合（``S_off`` 从零出发），以预拟合的负对数似然排序。预拟合
    不必收敛：实测 15 次迭代时正确图案已比错误图案低 40 以上（2400 s 数据）与
    100 以上（4800 s 数据），四个图案合计约 5~11 s。
    """
    pats = _sign_patterns(m, len(grp), limit=8)
    if len(pats) == 1:
        return phi
    from scipy.optimize import minimize

    n = fft.shape[0]
    ntri = m * (m - 1) // 2
    iu = np.triu_indices(m, 1)
    complex_s = _complex_spectral(len(grp))
    n_off = 2 * ntri if complex_s else ntri
    d_scale = float(np.mean(np.abs(fft) ** 2))
    if not np.isfinite(d_scale) or d_scale <= 0.0:
        d_scale = 1.0
    s_lo = float(np.log(max(d_scale * 1.0e-8, _SMIN)))
    s_hi = float(np.log(min(d_scale * 1.0e8, _SMAX)))
    s_off = float(d_scale * 1.0e8)
    lb = np.concatenate(
        [
            np.log(np.maximum(np.asarray(f, dtype=float) * 0.5, 1.0e-12)),
            np.full(m, np.log(_ZMIN)),
            np.full(m, s_lo),
            np.full(n_off, -s_off),
            np.full(len(grp), s_lo),
        ]
    )
    ub = np.concatenate(
        [
            np.log(np.asarray(f, dtype=float) * 2.0),
            np.full(m, np.log(_ZMAX)),
            np.full(m, s_hi),
            np.full(n_off, s_off),
            np.full(len(grp), s_hi),
        ]
    )
    xs0 = _pack_spectral(f, z, np.asarray(sdiag, dtype=float), np.zeros(ntri), se,
                         complex_s=complex_s)
    base_v = {key: np.eye(len(grp), dtype=complex) for key in _coherence_pairs(m, len(grp))}

    best_phi = phi
    best_val = np.inf
    for pat in pats:
        cand = _apply_sign_pattern(phi, pat, grp)
        val0 = nllf(*_unpack_spectral(xs0, m, len(grp), iu, complex_s), cand, base_v,
                    freqs, fft, grp, q)
        pen = float(val0) + max(1.0e4, 0.05 * abs(float(val0)))

        def obj(xv, _phi=cand, _pen=pen):
            val = nllf(*_unpack_spectral(xv, m, len(grp), iu, complex_s), _phi, base_v,
                       freqs, fft, grp, q)
            return float(val) if np.isfinite(val) else _pen

        res = minimize(
            obj,
            xs0,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxiter": int(maxiter), "ftol": 1.0e-12, "gtol": 1.0e-8, "maxls": 20},
        )
        val = float(res.fun) if np.isfinite(res.fun) else np.inf
        if val < best_val:
            best_val = val
            best_phi = cand
    return best_phi


def _initial_guess(freqs, fft, grp, m, q, f0):
    """初值：频率取第一条奇异值谱曲线的谱峰，振型取各阶频率处组内主特征向量。

    频率初值由 :func:`_peak_frequencies` 给出（平滑奇异值曲线 + 突出度选峰 +
    最小间隔强制）。模态形状初值取各阶初始频率附近带内平均的组内块主特征向量，
    按 ``sqrt(lambda_max - S_er)`` 缩放后拼接：组内同步使组内块近似秩一，但各组
    块的模态能量相差可达两个数量级，逐块归一化会丢失组间相对幅值。各组块拼接后
    只作逐列归一化，不施加正交化（式 10.47 只约束 ``u_i^T u_i = 1``）。

    模态力谱密度对角初值按共振峰高度反解：``E_k`` 在 ``f = f_i`` 处的组内块主
    特征值近似为 ``S_ii / (4 zeta_i^2) + S_er``（式 10.47 的传递函数在共振点取
    ``1/(2 zeta)``，加速度数据另有 ``(2 pi f_i)^{-2 qexp}`` 因子）。预测误差谱
    密度由 :func:`_noise_floor` 给出。
    """
    n, nf = fft.shape
    ng = len(grp)
    dhat = np.real(fft[:, None, :] * np.conj(fft)[None, :, :])
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    span = float(freqs[-1] - freqs[0]) if freqs.size > 1 else df
    width = max(3, int(round(0.01 * span / df)) | 1)
    min_sep = max(3.0 * df, 0.02 * span)
    half = max(3.0 * df, 0.01 * span)

    if f0 is None:
        f_init = _peak_frequencies(freqs, dhat, m, width, min_sep)
    else:
        arr = np.asarray(f0, dtype=float).reshape(-1)
        if arr.size != m:
            raise ValueError(f"频率初值应为 {m} 个，收到 {arr.size} 个")
        f_init = np.sort(arr)
    z_init = np.full(m, 0.01)

    se = _noise_floor(dhat, grp, width)
    cols, lam = _band_columns(dhat, grp, freqs, f_init, half, se)
    phi = _normalise_columns(cols, dhat, grp, m)

    sdiag = np.empty(m)
    for i in range(m):
        peak = float(np.max(lam[i]))
        sdiag[i] = max((peak - float(np.max(se))) * (2.0 * z_init[i]) ** 2, 1.0e-12) * (
            2.0 * np.pi * f_init[i]
        ) ** (2 * int(q))
    sdiag = np.maximum(sdiag, 1.0e-12)

    # 组间相对符号参与式 (10.47) 的交叉项，逐块取主特征向量得到的符号是任意的：
    # 以限迭代的谱预拟合排序候选图案，避免第 2 阶及以后的初值落到镜像局部极小。
    phi = _select_sign_pattern(
        phi, f_init, z_init, sdiag, se, grp, freqs, fft, q, m
    )

    return f_init, phi, sdiag, se


def _complex_spectral(ng: int) -> bool:
    """模态力谱密度矩阵是否取 Hermitian 复值，即 ``ng == 1`` 的情形。

    式 (10.47)–(10.49) 中 ``S`` 与 ``V`` 的整体相位是同一规范自由度
    （``S_ij -> S_ij e^{i alpha}``、``V_ij -> V_ij e^{-i alpha}`` 时 ``E_k``
    不变）。单测组时相干矩阵 ``V`` 不存在（``npair = 0``，式 (10.51) 的
    ``q x q`` 相干矩阵退化为 ``[1]``），``S_ij`` 的复相位不再与 ``V_ij`` 构成
    同一规范自由度，而是可辨识参数；多测组时相位由 ``V_ij`` 承担，``S`` 保持
    实对称。判据只依赖测组数，见模块说明第 2 条。
    """
    return int(ng) == 1


def _pack_spectral(f, z, sdiag, soff, se, complex_s: bool = False) -> np.ndarray:
    """谱参数打包为 ``[ln f, ln zeta, ln diag S, S_off, ln S_e]``（式 10.48–10.49）。

    ``soff`` 为式 (10.49) 的 ``Sv`` 中 ``S`` 的上三角非对角元。``complex_s`` 为假
    时（多测组，见模块说明第 2 条）按实值直接放入一段；为真时（单测组）拆成实部、
    虚部两段，长度由 ``ntri`` 变为 ``2 ntri``，顺序为先全部实部、后全部虚部。
    对角元取对数以保证式 (10.49) 的 ``S_ii > 0``，``f`` 与 ``zeta`` 同样取对数以
    保证式 (10.8) 的传递函数在可行域内有定义。
    """
    soff_arr = np.asarray(soff)
    if complex_s:
        soff_arr = np.asarray(soff_arr, dtype=complex).reshape(-1)
        off_block = np.concatenate([soff_arr.real, soff_arr.imag])
    else:
        off_block = np.asarray(soff_arr, dtype=float).reshape(-1)
    return np.concatenate(
        [
            np.log(np.asarray(f, dtype=float)),
            np.log(np.asarray(z, dtype=float)),
            np.log(np.asarray(sdiag, dtype=float)),
            off_block,
            np.log(np.asarray(se, dtype=float)),
        ]
    )


def _unpack_spectral(vec, m: int, ng: int, iu, complex_s: bool = False):
    """谱参数向量的逆变换，返回 ``(f, zeta, S, S_e)``（式 10.48–10.49）。

    ``S`` 由 ``m`` 个对角元（对数坐标，故恒为实数正）与 ``m(m-1)/2`` 个非对角元
    拼成。``complex_s`` 为假时非对角元取实值、构造上即为实对称；为真时非对角元由
    实部与虚部两段重建，``S`` 为按式 (10.49) 分块装配所需的 Hermitian 复矩阵
    （``S_ji = conj(S_ij)``）。``ng`` 决定 ``S_er`` 的个数，单测组时为 1。
    """
    arr = np.asarray(vec, dtype=float)
    ntri = iu[0].size
    n_off = 2 * ntri if complex_s else ntri
    f = np.exp(arr[:m])
    z = np.exp(arr[m : 2 * m])
    sdiag = np.exp(arr[2 * m : 3 * m])
    off = arr[3 * m : 3 * m + n_off]
    se = np.exp(arr[3 * m + n_off : 3 * m + n_off + ng])
    smat = np.zeros((m, m), dtype=complex)
    smat[np.diag_indices(m)] = sdiag
    if ntri:
        if complex_s:
            soff = off[:ntri] + 1j * off[ntri:]
            smat[iu] = soff
            smat[(iu[1], iu[0])] = np.conj(soff)
        else:
            smat[iu] = off
            smat[(iu[1], iu[0])] = off
    return f, z, smat, se


def _pack_v(vij, pairs, ng: int) -> np.ndarray:
    """相干矩阵的模、相位参数打包，按阶对依次给出 ``[模..., 相位...]``。"""
    out: list[float] = []
    for key in pairs:
        mat = np.asarray(vij[key], dtype=complex)
        mods = []
        phases = []
        for r in range(ng):
            for s in range(r + 1, ng):
                mods.append(float(abs(mat[r, s])))
                phases.append(float(np.angle(mat[r, s])))
        out.extend(mods)
        out.extend(phases)
    return np.asarray(out, dtype=float)


def _unpack_v(vec, pairs, ng: int, clip: bool = True) -> dict:
    """相干矩阵参数向量的逆变换，Hermitian 与对角元为 1 由构造保证。"""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    npair = ng * (ng - 1) // 2
    per = 2 * npair
    out = {}
    for t, key in enumerate(pairs):
        blk = arr[t * per : (t + 1) * per]
        mat = np.eye(ng, dtype=complex)
        k = 0
        for r in range(ng):
            for s in range(r + 1, ng):
                mod = float(blk[k])
                phase = float(blk[npair + k])
                k += 1
                if clip:
                    mod = min(max(mod, 0.0), _VMAX)
                mat[r, s] = mod * np.exp(1j * phase)
                mat[s, r] = np.conj(mat[r, s])
        out[key] = mat
    return out


def _expand_shapes(theta, phi0, bases) -> np.ndarray:
    """切空间坐标还原为单位范数形状（第 12.3 节，式 12.21 的约束参数化）。

    ``u_i = (u_i0 + Z_i theta_i) / ||u_i0 + Z_i theta_i||``，``Z_i`` 与 ``u_i0``
    正交且列正交归一，故 ``||u_i0 + Z_i theta_i||^2 = 1 + ||theta_i||^2``，归一化
    不会退化。``Z_i`` 为实数组时形状取实向量（每阶 ``n - 1`` 个坐标），为复数组
    时形状取复向量（每阶 ``2 m_0 - 2`` 个坐标，模块说明第 5 条）。

    单测组多阶时 ``phi0`` 的列是 ``m_0`` 维复坐标 ``alpha_i``，``Z_i`` 为
    ``C^{m_0}`` 中的切空间基，返回的坐标矩阵再经式 (13.6) 的 ``B_0`` 映射到
    通道空间（见 :func:`mpv` 内的 ``_unfold``）。``B_0`` 列正交归一，故该映射
    不改变各阶坐标的方差量级。
    """
    arr = np.asarray(theta, dtype=float).reshape(-1)
    phi0 = np.asarray(phi0)
    m = phi0.shape[1]
    dtype = complex if (np.iscomplexobj(phi0) or any(np.iscomplexobj(b) for b in bases)) else float
    coords = np.empty(phi0.shape, dtype=dtype)
    pos = 0
    for i in range(m):
        width = int(bases[i].shape[1])
        seg = arr[pos : pos + width]
        pos += width
        vec = phi0[:, i] + bases[i] @ seg
        coords[:, i] = vec / np.linalg.norm(vec)
    return coords


def _fix_signs(phi, S):
    """统一各阶模态形状的相位（实情形即符号）：绝对值最大的分量为正实数。

    整阶乘相位 ``u_i -> u_i e^{i alpha_i}`` 是式 (10.47) 的规范自由度，同时令
    ``S_ij -> S_ij e^{-i(alpha_i - alpha_j)}``（即 ``S -> D^* S D``，
    ``D = diag(e^{i alpha})``）即可保持 ``E_k`` 不变；``V_ij`` 的对角元固定为 1，
    故不参与变换。实情形下相位只有 ``0`` 与 ``pi`` 两种，退化为整阶反号，与
    第 10.4 节的符号规范一致。该变换与第 13 章 :func:`bayoma.multi_mode._solve`
    末尾的相位归一相同。
    """
    phi = np.array(phi, dtype=complex if np.iscomplexobj(phi) else float, copy=True)
    smat = np.array(S, dtype=complex, copy=True)
    m = phi.shape[1]
    for i in range(m):
        k = int(np.argmax(np.abs(phi[:, i])))
        if phi[k, i] == 0.0:
            continue
        if np.iscomplexobj(phi):
            phase = np.exp(-1j * np.angle(phi[k, i]))
            phi[:, i] = phi[:, i] * phase
            smat[i, :] = smat[i, :] * np.conj(phase)
            smat[:, i] = smat[:, i] * phase
        elif phi[k, i] < 0.0:
            phi[:, i] = -phi[:, i]
            smat[i, :] = -smat[i, :]
            smat[:, i] = -smat[:, i]
    return phi, smat


def _sign_patterns(m: int, ng: int, limit: int = 8) -> list:
    """组间相对符号的候选初值图案（式 10.47 交叉项的符号结构）。

    逐块取组内主特征向量时，每个 ``(i, r)`` 块的符号由特征向量的任意符号决定。
    式 (10.47) 的交叉项含 ``S_ij h_ik h_jk* t^(i)_r t^(j)T_r``：同一测组内各阶
    模态形状的相对符号由该项确定，取错符号会落入不同的局部极小。整阶反号是
    规范自由度（``S`` 的行列同步反号），故固定第 0 阶全为正，图案数为
    ``2^((m-1) q)``；组合过多时只枚举前 ``limit`` 个（按二进制顺序，含全同号与
    最低阶组合）。
    """
    if m <= 1 or ng <= 1:
        return [np.ones((m, ng))]
    total = (m - 1) * ng
    n_pat = min(2 ** total, int(limit))
    out = []
    for bits in range(n_pat):
        pat = np.ones((m, ng))
        k = 0
        for i in range(1, m):
            for r in range(ng):
                if (bits >> k) & 1:
                    pat[i, r] = -1.0
                k += 1
        out.append(pat)
    return out


def _apply_sign_pattern(phi, pat, grp) -> np.ndarray:
    """按图案把各阶模态形状的逐组块整体反号。"""
    out = np.array(phi, copy=True)
    for i in range(out.shape[1]):
        for r, idx in enumerate(grp):
            if pat[i, r] < 0.0:
                out[idx, i] = -out[idx, i]
    return out


def _fix_group_signs(phi, vij, grp):
    """按约定固定逐组块的整体相位（实情形即符号），并把该相位吸收进相干矩阵。

    ``U_i -> U_i D``（``D = diag(d_r)``，``|d_r| = 1``，各阶共用同一 ``D``）与
    ``V_ij -> D^H V_ij D`` 使式 (10.47) 的 ``E_k`` 不变，且 ``V_ij`` 的对角元仍为 1
    ——这是式 (10.47) 在 ``m >= 2`` 时唯一保对角约束的相位规范。

    此处取约定：第 r 组内**最低阶**（按输出次序的第 0 列）模态形状的模最大分量为
    正实数。基准必须取固定的某一阶，不能取「该组内块范数最大的那一阶」，因为后者在
    估计值与真值之间可能落在不同的阶上，从而给出互相矛盾的符号图案。模态形状为
    复值（单测组分支）时 ``d_r`` 取相应相位，单测组下该相位已被
    :func:`_fix_signs` 固定，故此处不改变结果。
    """
    phi = np.array(phi, copy=True)
    is_cpx = np.iscomplexobj(phi)
    ng = len(grp)
    d = np.ones(ng, dtype=complex)
    for r, idx in enumerate(grp):
        blk = phi[idx, 0]
        k = int(np.argmax(np.abs(blk)))
        if blk[k] == 0:
            continue
        if is_cpx:
            d[r] = np.exp(-1j * np.angle(blk[k]))
        elif np.real(blk[k]) < 0.0:
            d[r] = -1.0
        else:
            continue
        phi[idx, :] = phi[idx, :] * d[r]
    dmat = np.diag(d)
    out = {
        key: dmat.conj().T @ np.asarray(mat, dtype=complex) @ dmat
        for key, mat in vij.items()
    }
    return phi, out


def _reorder_pairs(vij, order):
    """按新的阶次重排相干矩阵字典，键仍取 ``(a, b)``（``a < b``）。"""
    m = len(order)
    if m == 1:
        # m = 1 时唯一的一项是式 (10.51) 的单阶相干矩阵，键固定为 (0, 0)
        return {(0, 0): np.asarray(vij[(0, 0)], dtype=complex)}
    out = {}
    for a in range(m):
        for b in range(a + 1, m):
            i, j = int(order[a]), int(order[b])
            mat = np.asarray(vij[(min(i, j), max(i, j))], dtype=complex)
            out[(a, b)] = mat if i < j else mat.conj().T
    return out


def _coherence_pairs(m: int, ng: int) -> list:
    """相干矩阵字典的键表。

    ``m >= 2`` 时每个模态阶对各有一个 ``q x q`` 相干矩阵，键为 ``(i, j)``、``i < j``。
    ``m = 1`` 时不存在阶对，但测组之间仍可以不完全相干（式 10.51），此时唯一的一项
    用键 ``(0, 0)`` 表示，与 :mod:`bayoma.asynchronous` 的单阶实现保持一致。单测组
    ``q = 1`` 时相干矩阵退化为 ``[1]``，无需参数化。
    """
    if m == 1:
        return [(0, 0)] if ng > 1 else []
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def mpv(
    freqs: np.ndarray,
    fft: np.ndarray,
    groups,
    n_mode: int,
    q: int = 0,
    f0=None,
    max_iter: int = 200,
    tol: float = 1e-8,
):
    """异步数据的多阶最可能值（MPV），第 10.4 节与第 13.5 节的交替极小化。

    交替更新两组参数：谱参数 ``{f_i, zeta_i, S, S_er}``（``S`` 的对角元取对数、
    非对角元直接取坐标值；多测组时为实值，单测组时实部与虚部各占一段，见模块
    说明第 2 条）与振型相关参数 ``{theta_i}``、``{V_ij}``（各非对角元取
    「模、相位」参数，模约束在 ``[0, 0.99]``）。收敛后追加一次联合精化，把全部
    参数拼成一个向量再极小化一次，以降低终止点偏离驻点的程度。

    Parameters
    ----------
    freqs : ndarray, shape (Nf,)
        频带内频率坐标（Hz）。
    fft : ndarray, shape (n, Nf), complex
        缩放 FFT。
    groups : list of list of int
        测组划分。
    n_mode : int
        频带内模态数 ``m``。
    q : int, optional
        数据类型指数，``0`` 加速度、``1`` 速度、``2`` 位移。
    f0 : array_like, optional
        各阶频率初值；缺省取第一条奇异值谱曲线的谱峰。
    max_iter : int, optional
        外层交替迭代上限，默认 200。
    tol : float, optional
        收敛容差，同时用于负对数似然函数的相对变化与参数步长。

    Returns
    -------
    AsynchronousMultiResult

    Notes
    -----
    输出前按频率升序重排各阶（``S`` 的行列、``V_ij`` 与 ``phi`` 的列同步重排），
    并统一各阶模态形状的符号（见 :func:`_fix_signs`）。
    """
    from .posterior import tangent_basis

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    fft = np.ascontiguousarray(np.asarray(fft, dtype=complex))
    if fft.ndim != 2:
        raise ValueError("fft 必须是二维数组，形状 (n, Nf)")
    n, nf = fft.shape
    if freqs.size != nf:
        raise ValueError("频率点数必须等于 fft 的列数")
    grp = _validate_groups(groups, n)
    ng = len(grp)
    m = int(n_mode)
    if m < 1:
        raise ValueError("n_mode 必须为正整数")
    if m > n:
        raise ValueError("频带内模态数不能超过测点数")

    f_init, phi, sdiag, se_init = _initial_guess(freqs, fft, grp, m, q, f0)
    z_init = np.full(m, 0.01)
    pairs = _coherence_pairs(m, ng)
    vij = {key: np.eye(ng, dtype=complex) for key in pairs}
    iu = np.triu_indices(m, 1)
    ntri = iu[0].size
    complex_s = _complex_spectral(ng)
    n_off = 2 * ntri if complex_s else ntri
    npair = ng * (ng - 1) // 2
    # 单测组且多阶时按式 (13.6) 参数化模态形状 ``phi = B_0 alpha``（模块说明
    # 第 5 条）：``B_0`` 取 ``A_0 = sum_k D_k`` 的前 m_0 个最大特征向量，在极小化
    # 过程中保持不变；``alpha`` 为 m_0 x m 复系数矩阵，初值取带内平均的模态形状
    # 初值在 ``B_0`` 上的坐标（见 :func:`_subspace_coordinates`）。其余情形形状取
    # 实向量，直接用通道空间的坐标。两条分支都在各自的坐标空间内按式 (12.21)
    # 作切空间参数化。
    complex_shape = _complex_shapes(ng, m)
    subspace_basis = None
    coords = None
    if complex_shape:
        dhat_full = np.einsum("ak,bk->abk", fft, fft.conj()).real
        m0 = int(min(m, n))
        subspace_basis = _initial_basis(dhat_full, m0)
        coords = _subspace_coordinates(phi, subspace_basis)
        phi = np.asarray(subspace_basis @ coords, dtype=complex)
    shape_origin = coords if coords is not None else np.asarray(phi)
    n_tan = int(sum(int(b.shape[1]) for b in
                    [tangent_basis(shape_origin[:, i]) for i in range(m)]))

    # 谱密度的上下界按数据量纲取：缩放 FFT 的平均谱能量给出 S 与 S_er 的自然
    # 尺度，相差八个数量级足以覆盖实际识别范围。若改用与数据无关的定界，有界
    # 拟牛顿法的线搜索会因搜索区间与曲率尺度失配而在初值处直接报告收敛。
    d_scale = float(np.mean(np.abs(fft) ** 2))
    if not np.isfinite(d_scale) or d_scale <= 0.0:
        d_scale = 1.0
    s_lo = float(np.log(max(d_scale * 1.0e-8, _SMIN)))
    s_hi = float(np.log(min(d_scale * 1.0e8, _SMAX)))
    f_lo = max(float(freqs[0]) * 0.5, 1.0e-12)
    f_hi = float(freqs[-1]) * 2.0
    s_off = float(d_scale * 1.0e8)

    lb_s = np.concatenate(
        [
            np.full(m, np.log(f_lo)),
            np.full(m, np.log(_ZMIN)),
            np.full(m, s_lo),
            # 非对角元：实部沿用 s_off，虚部取同样量级（complex_s 为真时两段）
            np.full(n_off, -s_off),
            np.full(ng, s_lo),
        ]
    )
    ub_s = np.concatenate(
        [
            np.full(m, np.log(f_hi)),
            np.full(m, np.log(_ZMAX)),
            np.full(m, s_hi),
            np.full(n_off, s_off),
            np.full(ng, s_hi),
        ]
    )
    lb_v = np.tile(
        np.concatenate([np.zeros(npair), np.full(npair, -np.pi)]), len(pairs)
    )
    ub_v = np.tile(
        np.concatenate([np.full(npair, _VMAX), np.full(npair, np.pi)]), len(pairs)
    )
    lb_y = np.concatenate([np.full(n_tan, -_TAN_MAX), lb_v])
    ub_y = np.concatenate([np.full(n_tan, _TAN_MAX), ub_v])

    # 非对角元初值取零（实部、虚部均为零）；参数向量的实际长度由
    # :func:`_pack_spectral` 依 complex_s 展开。
    xs = _pack_spectral(
        f_init, z_init, sdiag, np.zeros(ntri), se_init, complex_s=complex_s
    )
    f_cur, z_cur, S_cur, se_cur = _unpack_spectral(xs, m, ng, iu, complex_s)

    nllf_val = nllf(f_cur, z_cur, S_cur, se_cur, phi, vij, freqs, fft, grp, q)
    if not np.isfinite(nllf_val):
        raise ValueError("初值处的负对数似然函数非有限，请检查数据与测组划分")

    # 不可行点（谱矩阵非正定）的罚值。两条相互冲突的约束同时成立才不至于让
    # 极小化失效：其一，罚值须与目标函数同量级，若返回 ``1e300`` 或 ``inf``，有界
    # 拟牛顿法的线搜索在尝试跨越不可行区时因函数值跳变过大而判定相对下降收敛，
    # 迭代停在初值处；其二，罚值须严格高于任何可达的可行值，若按高斯似然的常数
    # 项 ``n N_f (ln pi + ln d_scale)`` 取，该常数可能低于真实可行点的函数值
    # （数据谱能量远高于 1 时尤甚），罚值区自身成为吸引子，最优点被推向 f、z、
    # S、Se 的边界。故以初值处的函数值为基准加有限余量。
    pen = float(nllf_val) + max(1.0e4, 0.05 * abs(float(nllf_val)))

    def _safe(val: float) -> float:
        return float(val) if np.isfinite(val) else pen

    # 模态形状的切空间展开。实分支（多测组或单阶）直接在通道空间内按
    # u_i = (u_i0 + Z_i theta_i)/||...|| 展开；单测组多阶分支先在 m_0 维坐标空间
    # 内展开 ``alpha``，再按式 (13.6) 过 ``B_0`` 映射到通道空间。两条分支的
    # 切空间基都用 :func:`bayoma.posterior.tangent_basis` 构造：实向量给出
    # ``n - 1`` 个方向，复向量给出 ``2 m_0 - 2`` 个方向（已剔除整体相位）。

    def _shape_state():
        """当前形状（或坐标）处的切空间展开基点与基。"""
        base = coords if coords is not None else np.asarray(phi)
        return base, [tangent_basis(base[:, i]) for i in range(m)]

    def _unfold(theta_vals, base, bs):
        """切空间坐标还原为通道空间模态形状与（复分支的）坐标矩阵。"""
        loc = _expand_shapes(theta_vals, base, bs)
        if subspace_basis is None:
            return np.asarray(loc), None
        return np.asarray(subspace_basis @ loc), np.asarray(loc)

    phi_base, bases = _shape_state()

    def spec_obj(xv):
        fv, zv, sv, sev = _unpack_spectral(xv, m, ng, iu, complex_s)
        return _safe(nllf(fv, zv, sv, sev, phi, vij, freqs, fft, grp, q))

    def shape_obj(yv):
        ph, _ = _unfold(yv[:n_tan], phi_base, bases)
        vv = _unpack_v(yv[n_tan:], pairs, ng)
        return _safe(nllf(f_cur, z_cur, S_cur, se_cur, ph, vv, freqs, fft, grp, q))

    nllf_val = nllf(f_cur, z_cur, S_cur, se_cur, phi, vij, freqs, fft, grp, q)
    if not np.isfinite(nllf_val):
        raise ValueError("初值处的负对数似然函数非有限，请检查数据与测组划分")

    converged = False
    n_iter = 0
    step_tol = max(np.sqrt(tol), 1.0e-6)

    for n_iter in range(1, int(max_iter) + 1):
        nllf_old = nllf_val
        xs_old = xs.copy()
        phi_old = phi.copy()
        vij_old = {key: val.copy() for key, val in vij.items()}

        # ---- 谱参数子问题：自然对数空间的有界拟牛顿极小化 ----
        res = minimize(
            spec_obj,
            xs,
            method="L-BFGS-B",
            bounds=list(zip(lb_s, ub_s)),
            options={"maxiter": 200, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
        )
        # 只接受使目标函数下降的步：线搜索若落入罚值区，返回点虽有限却更差，
        # 直接接受会让交替极小化沿罚值平台漂移到边界。
        if np.all(np.isfinite(res.x)) and float(res.fun) <= nllf_old:
            xs = res.x
        f_cur, z_cur, S_cur, se_cur = _unpack_spectral(xs, m, ng, iu, complex_s)

        # ---- 振型与相干子问题：切空间参数化 + 模相位参数化 ----
        phi_base, bases = _shape_state()
        y0 = np.concatenate([np.zeros(n_tan), _pack_v(vij, pairs, ng)])
        res2 = minimize(
            shape_obj,
            y0,
            method="L-BFGS-B",
            bounds=list(zip(lb_y, ub_y)),
            options={"maxiter": 200, "ftol": 1.0e-14, "gtol": 1.0e-10, "maxls": 40},
        )
        y = res2.x if (np.all(np.isfinite(res2.x)) and float(res2.fun) <= nllf_old) else y0
        phi, coords = _unfold(y[:n_tan], phi_base, bases)
        vij = _unpack_v(y[n_tan:], pairs, ng)

        nllf_val = nllf(f_cur, z_cur, S_cur, se_cur, phi, vij, freqs, fft, grp, q)
        if not np.isfinite(nllf_val):
            # 数值退化时退回上一步的可行点
            phi = phi_old
            vij = vij_old
            xs = xs_old
            f_cur, z_cur, S_cur, se_cur = _unpack_spectral(xs, m, ng, iu, complex_s)
            nllf_val = nllf_old
            break

        rel_l = abs(nllf_old - nllf_val) / max(abs(nllf_old), 1.0)
        f_old, z_old, S_old, se_old = _unpack_spectral(xs_old, m, ng, iu, complex_s)
        step = max(
            float(np.max(np.abs(f_cur / f_old - 1.0))),
            float(np.max(np.abs(z_cur / z_old - 1.0))),
            float(np.max(np.abs(np.real(np.diag(S_cur)) / np.real(np.diag(S_old)) - 1.0))),
            float(np.max(np.abs(se_cur / se_old - 1.0))),
        )
        if ntri:
            s_scale = np.sqrt(
                np.abs(np.real(np.diag(S_old))[iu[0]] * np.real(np.diag(S_old))[iu[1]])
            )
            d_re = np.abs(np.real(S_cur[iu]) - np.real(S_old[iu]))
            if complex_s:
                # 单测组时 S 的非对角元为复数，步长判据须同时比较实部与虚部
                d_re = np.maximum(d_re, np.abs(np.imag(S_cur[iu]) - np.imag(S_old[iu])))
            step = max(
                step,
                float(np.max(d_re / np.maximum(s_scale, 1.0e-300))),
            )
        step = max(step, float(np.max(np.abs(phi - phi_old))))
        if pairs:
            step = max(
                step,
                max(float(np.max(np.abs(vij[key] - vij_old[key]))) for key in pairs),
            )
        if rel_l < tol and step < step_tol:
            converged = True
            break

    # ---- 联合精化：谱参数、S、V、各阶切空间坐标同时极小化 ----
    phi_base, bases = _shape_state()
    n_spec = xs.size
    xj = np.concatenate([xs, np.zeros(n_tan), _pack_v(vij, pairs, ng)])
    lb_j = np.concatenate([lb_s, np.full(n_tan, -_TAN_MAX), lb_v])
    ub_j = np.concatenate([ub_s, np.full(n_tan, _TAN_MAX), ub_v])

    def joint_obj(vec):
        fv, zv, sv, sev = _unpack_spectral(vec[:n_spec], m, ng, iu, complex_s)
        ph, _ = _unfold(vec[n_spec : n_spec + n_tan], phi_base, bases)
        vv = _unpack_v(vec[n_spec + n_tan :], pairs, ng)
        return _safe(nllf(fv, zv, sv, sev, ph, vv, freqs, fft, grp, q))

    val_before = joint_obj(xj)
    res_j = minimize(
        joint_obj,
        xj,
        method="L-BFGS-B",
        bounds=list(zip(lb_j, ub_j)),
        options={"maxiter": 2000, "ftol": 1.0e-15, "gtol": 1.0e-11, "maxls": 60},
    )
    polished = False
    if np.all(np.isfinite(res_j.x)) and np.isfinite(res_j.fun) and res_j.fun <= val_before:
        xj = res_j.x
        polished = bool(res_j.fun < val_before - 1.0e-9 * max(abs(val_before), 1.0))

    f_fin, z_fin, S_fin, se_fin = _unpack_spectral(xj[:n_spec], m, ng, iu, complex_s)
    phi_fin, coords_fin = _unfold(xj[n_spec : n_spec + n_tan], phi_base, bases)
    vij_fin = _unpack_v(xj[n_spec + n_tan :], pairs, ng)
    best_val = float(joint_obj(xj))
    if polished:
        converged = True

    # 按频率升序排列各阶：极小化过程不约束频率次序，输出前统一排序，使结果与
    # 初值的次序无关；S 的行列、V_ij 与 phi 的列同步重排。
    order = np.argsort(np.asarray(f_fin, dtype=float))
    f_fin = np.asarray(f_fin, dtype=float)[order]
    z_fin = np.asarray(z_fin, dtype=float)[order]
    S_fin = np.asarray(S_fin, dtype=complex)[np.ix_(order, order)]
    phi_fin = np.asarray(phi_fin)[:, order]
    vij_fin = _reorder_pairs(vij_fin, order)

    phi_fin, S_fin = _fix_signs(phi_fin, S_fin)
    for i in range(m):
        nrm = float(np.linalg.norm(phi_fin[:, i]))
        if nrm > 0.0:
            phi_fin[:, i] = phi_fin[:, i] / nrm

    # 逐组块的整体符号是式 (10.47) 的规范自由度（见 :func:`_fix_group_signs`）：
    # 输出前按约定固定，使同一问题的任意两次识别给出可比对的模态形状与相干矩阵。
    phi_fin, vij_fin = _fix_group_signs(phi_fin, vij_fin, grp)

    LOG.debug(
        "异步多阶识别：%d 阶、%d 个测组、迭代 %d 次、收敛=%s、NLLF=%.6f",
        m,
        ng,
        n_iter,
        converged,
        best_val,
    )
    # 复分支的坐标矩阵按式 (13.6) 由最终的通道空间形状反算：``B_0`` 列正交归一，
    # 故 ``alpha = B_0^T phi`` 与 ``phi = B_0 alpha`` 严格互逆。输出前 :func:`_fix_signs`
    # 已给各阶乘上相位并归一化，直接沿用极小化过程中的坐标会与 ``phi`` 不一致。
    alpha_fin = None
    if subspace_basis is not None:
        alpha_fin = np.asarray(subspace_basis.T @ np.asarray(phi_fin, dtype=complex))

    return AsynchronousMultiResult(
        f=np.asarray(f_fin, dtype=float),
        z=np.asarray(z_fin, dtype=float),
        S=np.asarray(S_fin, dtype=complex),
        Se=np.asarray(se_fin, dtype=float),
        phi=np.asarray(phi_fin),
        groups=grp,
        coherence=vij_fin,
        nllf=float(best_val),
        n_iter=int(n_iter),
        converged=bool(converged),
        basis=(
            np.asarray(subspace_basis, dtype=float)
            if subspace_basis is not None
            else None
        ),
        alpha=alpha_fin,
    )


def posterior_covariance(
    result,
    freqs: np.ndarray,
    fft: np.ndarray,
    groups,
    q: int = 0,
    rel_step: float = 1e-4,
):
    """异步多阶问题的后验协方差矩阵（式 10.17，约束投影见式 12.21–12.27）。

    自由参数向量取

        y = [f/f_hat, z/z_hat, diag(S)/diag(S_hat), S_off/sqrt(S_ii S_jj),
             {S_er/S_er_hat}, {theta_i}, {delta mod, delta phase}]

    其中 ``theta_i`` 为模态形状在切空间中的坐标
    （``u_i = (u_i_hat + Z_i theta_i)/||...||``，``Z_i`` 与 ``u_i_hat`` 正交，
    自动满足单位范数约束 ``u_i^T u_i = 1``）；相干矩阵的非对角元以最可能值的
    模、相位为基准作加性增量。单测组且多阶时模态形状按式 (13.6) 取
    ``phi = B_0 alpha``，``theta_i`` 为 ``m_0`` 维复坐标 ``alpha_i`` 在切空间中的
    坐标，每阶实维数为 ``2 m_0 - 2``（模块说明第 5 条）；其余情形每阶为 ``n - 1``。
    谱参数按其最可能值无量纲化（非对角元按
    ``sqrt(S_ii S_jj)`` 无量纲化，以免最可能值为零时除零），使各自由参数在对角
    尺度上同量级；数值 Hessian 与正则化求逆沿用
    :func:`bayoma.posterior._hessian` 与 :func:`bayoma.posterior._regularised_inverse`。

    Parameters
    ----------
    result : AsynchronousMultiResult
        :func:`mpv` 的返回结果。
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
        各参数的变异系数：``f``、``z``、``S``（对角元）、``phi`` 为长度 m 的
        数组，``Se`` 为按测组的列表，``S_off`` 为非对角元的变异系数（单测组时
        为复数数组，实部与虚部两段各自求方差后拼接），``v`` 为键 ``(i, j)`` 的
        字典（每项含 ``mod`` 与 ``phase`` 两个数组）。
    """
    from .posterior import _hessian, _regularised_inverse, tangent_basis

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    fft = np.asarray(fft, dtype=complex)
    if fft.ndim != 2:
        raise ValueError("fft 必须是二维数组，形状 (n, Nf)")
    n, nf = fft.shape
    if freqs.size != nf:
        raise ValueError("频率点数必须等于 fft 的列数")
    grp = _validate_groups(groups, n)
    ng = len(grp)

    f_hat = np.asarray(result.f, dtype=float).reshape(-1)
    z_hat = np.asarray(result.z, dtype=float).reshape(-1)
    smat_hat = _as_square(result.S, f_hat.size)
    se_hat = np.asarray(result.Se, dtype=float).reshape(-1)
    phi_hat = _as_shapes(result.phi, n)
    m = f_hat.size
    if phi_hat.shape[1] != m:
        raise ValueError("模态形状的列数必须等于模态数")

    iu = np.triu_indices(m, 1)
    ntri = iu[0].size
    # 单测组时 S 取 Hermitian 复矩阵，非对角元的实部与虚部各占一段参数
    # （模块说明第 2 条），自由参数个数相应由 ntri 变作 2 ntri。
    complex_s = _complex_spectral(ng)
    n_off = 2 * ntri if complex_s else ntri
    pairs = _coherence_pairs(m, ng)
    npair = ng * (ng - 1) // 2
    sd_hat = np.abs(np.real(np.diag(smat_hat)))
    sd_hat = np.where(sd_hat > 0.0, sd_hat, 1.0e-30)
    so_hat = smat_hat[iu]
    if not complex_s:
        so_hat = np.real(so_hat)
    so_scale = np.sqrt(np.maximum(sd_hat[iu[0]] * sd_hat[iu[1]], 1.0e-300))
    v0 = _pack_v(result.coherence, pairs, ng)
    # 单测组且多阶时模态形状按式 (13.6) 写成 ``phi = B_0 alpha``，自由参数取
    # ``m_0`` 维坐标空间内的切空间坐标，每阶 ``2 m_0 - 2`` 个（模块说明第 5 条）；
    # 多测组或单阶时形状为实向量，每阶 ``n - 1`` 个。两条分支的整体相位都已由
    # :func:`_fix_signs` 固定，切空间内不再含零曲率方向。
    complex_shape = _complex_shapes(ng, m)
    if np.iscomplexobj(phi_hat) and not complex_shape:
        raise ValueError(
            "复值模态形状只在单测组且多阶（q = 1、m > 1）时成立，多测组时形状必须取实向量"
        )
    basis_hat = None
    shape_base = phi_hat
    if complex_shape:
        basis_hat = result.basis
        alpha_hat = result.alpha
        if basis_hat is None or alpha_hat is None:
            # 兼容未携带基与系数的结果对象：由通道空间形状反算坐标
            dhat_full = np.einsum("ak,bk->abk", fft, fft.conj()).real
            basis_hat = _initial_basis(dhat_full, int(min(m, n)))
            alpha_hat = _subspace_coordinates(phi_hat, basis_hat)
        basis_hat = np.asarray(basis_hat, dtype=float)
        shape_base = np.asarray(alpha_hat, dtype=complex)
    bases = [tangent_basis(shape_base[:, i]) for i in range(m)]
    n_tan = int(sum(int(b.shape[1]) for b in bases))
    n_spec = 3 * m + n_off + ng
    n_v = 2 * npair * len(pairs)

    def expand(y):
        fv = y[0:m] * f_hat
        zv = y[m : 2 * m] * z_hat
        sdiag = y[2 * m : 3 * m] * sd_hat
        if complex_s:
            blk = y[3 * m : 3 * m + n_off]
            soff = (blk[:ntri] + 1j * blk[ntri:]) * so_scale
        else:
            soff = y[3 * m : 3 * m + n_off] * so_scale
        sev = y[3 * m + n_off : n_spec] * se_hat
        smat = np.zeros((m, m), dtype=complex)
        smat[np.diag_indices(m)] = sdiag
        if ntri:
            smat[iu] = soff
            smat[(iu[1], iu[0])] = np.conj(soff) if complex_s else soff
        loc = _expand_shapes(y[n_spec : n_spec + n_tan], shape_base, bases)
        ph = loc if basis_hat is None else np.asarray(basis_hat @ loc)
        vv = _unpack_v(v0 + y[n_spec + n_tan :], pairs, ng)
        return fv, zv, smat, sev, ph, vv

    d_scale = float(np.mean(np.abs(fft) ** 2))
    if not np.isfinite(d_scale) or d_scale <= 0.0:
        d_scale = 1.0
    pen = float(n * nf * (_LOG_PI + np.log(max(d_scale, 1.0e-300)))) + 1.0e3

    def nllf_free(y):
        fv, zv, smat, sev, ph, vv = expand(y)
        if np.any(sev <= 0.0) or np.any(fv <= 0.0):
            return pen
        if np.any(zv <= 0.0) or np.any(zv >= 1.0):
            return pen
        if np.any(np.real(np.diag(smat)) <= 0.0):
            return pen
        val = nllf(fv, zv, smat, sev, ph, vv, freqs, fft, grp, q)
        return float(val) if np.isfinite(val) else pen

    # 谱参数块的基准点必须是各自由参数的最可能值本身：``S`` 的非对角元按
    # ``sqrt(S_ii S_jj)`` 无量纲化后取最可能值之比，不能取 1。取 1 相当于把
    # 非对角元换成 ``sqrt(S_ii S_jj)``，此时 ``Sv`` 接近奇异、``E0_k`` 的
    # Cholesky 分解失败，似然在基准点即返回罚值，数值 Hessian 因而全为零。
    y0_spec = np.concatenate(
        [np.ones(3 * m), np.real(so_hat) / so_scale]
        + ([np.imag(so_hat) / so_scale] if complex_s else [])
        + [np.ones(ng)]
    )
    y0 = np.concatenate([y0_spec, np.zeros(n_tan), np.zeros(n_v)])
    steps = np.full(y0.size, float(rel_step))
    hess = _hessian(nllf_free, y0, steps)
    cov = _regularised_inverse(hess)

    diag = np.maximum(np.diag(cov), 0.0)
    sd = np.sqrt(diag)
    # 各阶形状的变异系数按该阶切空间段的方差和开方；段宽由基的列数给出
    # （实形状 ``n - 1``，单测组复形状 ``2 m_0 - 2``），故按累计宽度定位而
    # 不写死 ``n - 1``。
    phi_sd = []
    pos = n_spec
    for i in range(m):
        width = int(bases[i].shape[1])
        var = float(np.sum(diag[pos : pos + width]))
        phi_sd.append(float(np.sqrt(max(var, 0.0))))
        pos += width
    coefv = {
        "f": sd[:m],
        "z": sd[m : 2 * m],
        "S": sd[2 * m : 3 * m],
        "Se": [float(v) for v in sd[3 * m + n_off : n_spec]],
        "phi": np.asarray(phi_sd),
        "S_off": (
            sd[3 * m : 3 * m + ntri] + 1j * sd[3 * m + ntri : 3 * m + n_off]
            if complex_s
            else sd[3 * m : 3 * m + n_off]
        ),
        "v": {},
    }
    for t, key in enumerate(pairs):
        blk = sd[n_spec + n_tan + t * 2 * npair : n_spec + n_tan + (t + 1) * 2 * npair]
        coefv["v"][key] = {"mod": blk[:npair], "phase": blk[npair:]}
    return cov, coefv


def simulate_asynchronous_multi(
    f,
    z,
    S,
    Se_list,
    phi,
    vij,
    groups,
    fs: float,
    duration: float,
    seed: int | None = None,
    q: int = 0,
) -> np.ndarray:
    """按式 (10.47)–(10.49) 的谱矩阵合成异步多阶环境振动数据。

    合成采用频域法：先在全频段上按式 (10.47) 构造 ``E_k``，再对每个频率点抽取
    协方差等于 ``E_k`` 的圆对称复高斯向量，最后由单边谱补全负频率半轴并逆变换
    回时程（与第 10.1 节的缩放 FFT 约定一致，``E[F_k F_k^*] = E_k``）。所得数据
    的缩放 FFT 样本协方差在统计意义上等于式 (10.47) 的谱矩阵。组间相干矩阵的复
    相位对应各组之间的时钟漂移；合成数据在时域上以记录长度为周期，这是频域合成
    的固有性质。

    Parameters
    ----------
    f, z : array_like, shape (m,)
        各阶固有频率（Hz）与阻尼比。
    S : ndarray, shape (m, m)
        模态力谱密度矩阵。
    Se_list : array_like, shape (q,)
        各测组的预测误差谱密度。
    phi : ndarray, shape (n, m)
        模态形状矩阵，各列欧氏范数应为 1。
    vij : dict
        各阶对的相干矩阵，键 ``(i, j)``（``i < j``）。
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
    phi_arr = np.asarray(phi)
    if np.iscomplexobj(phi_arr):
        # 单测组时模态形状可取复向量（模块说明第 5 条）；多测组时保持实值语义
        # 不变，由 :func:`theoretical_psd_block` 拒绝复形状。
        phi_arr = np.asarray(phi_arr, dtype=complex)
    else:
        phi_arr = np.asarray(phi_arr, dtype=float)
    if phi_arr.ndim != 2:
        raise ValueError("phi 必须是二维数组，形状 (n, m)")
    freqs_all = np.fft.rfftfreq(nsam, d=1.0 / fs)
    psd = theoretical_psd_block(f, z, S, Se_list, phi_arr, vij, freqs_all, groups, q)
    data = synthesize_time_history(psd, fs, nsam, seed=seed)
    LOG.info(
        "合成异步多阶数据：%d 通道、%d 个测组、%d 阶模态、%d 样本、%.1f s、fs=%.4g Hz",
        data.shape[1],
        len(groups),
        phi_arr.shape[1],
        data.shape[0],
        duration,
        fs,
    )
    return data
