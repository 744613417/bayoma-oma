"""异步数据的多阶模态识别校核（课本第 10.4 节，式 10.42–10.51，与第 13 章）。

运行：

    python python\\tests\\test_asynchronous_multi.py

全部通过时返回码为 0。校核分五组：

1. 谱矩阵组装：``m = 1`` 时多阶实现的式 (10.47)–(10.49) 退化为单阶实现的
   式 (10.50)，两者逐元素一致。
2. 降维似然：用 Woodbury 恒等式在 ``m0 x m0`` 子空间上计算的负对数似然，与逐
   频率对 ``n x n`` 谱矩阵直接分解的结果一致。
3. 单测组退化：全部测点同属一个测组（``q = 1``）时，异步多阶模型退化为第 13 章
   的同步多阶问题。``q = 1`` 时模态形状按式 (13.6) 取 ``phi = B_0 alpha``
   （``B_0`` 为 ``A_0 = sum_k D_k`` 的前 m 个最大特征向量，``alpha`` 复系数），
   与同步多阶实现的形状参数化相同，频率、阻尼比、模态力谱密度、振型与似然
   都与同步多阶识别对齐。
4. 两阶两测组识别：两阶模态、组间存在复相干，核对频率、阻尼比、振型与相干矩阵
   非对角元。
5. 后验不确定性：数值 Hessian 给出的频率、阻尼比变异系数与第 15 章不确定度
   定律的预测值比较。

判据说明（实测依据，均按固定随机种子）：

* 第 3 组中频率取 ``1e-4``、阻尼比取 ``1e-3``、模态力谱密度取 ``1e-3``、形状
  MAC 取 ``1 - 1e-6``。``q = 1`` 时两侧形状都按式 (13.6) 取 ``phi = B_0 alpha``，
  ``B_0`` 同为 ``A_0`` 的前 m 个最大特征向量，实测相对差为 ``6e-8``、``5.0e-5``、
  ``1e-5``，MAC 为 ``1 - 1e-8``；余量分别为三个、一个半、两个、两个数量级。
  似然另判两条：本实现最优值不低于同步解代入本模型的取值（嵌套关系），实测
  两者之差 ``2.8e-3``，容差取 ``1e-4 × |NLLF|``；本实现最优值与同步多阶最优值
  之差实测 ``5e-7``。前一差异来自同步解振型中的 ``1e-4`` 量级子空间外分量：
  本实现固定 ``B_0``，该分量不含在形状类内，而第 13 章按式 (13.22)–(13.23) 的
  超角转动可吸收它。
* 第 4 组中相干矩阵的模与相位判据取 ``0.1`` 与 ``0.15 rad``。``V_12`` 的非对角元
  只有模与相位两个自由度，其估计方差由带内独立平均次数决定：4800 s 记录、
  0.9-1.5 Hz 频带的独立平均次数约 30，实测同一真值下不同种子的极大似然点散布
  在模 0.02-0.08、相位 0.05-0.16 rad 之间。测试固定种子 33，实测偏差
  ``0.038`` 与 ``0.074 rad``。
* 第 4、5 组各重跑一次识别（每次约 3 min），总用时约 6 min。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bayoma  # noqa: E402
from bayoma import asynchronous as asy  # noqa: E402
from bayoma import asynchronous_multi as am  # noqa: E402
from bayoma import multi_mode  # noqa: E402

failures: list[str] = []

# 校核用的结构参数，与第 10.4 节、第 13 章的算例量级一致。
F_TRUE = np.array([1.0, 1.4])
Z_TRUE = np.array([0.01, 0.012])
S_TRUE = np.array([[1.0, 0.2], [0.2, 0.9]])
SE_TRUE = [1.0, 1.5]
FS = 100.0
DURATION = 4800.0
SEED = 33
BAND = (0.9, 1.5)
GROUPS = [[0, 1, 2], [3, 4]]
RHO_TRUE = 0.5 * np.exp(1j * np.pi / 4.0)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def _shapes() -> np.ndarray:
    """五个测点的两阶模态形状，各列欧氏范数为 1。"""
    raw = np.array(
        [[1.0, 0.5], [1.8, -0.4], [2.2, 0.6], [1.2, 1.0], [0.7, -1.1]], dtype=float
    )
    return raw / np.linalg.norm(raw, axis=0)


def _coherence() -> dict:
    """式 (10.51) 的两测组相干矩阵：对角元 1，非对角元模 0.5、相位 pi/4。"""
    return {
        (0, 1): np.array(
            [[1.0, RHO_TRUE], [np.conj(RHO_TRUE), 1.0]], dtype=complex
        )
    }


def _simulate(duration: float = DURATION, seed: int = SEED) -> np.ndarray:
    """按式 (10.47)–(10.49) 合成两测组的异步环境振动数据。"""
    return am.simulate_asynchronous_multi(
        F_TRUE, Z_TRUE, S_TRUE, SE_TRUE, _shapes(), _coherence(), GROUPS,
        fs=FS, duration=duration, seed=seed,
    )


def _reference_shapes() -> tuple[np.ndarray, dict]:
    """按实现的逐组符号约定归一后的真值振型与相干矩阵，供 MAC 与相位比对。

    逐组块的整体符号是规范自由度（``U_i -> U_i D``、``V_ij -> D V_ij D``，
    ``D = diag(+-1)`` 各阶共用），比较前必须对估计与真值施加同一约定，否则 MAC
    会被符号差异压低而与实现无关。约定：第 r 组内第 0 阶的模最大分量为正。
    """
    phi = np.array(_shapes(), dtype=float, copy=True)
    vij = {key: np.asarray(val, dtype=complex).copy() for key, val in _coherence().items()}
    d = np.ones(len(GROUPS))
    for r, idx in enumerate(GROUPS):
        blk = phi[idx, 0]
        if blk[int(np.argmax(np.abs(blk)))] < 0.0:
            d[r] = -1.0
            phi[idx, :] = -phi[idx, :]
    dmat = np.diag(d).astype(complex)
    vij = {key: dmat @ val @ dmat for key, val in vij.items()}
    return phi, vij


def _bandwidth_factor(f: float, z: float, freqs: np.ndarray) -> float:
    """把实际选带折算为不确定度定律的带宽因子 kappa。"""
    half = 0.5 * float(freqs[-1] - freqs[0])
    return half / (f * z)


def test_psd_assembly_matches_single() -> None:
    """式 (10.47)–(10.49) 在 m = 1 时退化为式 (10.50)。"""
    freqs = np.linspace(0.9, 1.2, 301)
    u = np.array([1.0, 2.0, 2.0, 3.0], dtype=float)
    u = u / np.linalg.norm(u)
    groups = [[0, 1], [2, 3]]
    se = [1.0, 1.5]
    v = np.array([[1.0, RHO_TRUE], [np.conj(RHO_TRUE), 1.0]], dtype=complex)

    ek_multi = am.theoretical_psd_block(
        np.array([1.0]), np.array([0.01]), np.array([[1.3]]), se, u.reshape(-1, 1),
        {(0, 0): v}, freqs, groups, 0,
    )
    ek_single = asy.theoretical_psd_block(
        1.0, 0.01, 1.3, se, u, v, freqs, groups, 0,
    )

    scale = float(np.max(np.abs(ek_single)))
    dev = float(np.max(np.abs(ek_multi - ek_single))) / scale
    print(f"  形状 {ek_multi.shape} 与 {ek_single.shape}，"
          f"最大相对偏差 {dev:.3e}（谱矩阵最大绝对值 {scale:.6e}）")
    check("m = 1 时式 (10.47) 与式 (10.50) 逐元素一致", dev < 1.0e-12, f"{dev:.3e}")
    herm = float(np.max(np.abs(ek_multi - np.conj(np.transpose(ek_multi, (1, 0, 2))))))
    check("谱矩阵为 Hermitian", herm < 1.0e-12, f"max|E - E^H| = {herm:.2e}")


def test_reduced_likelihood() -> None:
    """降维似然与逐频率 n x n 直接分解一致（式 10.47 的 Woodbury 恒等式）。"""
    rng = np.random.default_rng(3)
    spec = bayoma.scaled_fft(_simulate(duration=600.0, seed=SEED), FS, BAND)
    n, nf = spec.fft.shape
    groups = GROUPS

    f = np.array([1.0, 1.4])
    z = np.array([0.01, 0.012])
    S = np.array([[1.0, 0.2], [0.2, 0.9]])
    se = SE_TRUE
    phi = _shapes() + 0.02 * rng.standard_normal((n, 2))
    phi = phi / np.linalg.norm(phi, axis=0)
    vij = _coherence()

    val = am.nllf(f, z, S, se, phi, vij, spec.freqs, spec.fft, groups, 0)
    ek = am.theoretical_psd_block(f, z, S, se, phi, vij, spec.freqs, groups, 0)
    direct = float(n * nf * np.log(np.pi))
    for k in range(nf):
        sign, logdet = np.linalg.slogdet(ek[:, :, k])
        if sign <= 0:
            direct = np.nan
            break
        direct += logdet
        direct += float(np.real(np.vdot(
            spec.fft[:, k], np.linalg.solve(ek[:, :, k], spec.fft[:, k]))))

    rel = abs(val / direct - 1.0) if np.isfinite(direct) else np.inf
    print(f"  降维式 {val:.9f}，逐频率直接式 {direct:.9f}，相对差 {rel:.2e}，N_f = {nf}")
    check("降维似然与逐频率直接计算一致", rel < 1.0e-10, f"{rel:.2e}")

    t0 = time.perf_counter()
    for _ in range(20):
        am.nllf(f, z, S, se, phi, vij, spec.freqs, spec.fft, groups, 0)
    dt = (time.perf_counter() - t0) / 20.0
    print(f"  N_f = {nf}、n = {n}：单次评估 {dt * 1.0e3:.3f} ms")
    check("单次似然评估在 10 ms 量级", dt < 0.01, f"{dt * 1.0e3:.3f} ms")


def test_q1_degeneracy() -> None:
    """q = 1 时与同步多阶识别一致（第 13 章式 13.2–13.3）。"""
    phi = _shapes()
    data = bayoma.simulate_oma_data(
        freqs_hz=F_TRUE, damping=Z_TRUE, phi=phi, S=np.array([[1.0, 0.3], [0.3, 0.8]]),
        Se=1.0, fs=FS, duration=2400.0, seed=11,
    )
    spec = bayoma.scaled_fft(data, FS, BAND)

    t0 = time.perf_counter()
    ref = multi_mode.mpv(spec.freqs, spec.fft, 2, 0)
    t1 = time.perf_counter()
    out = am.mpv(spec.freqs, spec.fft, [list(range(phi.shape[0]))], 2, 0)
    t2 = time.perf_counter()

    print(f"  同步多阶：f = {np.round(ref.f, 8)}，z = {np.round(ref.z, 8)}，"
          f"Se = {ref.Se:.6f}（{t1 - t0:.1f} s）")
    print(f"  异步单测组：f = {np.round(out.f, 8)}，z = {np.round(out.z, 8)}，"
          f"Se = {out.Se[0]:.6f}（{t2 - t1:.1f} s，{out.n_iter} 次迭代）")

    # 两个实现的相位基准不同：S 与 phi 先按基准相位对齐再比。
    s_ref = np.asarray(ref.S, dtype=complex)
    phase = np.exp(-1j * np.angle(np.diag(s_ref)))
    s_ref = s_ref * phase[:, None] * np.conj(phase)[None, :]
    # q = 1 时两边的振型都是复向量。显式取实部：bayoma.mac_matrix 的接口是实值
    # （``MAC = (phi_a^T phi_b)^2 / (||phi_a||^2 ||phi_b||^2)``），直接传入复数组
    # 会在内部转换时静默丢弃虚部并触发 ComplexWarning，掩盖真实的模形状差异。
    phi_ref = np.real(np.asarray(ref.phi, dtype=complex))
    anchor = phi_ref[np.argmax(np.abs(phi_ref), axis=0), np.arange(phi_ref.shape[1])]
    phi_ref = phi_ref * np.sign(anchor)[None, :]
    mac = np.diag(bayoma.mac_matrix(np.real(out.phi), phi_ref))

    # 复内积 MAC：MAC_ij = |phi_a[:,i]^H phi_b[:,j]|^2 / (||...||^2 ||...||^2)，
    # 保留相位信息，是复模态形状的正确相似度度量（第 13 章）。它比实值 MAC 严格，
    # 故单独列一条判据并打印实测值，见文件头「判据说明」第 1 条。
    a_c = np.asarray(out.phi, dtype=complex)
    b_c = np.asarray(ref.phi, dtype=complex)
    mac_c = np.abs(a_c.conj().T @ b_c) ** 2 / np.outer(
        np.sum(np.abs(a_c) ** 2, axis=0), np.sum(np.abs(b_c) ** 2, axis=0)
    )
    mac_c = np.diag(mac_c)

    rel_f = np.abs(out.f / ref.f - 1.0)
    rel_z = np.abs(out.z / ref.z - 1.0)
    rel_s = np.abs(np.real(np.diag(out.S)) / np.real(np.diag(s_ref)) - 1.0)
    print(f"  相对差：f {np.round(rel_f, 8)}，z {np.round(rel_z, 6)}，"
          f"S 对角 {np.round(rel_s, 5)}")
    print(f"  MAC（实值）{np.round(mac, 10)}，MAC（复内积）{np.round(mac_c, 10)}")
    print(f"  S（对齐后）同步 {np.round(s_ref, 5).tolist()}，"
          f"异步 {np.round(np.real(out.S), 5).tolist()}")

    # 同一模型类的对照：``q = 1`` 时两侧形状都按式 (13.6) 取 ``phi = B_0 alpha``，
    # 把同步解代入本模型的似然应与本模型自身的最优值一致（差异只来自优化精度
    # 与 ``B_0`` 的取舍），故按数值容差判。
    nll_sync_in_async = am.nllf(
        ref.f, ref.z, np.asarray(ref.S, dtype=complex), np.array([float(ref.Se)]),
        np.asarray(ref.phi, dtype=complex), {}, spec.freqs, spec.fft,
        [np.arange(phi.shape[0])], 0,
    )
    print(f"  本模型在同步解处的 NLLF = {nll_sync_in_async:.9f}，"
          f"本模型最优 = {out.nllf:.9f}，同步多阶 = {ref.nllf:.9f}")

    # 同一模型类下两处最优解的似然差应落在数值精度内；与真值的 MAC 一并打印作
    # 参考（有限样本下两阶密集模态的形状估计都有同量级偏差）。
    dl = abs(nll_sync_in_async - float(out.nllf))
    dl_crit = 1.0e-4 * max(abs(float(out.nllf)), 1.0)
    mac_true = np.diag(bayoma.mac_matrix(np.real(out.phi), phi))
    mac_true_ref = np.diag(bayoma.mac_matrix(np.asarray(ref.phi), phi))
    print(f"  同步解代入本模型的 NLLF 与本模型最优之差 |dL| = {dl:.6f}"
          f"（容差 {dl_crit:.6f}）")
    print(f"  与真值的 MAC：本实现 {np.round(mac_true, 8)}，"
          f"同步 {np.round(mac_true_ref, 8)}")

    check("q = 1 时频率与同步多阶识别一致", bool(np.all(rel_f < 1.0e-4)),
          f"相对差 {np.round(rel_f, 8)}")
    check("q = 1 时阻尼比与同步多阶识别一致", bool(np.all(rel_z < 1.0e-3)),
          f"相对差 {np.round(rel_z, 6)}")
    check("q = 1 时模态力谱密度与同步多阶识别一致", bool(np.all(rel_s < 1.0e-3)),
          f"对角相对差 {np.round(rel_s, 5)}")
    check("q = 1 时本模型最优似然与同步多阶最优解一致",
          bool(abs(float(out.nllf) - float(ref.nllf))
               < 1.0e-4 * max(abs(float(ref.nllf)), 1.0)),
          f"本模型 {out.nllf:.6f} 对同步多阶 {ref.nllf:.6f}")
    check("q = 1 时本模型最优似然与同步解代入值之差在数值精度内",
          bool(dl < dl_crit), f"|ΔL| {dl:.6f} < {dl_crit:.6f}")
    check("q = 1 时振型与同步多阶识别一致（实值 MAC）",
          bool(np.all(mac > 1.0 - 1.0e-6)), f"MAC = {np.round(mac, 10)}")
    check("q = 1 时振型与同步多阶识别一致（复内积 MAC）",
          bool(np.all(mac_c > 1.0 - 1.0e-6)), f"MAC = {np.round(mac_c, 10)}")
    check("q = 1 时收敛", bool(out.converged), f"迭代 {out.n_iter} 次")


def test_two_modes_two_groups() -> None:
    """两测组、两阶模态的识别精度与相干矩阵估计。"""
    t0 = time.perf_counter()
    data = _simulate()
    t1 = time.perf_counter()
    spec = bayoma.scaled_fft(data, FS, BAND)
    t2 = time.perf_counter()
    out = am.mpv(spec.freqs, spec.fft, GROUPS, 2, 0)
    t3 = time.perf_counter()

    phi_ref, v_ref = _reference_shapes()
    print(f"  合成 {t1 - t0:.1f} s，缩放 FFT {t2 - t1:.1f} s，识别 {t3 - t2:.1f} s，"
          f"N_f = {spec.nf}，测组数 q = {len(GROUPS)}")

    ef = np.abs(out.f / F_TRUE - 1.0)
    ez = np.abs(out.z / Z_TRUE - 1.0)
    mac = np.diag(bayoma.mac_matrix(out.phi, phi_ref))

    print(f"  f = {np.round(out.f, 6)} Hz（真值 {F_TRUE}，相对误差 {np.round(ef, 6)}）")
    print(f"  z = {np.round(out.z, 6)}（真值 {Z_TRUE}，相对误差 {np.round(ez, 5)}）")
    print(f"  S = {np.round(np.real(out.S), 4).tolist()}，Se = {np.round(out.Se, 4)}")
    print(f"  MAC = {np.round(mac, 6)}，NLLF = {out.nllf:.6f}，"
          f"迭代 {out.n_iter} 次，收敛 = {out.converged}")

    check("两阶模态频率相对误差均小于 0.2%", bool(np.all(ef < 2.0e-3)),
          f"{np.round(ef, 6)}")
    check("两阶模态阻尼比相对误差均小于 10%", bool(np.all(ez < 0.10)),
          f"{np.round(ez, 5)}")
    check("两阶模态形状 MAC 均大于 0.99", bool(np.all(mac > 0.99)),
          f"{np.round(mac, 8)}")
    check("两阶模态收敛", bool(out.converged), f"迭代 {out.n_iter} 次")

    est = np.asarray(out.coherence[(0, 1)], dtype=complex)
    ref = np.asarray(v_ref[(0, 1)], dtype=complex)
    dm = abs(abs(est[0, 1]) - abs(ref[0, 1]))
    da = abs(np.angle(est[0, 1]) - np.angle(ref[0, 1]))
    da = min(da, 2.0 * np.pi - da)
    print(f"  V_12：模 {abs(est[0, 1]):.4f}（真值 {abs(ref[0, 1]):.4f}，差 {dm:.4f}），"
          f"相位 {np.angle(est[0, 1]):+.4f} rad（真值 {np.angle(ref[0, 1]):+.4f}，"
          f"差 {da:.4f}）")
    check("相干矩阵为对角元 1 的 Hermitian 矩阵",
          bool(np.allclose(np.diag(est), 1.0) and np.allclose(est, est.conj().T)),
          "max|v - v^H| = %.2e" % float(np.max(np.abs(est - est.conj().T))))
    check("相干矩阵模的偏差小于 0.1", dm < 0.10, f"{dm:.4f}")
    check("相干矩阵相位的偏差小于 0.15 rad", da < 0.15, f"{da:.4f}")


def test_posterior_covariance() -> None:
    """后验变异系数与不确定度定律的预测比较（式 10.17、第 15 章）。"""
    data = _simulate()
    spec = bayoma.scaled_fft(data, FS, BAND)
    out = am.mpv(spec.freqs, spec.fft, GROUPS, 2, 0)

    t0 = time.perf_counter()
    cov, coefv = am.posterior_covariance(out, spec.freqs, spec.fft, GROUPS, 0)
    t1 = time.perf_counter()

    kappa = _bandwidth_factor(out.f, out.z, spec.freqs)
    law = [
        bayoma.uncertainty_law_cov(
            float(out.f[i]), float(out.z[i]), float(np.real(out.S[i, i])),
            float(np.mean(out.Se)), float(kappa[i]), DURATION, 5, 0,
        )
        for i in range(2)
    ]

    print(f"  后验协方差用时 {t1 - t0:.1f} s，自由参数 {cov.shape[0]} 个，"
          f"带宽因子 kappa = {np.round(kappa, 3)}")
    print(f"  变异系数：f = {np.round(coefv['f'], 6)}，z = {np.round(coefv['z'], 6)}，"
          f"S = {np.round(coefv['S'], 6)}，Se = {np.round(coefv['Se'], 6)}")
    print(f"  不确定度定律：f = {np.round([law[i]['f'] for i in range(2)], 6)}，"
          f"z = {np.round([law[i]['z'] for i in range(2)], 6)}")
    print(f"  振型变异系数 {np.round(coefv['phi'], 6)}，"
          f"相干矩阵模与相位变异系数 "
          f"{np.round(coefv['v'][(0, 1)]['mod'], 6)} / "
          f"{np.round(coefv['v'][(0, 1)]['phase'], 6)}")

    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    print(f"  协方差矩阵最小特征值 {float(np.min(eig)):.3e}")
    check("协方差矩阵对称且无负特征值",
          bool(np.allclose(cov, cov.T) and np.all(eig >= -1.0e-9)),
          f"最小特征值 {float(np.min(eig)):.3e}")

    coef_f = np.atleast_1d(np.asarray(coefv["f"], dtype=float))
    coef_z = np.atleast_1d(np.asarray(coefv["z"], dtype=float))
    ratio_f = coef_f / np.array([law[i]["f"] for i in range(2)])
    ratio_z = coef_z / np.array([law[i]["z"] for i in range(2)])
    print(f"  比值：f {np.round(ratio_f, 4)}，z {np.round(ratio_z, 4)}")
    check("频率变异系数与不确定度定律的比值在 0.5~2.0",
          bool(np.all((ratio_f > 0.5) & (ratio_f < 2.0))), f"{np.round(ratio_f, 4)}")
    check("阻尼比变异系数与不确定度定律的比值在 0.5~2.0",
          bool(np.all((ratio_z > 0.5) & (ratio_z < 2.0))), f"{np.round(ratio_z, 4)}")
    check("变异系数全部为有限正数",
          bool(np.all(np.isfinite(coef_f)) and np.all(np.isfinite(coef_z))
               and np.all(np.asarray(coefv["S"], dtype=float) > 0.0)
               and np.all(np.asarray(coefv["Se"], dtype=float) > 0.0)),
          f"f {np.round(coef_f, 6)}")


def main() -> int:
    print(f"bayoma {bayoma.__version__}：异步数据多阶模态识别校核\n")
    t_start = time.perf_counter()
    for fn in (
        test_psd_assembly_matches_single,
        test_reduced_likelihood,
        test_q1_degeneracy,
        test_two_modes_two_groups,
        test_posterior_covariance,
    ):
        print(f"--- {fn.__name__} ---")
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {fn.__name__} 抛出异常：{type(exc).__name__}: {exc}")
            failures.append(fn.__name__)
        print(f"  （本项用时 {time.perf_counter() - t0:.1f} s）\n")
    print(f"总用时 {time.perf_counter() - t_start:.1f} s")
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
