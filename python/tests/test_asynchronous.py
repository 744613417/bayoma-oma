"""异步数据的单模态贝叶斯识别校核（课本第 10.4 节，式 10.42–10.51）。

运行：

    python python\\tests\\test_asynchronous.py

全部通过时返回码为 0。校核分三组：

1. 退化性：全部测点同属一个测组（q = 1）时，异步数据的似然函数退化为第 12 章
   的同步单模态问题，故在同一份同步数据上，异步识别的最可能值应与
   :func:`bayoma.single_mode.mpv` 一致。
2. 识别精度：两个测组各两个测点（q = 2），组间存在复相干，核对频率、阻尼比、
   振型与相干矩阵非对角元的估计。
3. 后验不确定性：数值 Hessian 给出的频率、阻尼比变异系数，与第 15 章不确定度
   定律的预测值比较。定律中的带宽因子按实际选带折算，即
   ``kappa = 半带宽 / (f * zeta)``。

容差说明：单模态识别中频率的变异系数约为 5e-4，记录时长 4800 s 时同一批
20 组独立实现样本的阻尼比相对误差标准差约 4%，因此阻尼比的 5% 判据在单次
实现上处于统计波动量级。测试取固定随机种子以保证结果可复现，取 4800 s 记录
以抑制阻尼比的抽样波动。
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

failures: list[str] = []

# 校核用的结构参数，与第 10.4 节算例的量级一致。
F_TRUE = 1.0
Z_TRUE = 0.01
S_TRUE = 1.0
SE_TRUE = [1.0, 1.5]
FS = 100.0
DURATION = 4800.0
SEED = 4
BAND = (0.9, 1.1)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def _mode_shape() -> np.ndarray:
    """四个测点的模态形状，欧氏范数为 1。"""
    u = np.array([1.0, 2.0, 2.0, 3.0])
    return u / np.linalg.norm(u)


def _coherence() -> np.ndarray:
    """两测组的复相干矩阵，非对角元模 0.5、相位 pi/4。"""
    off = 0.5 * np.exp(1j * np.pi / 4.0)
    return np.array([[1.0, off], [np.conj(off), 1.0]])


def _simulate(duration: float = DURATION, seed: int = SEED) -> np.ndarray:
    """按式 (10.50) 合成两测组的异步环境振动数据。"""
    return asy.simulate_asynchronous(
        F_TRUE, Z_TRUE, S_TRUE, SE_TRUE, _mode_shape(), _coherence(), [[0, 1], [2, 3]],
        fs=FS, duration=duration, seed=seed,
    )


def _bandwidth_factor(f: float, z: float, freqs: np.ndarray) -> float:
    """把实际选带折算为不确定度定律的带宽因子 kappa。"""
    half = 0.5 * float(freqs[-1] - freqs[0])
    return half / (f * z)


def test_q1_degeneracy() -> None:
    """q = 1 时与同步单模态识别一致。"""
    phi = np.array([1.0, 2.0, 2.0]) / 3.0
    data = bayoma.simulate_oma_data(
        freqs_hz=[F_TRUE], damping=[Z_TRUE], phi=phi.reshape(-1, 1),
        S=np.array([[S_TRUE]]), Se=SE_TRUE[0], fs=FS, duration=2400.0, seed=7,
    )
    spec = bayoma.scaled_fft(data, FS, BAND)

    t0 = time.perf_counter()
    ref = bayoma.single_mode.mpv(spec.freqs, spec.psd_matrices, spec.d_total, 0)
    t1 = time.perf_counter()
    out = asy.mpv(spec.freqs, spec.fft, [[0, 1, 2]], 0)
    t2 = time.perf_counter()

    print(f"  同步单模态：f = {ref.f:.8f} Hz，z = {ref.z:.8f}，"
          f"S = {ref.S:.8f}，Se = {ref.Se:.8f}（{t1 - t0:.2f} s）")
    print(f"  异步单测组：f = {out.f:.8f} Hz，z = {out.z:.8f}，"
          f"S = {out.S:.8f}，Se = {out.Se[0]:.8f}（{t2 - t1:.2f} s）")

    tol = 1.0e-3
    for name, a, b in (
        ("频率", out.f, ref.f),
        ("阻尼比", out.z, ref.z),
        ("模态力谱密度", out.S, ref.S),
        ("预测误差谱密度", out.Se[0], ref.Se),
    ):
        rel = abs(a / b - 1.0)
        check(f"q = 1 时{name}与同步识别一致", rel < tol, f"相对差 {rel:.2e}")

    mac = float(bayoma.mac_matrix(out.phi, ref.phi)[0, 0])
    check("q = 1 时振型与同步识别一致", mac > 1.0 - 1.0e-6, f"MAC = {mac:.10f}")
    check("q = 1 时收敛", out.converged, f"迭代 {out.n_iter} 次")


def test_q2_identification() -> None:
    """q = 2 时识别频率、阻尼比、振型与组间相干。"""
    t0 = time.perf_counter()
    data = _simulate()
    t1 = time.perf_counter()
    spec = bayoma.scaled_fft(data, FS, BAND)
    t2 = time.perf_counter()
    out = asy.mpv(spec.freqs, spec.fft, [[0, 1], [2, 3]], 0)
    t3 = time.perf_counter()

    print(f"  合成用时 {t1 - t0:.2f} s，缩放 FFT 用时 {t2 - t1:.2f} s，"
          f"识别用时 {t3 - t2:.2f} s，N_f = {spec.nf}")

    ef = abs(out.f / F_TRUE - 1.0)
    ez = abs(out.z / Z_TRUE - 1.0)
    mac = float(bayoma.mac_matrix(out.phi, _mode_shape())[0, 0])
    mod = abs(out.coherence[0, 1])
    phase = float(np.angle(out.coherence[0, 1]))

    print(f"  f = {out.f:.6f} Hz（真值 {F_TRUE:.4f}，相对误差 {ef * 100:.3f}%）")
    print(f"  z = {out.z:.6f}（真值 {Z_TRUE:.4f}，相对误差 {ez * 100:.3f}%）")
    print(f"  S = {out.S:.6f}，Se = [{out.Se[0]:.6f} {out.Se[1]:.6f}]")
    print(f"  MAC = {mac:.8f}")
    print(f"  |v12| = {mod:.6f}，相位 = {phase:+.6f} rad"
          f"（真值 0.5 与 pi/4 = {np.pi / 4:.6f}）")
    print(f"  迭代 {out.n_iter} 次，收敛 = {out.converged}，"
          f"负对数似然函数 = {out.nllf:.6f}")

    check("两测组时频率误差小于 5%", ef < 0.05, f"{ef * 100:.3f}%")
    check("两测组时阻尼比误差小于 5%", ez < 0.05, f"{ez * 100:.3f}%")
    check("两测组时振型 MAC 大于 0.99", mac > 0.99, f"{mac:.8f}")
    check("组间相干模落在 0.2~0.8", 0.2 < mod < 0.8, f"|v12| = {mod:.6f}")
    check("两测组时收敛", out.converged, f"迭代 {out.n_iter} 次")
    check("相干矩阵为对角元 1 的 Hermitian 矩阵",
          np.allclose(np.diag(out.coherence), 1.0)
          and np.allclose(out.coherence, out.coherence.conj().T),
          "max|v - v^H| = %.2e" % float(np.max(np.abs(out.coherence - out.coherence.conj().T))))


def test_posterior_covariance() -> None:
    """后验变异系数与不确定度定律的预测比较。"""
    data = _simulate()
    spec = bayoma.scaled_fft(data, FS, BAND)
    out = asy.mpv(spec.freqs, spec.fft, [[0, 1], [2, 3]], 0)

    t0 = time.perf_counter()
    cov, coefv = asy.posterior_covariance(out, spec.freqs, spec.fft, [[0, 1], [2, 3]], 0)
    t1 = time.perf_counter()

    kappa = _bandwidth_factor(out.f, out.z, spec.freqs)
    law = bayoma.uncertainty_law_cov(
        out.f, out.z, out.S, float(np.mean(out.Se)), kappa, DURATION, 4, 0,
    )

    print(f"  后验协方差用时 {t1 - t0:.2f} s，自由参数 {cov.shape[0]} 个，"
          f"带宽因子 kappa = {kappa:.4f}")
    print(f"  变异系数：f = {coefv['f']:.6f}，z = {coefv['z']:.6f}，"
          f"S = {coefv['S']:.6f}，Se = [{coefv['Se'][0]:.6f} {coefv['Se'][1]:.6f}]")
    print(f"  不确定度定律：f = {law['f']:.6f}，z = {law['z']:.6f}")
    print(f"  振型变异系数 {coefv['phi']:.6f}，相干矩阵变异系数 {coefv['v']:.6f}")

    check("协方差矩阵对称正定",
          np.allclose(cov, cov.T) and bool(np.all(np.linalg.eigvalsh(cov) >= -1.0e-12)),
          f"最小特征值 {np.min(np.linalg.eigvalsh(cov)):.3e}")

    ratio_f = coefv["f"] / law["f"]
    ratio_z = coefv["z"] / law["z"]
    print(f"  比值：f {ratio_f:.4f}，z {ratio_z:.4f}")
    check("频率变异系数与不确定度定律的比值在 0.5~2.0", 0.5 < ratio_f < 2.0, f"{ratio_f:.4f}")
    check("阻尼比变异系数与不确定度定律的比值在 0.5~2.0", 0.5 < ratio_z < 2.0, f"{ratio_z:.4f}")

    check("变异系数全部为有限正数",
          all(np.isfinite(v) and v > 0.0 for v in (coefv["f"], coefv["z"], coefv["S"]))
          and all(v > 0.0 for v in coefv["Se"]),
          f"f {coefv['f']:.4e}")


def test_likelihood_consistency() -> None:
    """式 (10.42) 的快速计算与逐频率直接计算一致，且单次评估满足实时要求。"""
    data = _simulate(duration=480.0, seed=SEED)
    spec = bayoma.scaled_fft(data, FS, BAND)
    u = _mode_shape()
    v = _coherence()
    groups = [[0, 1], [2, 3]]

    val = asy.nllf(F_TRUE, Z_TRUE, S_TRUE, SE_TRUE, u, v, spec.freqs, spec.fft, groups, 0)

    ek = asy.theoretical_psd_block(F_TRUE, Z_TRUE, S_TRUE, SE_TRUE, u, v, spec.freqs, groups, 0)
    n, nf = spec.fft.shape
    direct = float(n * nf * np.log(np.pi))
    for k in range(nf):
        sign, logdet = np.linalg.slogdet(ek[:, :, k])
        if sign <= 0:
            direct = np.nan
            break
        direct += logdet
        direct += float(np.real(np.vdot(spec.fft[:, k], np.linalg.solve(ek[:, :, k], spec.fft[:, k]))))

    rel = abs(val / direct - 1.0) if np.isfinite(direct) else np.inf
    print(f"  快速式 {val:.9f}，逐频率直接式 {direct:.9f}，相对差 {rel:.2e}")
    check("式 (10.42) 的快速计算与直接计算一致", rel < 1.0e-10, f"{rel:.2e}")

    # 单次似然评估耗时，按 2400 s、fs = 100 Hz 的频带规模衡量
    big = _simulate(duration=2400.0, seed=SEED)
    bspec = bayoma.scaled_fft(big, FS, BAND)
    t0 = time.perf_counter()
    for _ in range(20):
        asy.nllf(F_TRUE, Z_TRUE, S_TRUE, SE_TRUE, u, v, bspec.freqs, bspec.fft, groups, 0)
    dt = (time.perf_counter() - t0) / 20.0
    print(f"  N_f = {bspec.nf}、n = 4 时单次评估 {dt * 1.0e3:.3f} ms")
    check("单次似然评估在 10 ms 量级", dt < 0.01, f"{dt * 1.0e3:.3f} ms")


def test_block_operator() -> None:
    """按组分块的振型矩阵与式 (10.50) 的谱矩阵结构。"""
    u = _mode_shape()
    v = _coherence()
    groups = [[0, 1], [2, 3]]
    uu = asy.block_operator(u, groups)
    prod = uu @ v @ uu.T
    check("按组分块矩阵的形状为 n x q", uu.shape == (4, 2), f"{uu.shape}")
    check("第 r 行块只有第 r 列非零",
          bool(np.all(np.abs(uu[0:2, 1]) == 0.0)) and bool(np.all(np.abs(uu[2:4, 0]) == 0.0)),
          f"第 1 组第 2 列 {np.max(np.abs(uu[0:2, 1])):.1e}")
    check("u~ v u~^T 的对角块等于 t_r t_r^T",
          np.allclose(prod[0:2, 0:2], np.outer(u[0:2], u[0:2]))
          and np.allclose(prod[2:4, 2:4], np.outer(u[2:4], u[2:4])),
          "最大偏差 %.2e" % float(np.max(np.abs(prod[0:2, 0:2] - np.outer(u[0:2], u[0:2])))))
    check("u~ v u~^T 的非对角块含相干因子",
          np.allclose(prod[0:2, 2:4], v[0, 1] * np.outer(u[0:2], u[2:4])),
          "最大偏差 %.2e" % float(np.max(np.abs(prod[0:2, 2:4] - v[0, 1] * np.outer(u[0:2], u[2:4])))))

    ek = asy.theoretical_psd_block(F_TRUE, Z_TRUE, S_TRUE, SE_TRUE, u, v, np.array([F_TRUE]), groups, 0)
    dk = float(bayoma.single_mode.dynamic_factor(F_TRUE, Z_TRUE, np.array([F_TRUE]), 0)[0])
    expected = S_TRUE * dk * (uu @ v @ uu.T) + np.diag([SE_TRUE[0], SE_TRUE[0], SE_TRUE[1], SE_TRUE[1]])
    dev = float(np.max(np.abs(ek[:, :, 0] - expected))) / float(np.max(np.abs(expected)))
    print(f"  式 (10.50) 谱矩阵的相对偏差 {dev:.2e}")
    check("谱矩阵按式 (10.50) 组装", dev < 1.0e-12, f"{dev:.2e}")
    check("谱矩阵为 Hermitian 正定",
          np.allclose(ek[:, :, 0], ek[:, :, 0].conj().T)
          and bool(np.all(np.linalg.eigvalsh(ek[:, :, 0]) > 0.0)),
          "最小特征值 %.4e" % float(np.min(np.linalg.eigvalsh(ek[:, :, 0]))))


def main() -> int:
    print(f"bayoma {bayoma.__version__}：异步数据单模态识别校核\n")
    for fn in (
        test_block_operator,
        test_likelihood_consistency,
        test_q1_degeneracy,
        test_q2_identification,
        test_posterior_covariance,
    ):
        print(f"--- {fn.__name__} ---")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {fn.__name__} 抛出异常：{type(exc).__name__}: {exc}")
            failures.append(fn.__name__)
        print()
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
