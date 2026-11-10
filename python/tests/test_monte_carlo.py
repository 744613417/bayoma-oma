"""单模态估计器的统计校核：多个独立种子的偏差与后验区间宽度。

用法::

    python python\\tests\\test_monte_carlo.py [种子数]

缺省 20 个种子约需 190 s，100 个种子约需 1030 s。本脚本不在 ``test_smoke.py``
的快速自检内：它校核的是估计量的统计性质，不是单次识别是否成功。
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bayoma import api, single_mode, synth  # noqa: E402

FS = 100.0
DUR = 2400.0
BAND = [0.9, 1.1]
F_TRUE = 1.0
Z_TRUE = 0.01
PHI_TRUE = np.array([1.0, 2.0, 2.0]) / 3.0
S_TRUE = 1.0
SE_TRUE = 1.0


def run_once(seed: int) -> dict:
    """生成一份记录并识别，返回点估计与后验变异系数。"""
    data = synth.simulate_oma_data(
        np.array([F_TRUE]), np.array([Z_TRUE]), PHI_TRUE.reshape(-1, 1),
        np.array([[S_TRUE]]), SE_TRUE, FS, DUR, seed=seed, q=0,
    )
    spec = api.scaled_fft(data, FS, BAND)
    res = single_mode.mpv(spec.freqs, spec.psd_matrices, spec.d_total, q=0, f0=F_TRUE)
    cov = single_mode.posterior_covariance(
        res, spec.freqs, spec.psd_matrices, spec.d_total, q=0
    )
    cv = single_mode.coefficients_of_variation(res, cov)
    mac = abs(np.vdot(PHI_TRUE, res.phi)) / (
        np.linalg.norm(PHI_TRUE) * np.linalg.norm(res.phi)
    ) ** 2
    return {
        "f": float(res.f), "z": float(res.z), "S": float(res.S), "Se": float(res.Se),
        "mac": float(mac),
        "cv": {k: float(cv[k]) for k in ("f", "z", "S", "Se", "phi")},
    }


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    started = time.time()
    rows = [run_once(seed) for seed in range(1, count + 1)]

    print("种子数 %d，耗时 %.1f s" % (count, time.time() - started))
    print("\n点估计的偏差与离散度：")
    print("%-4s %12s %12s %12s %10s" % ("量", "均值", "标准差", "偏差", "相对标准差"))
    for key, true_value in (("f", F_TRUE), ("z", Z_TRUE), ("S", S_TRUE), ("Se", SE_TRUE)):
        values = np.array([row[key] for row in rows])
        print("%-4s %12.6f %12.3e %+12.3e %10.3e" % (
            key, values.mean(), values.std(ddof=1),
            values.mean() - true_value, values.std(ddof=1) / true_value))

    mac = np.array([row["mac"] for row in rows])
    print("振型 MAC：均值 %.8f，最小 %.8f" % (mac.mean(), mac.min()))

    print("\n后验变异系数与实测相对标准差的对照：")
    print("%-4s %12s %14s %8s" % ("量", "后验均值", "实测相对标准差", "实测/后验"))
    for key, true_value in (("f", F_TRUE), ("z", Z_TRUE), ("S", S_TRUE), ("Se", SE_TRUE)):
        values = np.array([row[key] for row in rows])
        posterior = float(np.mean([row["cv"][key] for row in rows]))
        actual = values.std(ddof=1) / true_value
        print("%-4s %12.4e %14.4e %8.3f" % (key, posterior, actual, actual / posterior))

    # 判定：偏差应在 2 个标准误以内，实测/后验比值应在 0.6 至 1.4 之间。
    ok = True
    for key, true_value in (("f", F_TRUE), ("z", Z_TRUE), ("S", S_TRUE), ("Se", SE_TRUE)):
        values = np.array([row[key] for row in rows])
        stderr = values.std(ddof=1) / np.sqrt(count)
        if abs(values.mean() - true_value) > 2.0 * stderr:
            print("偏差超出 2 个标准误：%s" % key)
            ok = False
        posterior = float(np.mean([row["cv"][key] for row in rows]))
        ratio = (values.std(ddof=1) / true_value) / posterior
        if not 0.6 <= ratio <= 1.4:
            print("实测/后验比值超出区间：%s = %.3f" % (key, ratio))
            ok = False
    print("\n全部通过" if ok else "\n存在超出判据的项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
