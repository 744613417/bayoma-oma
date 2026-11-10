"""在 pcode 自带的现场实测数据上运行本实现，作为非合成数据的校核。

数据来源：`A:\20211224 bayoma pcode 20210601 (public)\\example\\field data\\`
（`data.mat` 为 12 通道、115200 点、采样率 64 Hz 的加速度记录；`band.mat` 给出四个识别
频带与各频带的固有频率初值）。该算例不含真值，因此核对的是结果的自洽性与合理性：
频率应落在给定频带内，阻尼比应为正的小量，振型应为单位范数。

运行：

    python python\\tests\\test_field_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bayoma  # noqa: E402

FIELD = Path(os.environ.get("BAYOMA_FIELD_DATA",
                            "pcode-public/example/field data"))


def main() -> int:
    if not (FIELD / "data.mat").exists():
        print(f"缺少现场数据 {FIELD / 'data.mat'}，跳过本项校核。")
        return 0

    data = loadmat(FIELD / "data.mat")
    band = loadmat(FIELD / "band.mat")

    tdata = np.asarray(data["tdata"], dtype=float)
    fs = float(np.asarray(data["fs"]).ravel()[0])
    f1f2 = np.asarray(band["f1f2"], dtype=float)
    f0_raw = np.asarray(band["f0"]).ravel()

    f0 = []
    for item in f0_raw:
        arr = np.atleast_1d(np.asarray(item, dtype=float)).ravel()
        f0.append([float(x) for x in arr])

    print(f"数据：{tdata.shape[0]} 点 × {tdata.shape[1]} 通道，fs = {fs:g} Hz，"
          f"记录长度 {tdata.shape[0] / fs:.0f} s")
    print(f"频带：{f1f2.tolist()}")
    print(f"初值：{f0}")

    out = bayoma.bayoma(tdata, fs, f1f2=f1f2, f0=f0, mute=1)

    failures = 0
    print("\n识别结果：")
    for i in range(len(out["f"])):
        lo, hi = None, None
        for j, (a, b) in enumerate(np.asarray(f1f2)):
            if a <= out["f"][i] <= b:
                lo, hi = a, b
        inside = lo is not None
        if not inside:
            failures += 1
        print(f"  阶 {i + 1}: f = {out['f'][i]:.5f} Hz  z = {out['z'][i]:.5f}  "
              f"Sii = {out['Sii'][i]:.4e}  Se = {out['Se'][i]:.4e}  "
              f"sn = {out['sn'][i]:.1f}  频带 {'内' if inside else '外'}")

    norms = np.linalg.norm(out["phi"], axis=0)
    print(f"\n振型范数：{np.round(norms, 9).tolist()}")
    if not np.allclose(norms, 1.0, atol=1e-9):
        failures += 1
        print("振型范数不为 1。")

    if np.any(np.asarray(out["z"]) <= 0.0) or np.any(np.asarray(out["z"]) >= 0.5):
        failures += 1
        print("存在越界的阻尼比。")

    print("\n后验变异系数：")
    for key in ("f", "z", "Sii", "Se", "phi", "rms"):
        vals = np.atleast_1d(np.asarray(out["coefv"][key], dtype=float))
        print(f"  {key:>4}: {np.round(vals, 6).tolist()}")

    if failures == 0:
        print("\n全部检查通过")
        return 0
    print(f"\n{failures} 项检查未通过")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
