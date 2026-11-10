"""示例：单模态识别。

生成一段三通道合成记录，识别 1.0 Hz 附近的一阶模态，并与真值比较。
运行：

    python examples\\py_single_mode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import bayoma  # noqa: E402

# 真值：一阶模态，频率 1.0 Hz，阻尼比 1%，振型 [1, 2, 2]/3
f_true = np.array([1.0])
z_true = np.array([0.01])
phi_true = np.array([[1.0], [2.0], [2.0]]) / 3.0
S_true = np.array([[1.0]])
Se_true = 1.0
fs = 100.0

data = bayoma.simulate_oma_data(
    freqs_hz=f_true,
    damping=z_true,
    phi=phi_true,
    S=S_true,
    Se=Se_true,
    fs=fs,
    duration=2400.0,
    seed=7,
)
print(f"数据：{data.shape[0]} 样本、{data.shape[1]} 通道、{data.shape[0] / fs:.0f} s")

out = bayoma.bayoma(data, fs, f1f2=[[0.9, 1.1]], f0=[[1.0]], mute=1)

print("\n识别结果")
print(f"  频率    : {out['f'][0]:.6f} Hz（真值 {f_true[0]:.6f}）")
print(f"  阻尼比  : {out['z'][0]:.6f}（真值 {z_true[0]:.6f}）")
print(f"  模态力谱: {out['Sii'][0]:.6f}（真值 {S_true[0, 0]:.6f}）")
print(f"  噪声谱  : {out['Se'][0]:.6f}（真值 {Se_true:.6f}）")
print(f"  信噪比  : {out['sn'][0]:.1f}")

mac = bayoma.mac_matrix(out["phi"], phi_true)
print(f"  振型    : {np.round(out['phi'].ravel(), 6)}，MAC = {mac[0, 0]:.6f}")

cv = out["coefv"]
print("\n后验变异系数")
print(f"  频率    : {cv['f'][0]:.6f}")
print(f"  阻尼比  : {cv['z'][0]:.6f}")
print(f"  模态力谱: {cv['Sii'][0]:.6f}")
print(f"  噪声谱  : {cv['Se'][0]:.6f}")
print(f"  振型    : {cv['phi'][0]:.6f}")

law = bayoma.uncertainty_law_cov(
    f=out["f"][0],
    z=out["z"][0],
    S=out["Sii"][0],
    Se=out["Se"][0],
    kappa=10.0,
    duration=data.shape[0] / fs,
    nch=data.shape[1],
    q=0,
)
print("\n不确定度定律预测（kappa = 10）")
print(f"  频率    : {law['f']:.6f}")
print(f"  阻尼比  : {law['z']:.6f}")
print(f"  振型    : {law['phi']:.6f}")
