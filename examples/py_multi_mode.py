"""示例：多模态识别（含密集模态）。

生成课本例 13.2 的两阶密集模态数据（1.00 与 1.02 Hz，模态力相关系数 0.5），
识别后比较频率、阻尼比、模态力谱矩阵与振型。
运行：

    python examples\\py_multi_mode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import bayoma  # noqa: E402

f_true = np.array([1.0, 1.02])
z_true = np.array([0.01, 0.01])
phi_true = np.column_stack([[1.0, 2.0, 2.0], [-2.0, 1.0, 2.0]]) / 3.0
phi_true = phi_true / np.linalg.norm(phi_true, axis=0, keepdims=True)

# 模态力谱矩阵：两阶自谱均为 1，相关系数 0.5、相位 π/4
coh = 0.5 * np.exp(1j * np.pi / 4.0)
S_true = np.array([[1.0, coh], [np.conj(coh), 1.0]])
Se_true = 1.0
fs = 100.0
duration = 9600.0

data = bayoma.simulate_oma_data(
    freqs_hz=f_true,
    damping=z_true,
    phi=phi_true,
    S=S_true,
    Se=Se_true,
    fs=fs,
    duration=duration,
    seed=3,
)
print(f"数据：{data.shape[0]} 样本、{data.shape[1]} 通道、{duration:.0f} s")

out = bayoma.bayoma(
    data,
    fs,
    f1f2=[[0.9, 1.1]],
    f0=[[1.0, 1.02]],
    maxiter=40,
    mute=1,
)

print("\n识别结果")
print(f"  频率    : {np.round(out['f'], 6)}（真值 {f_true}）")
print(f"  阻尼比  : {np.round(out['z'], 6)}（真值 {z_true}）")
print(f"  噪声谱  : {out['Se'][0]:.6f}（真值 {Se_true:.6f}）")

S_est = np.atleast_2d(out["S"][0])
print("\n模态力谱矩阵")
print(f"  估计\n{np.round(S_est, 6)}")
print(f"  真值\n{S_true}")
print(f"  互谱模：估计 {abs(S_est[0, 1]):.6f}，真值 {abs(coh):.6f}")

mac = bayoma.mac_matrix(out["phi"], phi_true)
print("\n振型 MAC 矩阵")
print(np.round(mac, 6))

cv = out["coefv"]
print("\n后验变异系数")
print(f"  频率    : {np.round(cv['f'], 6)}")
print(f"  阻尼比  : {np.round(cv['z'], 6)}")
print(f"  模态力谱: {np.round(cv['Sii'], 6)}")
print(f"  噪声谱  : {np.round(cv['Se'], 6)}")
print(f"  振型    : {np.round(cv['phi'], 6)}")

print("\n说明：密集模态且相干较弱时，模态力谱矩阵的非对角元方差很大，")
print("      记录加长后收敛，短记录下不宜与真值逐项比较（见 docs/VERIFICATION.md）。")
