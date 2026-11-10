"""示例：多测组识别。

四个测组覆盖六自由度结构的不同测点子集，各测组的频率、阻尼比、模态力谱与噪声
水平均不同，全局振型共用。识别后比较全局振型与真值。
运行：

    python examples\\py_multi_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import bayoma  # noqa: E402
from bayoma import SetupData  # noqa: E402

# 六自由度结构的一阶模态，振型取自课本例 14.2
u_true = np.array([0.162, 0.286, 0.315, 0.477, 0.546, 0.516])
u_true = (u_true / np.linalg.norm(u_true)).reshape(-1, 1)

fs = 100.0
duration = 2400.0
f_true = 3.0
z_true = 0.01

# 四个测组：测点子集、噪声谱、模态力谱各不相同
setups_spec = [
    {"dofs": [0, 1, 2], "Se": 1.0, "S": 1.0},
    {"dofs": [2, 3, 4], "Se": 1.2, "S": 0.8},
    {"dofs": [3, 4], "Se": 0.9, "S": 1.1},
    {"dofs": [4, 5], "Se": 1.0, "S": 1.0},
]

records = []
for spec in setups_spec:
    local = u_true[spec["dofs"], :]          # 局部振型取全局振型的对应分量，不单独归一
    data = bayoma.simulate_oma_data(
        freqs_hz=[f_true],
        damping=[z_true],
        phi=local,
        S=np.array([[spec["S"]]]),
        Se=spec["Se"],
        fs=fs,
        duration=duration,
        seed=11,
    )
    records.append(data)
    print(f"测组 {len(records)}：自由度 {spec['dofs']}，{data.shape[1]} 通道")

setups = []
for spec, data in zip(setups_spec, records):
    sp = bayoma.scaled_fft(data, fs, (2.9, 3.1))
    setups.append(SetupData(freqs=sp.freqs, fft=sp.fft, dofs=spec["dofs"]))

res = bayoma.mpv_multi_setup(setups, n_dof=6, q=0)

print("\n各测组的谱参数")
for i, (freq, damp, noise, force) in enumerate(zip(res.f, res.z, res.Se, np.real(res.S)), start=1):
    print(f"  测组 {i}：f = {freq:.6f} Hz，z = {damp:.6f}，Se = {noise:.6f}，S = {force:.6f}")

mac = bayoma.mac_matrix(res.phi, u_true)
print("\n全局振型")
print(f"  估计 : {np.round(res.phi.ravel(), 6)}")
print(f"  真值 : {np.round(u_true.ravel(), 6)}")
print(f"  MAC  : {mac[0, 0]:.6f}")
print(f"  迭代 : {res.n_iter} 次，收敛 {res.converged}")

weights = bayoma.asymptotic_weights(res, setups, q=0)
print("\n全局最小二乘的渐近权重（式 14.50）")
print(f"  {np.round(weights, 6)}")
