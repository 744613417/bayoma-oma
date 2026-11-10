"""示例：结果分组与振型可视化。

生成一段三通道合成记录，识别 1.0 Hz 附近的一阶模态，给出频带对应、密集模态区分度、
联合后验协方差与索引、逐阶振型不确定性、PSD/SV 谱与振型空间图。运行：

    E:\\anaconda3\\python.exe examples\\py_results.py

图件写在 examples\\figs\\ 下。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import bayoma  # noqa: E402

f_true = np.array([1.0])
z_true = np.array([0.01])
phi_true = np.array([[1.0], [2.0], [2.0]]) / 3.0
fs = 100.0

data = bayoma.simulate_oma_data(
    freqs_hz=f_true,
    damping=z_true,
    phi=phi_true,
    S=np.array([[1.0]]),
    Se=1.0,
    fs=fs,
    duration=2400.0,
    seed=7,
)

# 三个通道沿竖向等间距布置，方向同为水平 x，相邻通道连线
geometry = {
    "channels": [
        {"name": "ch1", "position": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch2", "position": [0.0, 0.0, 0.6], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch3", "position": [0.0, 0.0, 1.2], "direction": [1.0, 0.0, 0.0]},
    ],
    "connections": [[0, 1], [1, 2]],
}

figure_dir = Path(__file__).resolve().parent / "figs"
out = bayoma.bayoma(
    data,
    fs,
    f1f2=[[0.9, 1.1]],
    f0=[[1.0]],
    mute=1,
    geometry=geometry,
    figure_dir=str(figure_dir),
    spectra=True,
)

rec = out["recognition"]

print("频带与阶次对应")
for band in rec["bands"]:
    print(
        f"  频带 {band['band_index']}: {band['lower']:.3f}–{band['upper']:.3f} Hz，"
        f"带宽 {band['bandwidth']:.3f} Hz，模态数 {band['n_modes']}，f0 = {band['f0']}"
    )
for mode in rec["modes"]:
    print(
        f"  第 {mode['order']} 阶: {mode['frequency']:.6f} Hz，"
        f"属频带 {mode['band_index']} 内第 {mode['order_in_band']} 阶，f0 = {mode['f0']}"
    )

sep = rec["separation"]
print("\n密集模态区分度")
print(f"  判据: {sep['criterion']}")
print(f"  阈值: {sep['threshold']}，存在密集模态对: {sep['has_dense_modes']}")
for pair in sep["pairs"]:
    print(
        f"  阶对 {pair['modes']}: df = {pair['df']:.6f} Hz，"
        f"separation_in_bw = {pair['separation_in_bw']:.3f}，MAC = {pair['mac']:.6f}"
    )

joint = rec["joint_covariance"]
print("\n联合后验协方差")
print(f"  矩阵维数: {joint['size']}，结构: {joint['structure']}")
for item in joint["index"]:
    print(f"  行 {item['index']:2d}: {item['name']:<18s} 类别 {item['kind']}")
print(f"  矩阵: {np.asarray(rec['joint_covariance_matrix']).shape}")

print("\n逐阶振型不确定性")
for shape in rec["shape_uncertainty"]:
    std = np.asarray(shape["std_per_channel"])
    print(f"  第 {shape['order']} 阶: coefv = {shape['coefv']:.6f}，逐通道标准差 {std}")

spec = rec["spectra"]
print("\nPSD 谱与 SV 谱")
print(f"  频率点: 全 {spec['n_freq_full']} 点，抽稀后 {spec['n_freq_kept']} 点（{spec['reduction']}）")
for band in spec["bands"]:
    print(
        f"  频带 {band['band_index']}: {band['lower']:.3f}–{band['upper']:.3f} Hz，"
        f"f0 = {band['f0']}，峰频 {band['peak_freq']:.6f} Hz"
    )

print("\n振型空间可视化")
print(f"  可用: {rec['spatial']['available']}")
if rec["spatial"]["available"]:
    for name in rec["spatial"]["figures"]:
        print(f"  {name}")
else:
    print(f"  原因: {rec['spatial']['reason']}")

mac = bayoma.mac_matrix(out["phi"], phi_true)
print(f"\n振型与真值的 MAC: {mac[0, 0]:.6f}")
