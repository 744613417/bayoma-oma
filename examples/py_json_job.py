"""示例：JSON 接口端到端。

写一份描述文件，按描述执行识别，读回结果文件。演示其余软件与智能体调用本工程的
方式。
运行：

    python examples\\py_json_job.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import bayoma  # noqa: E402

work = ROOT / "examples" / "_json_demo"
work.mkdir(parents=True, exist_ok=True)

# 生成一段合成记录并写成文本文件
data = bayoma.simulate_oma_data(
    freqs_hz=[1.0],
    damping=[0.01],
    phi=np.array([[1.0], [2.0], [2.0]]) / 3.0,
    S=np.array([[1.0]]),
    Se=1.0,
    fs=100.0,
    duration=1200.0,
    seed=7,
)
data_path = work / "record.csv"
np.savetxt(data_path, data, delimiter=",")
print(f"数据文件：{data_path}（{data.shape[0]} 行、{data.shape[1]} 列）")

spec = {
    "data": {"path": str(data_path), "format": "csv", "fs": 100.0},
    "bands": [[0.9, 1.1]],
    "f0": [[1.0]],
    "options": {"q": 0, "mute": 1},
    "output": {"path": str(work / "result.json")},
}
spec_path = work / "job.json"
spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"描述文件：{spec_path}")

result = bayoma.io_json.run_file(spec_path)
print("\n结果")
print(f"  模态数  : {result['n_modes']}")
print(f"  频率    : {result['f']}")
print(f"  阻尼比  : {result['z']}")
print(f"  模态力谱: {result['Sii']}")
print(f"  噪声谱  : {result['Se']}")
print(f"  频带    : {result['bands']}")
print(f"  输入信息: {result['input']}")
print(f"  变异系数: f = {result['coefv']['f']}, z = {result['coefv']['z']}")

print("\n结果文件片段")
print((work / "result.json").read_text(encoding="utf-8")[:400] + " …")

print("\n命令行调用方式")
print(f"  python -m bayoma.io_json {spec_path}")
