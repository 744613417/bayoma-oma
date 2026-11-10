"""核对异步数据的 JSON 接口端到端可用。"""
import json
import os
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bayoma import io_json, simulate_asynchronous

FS = 100.0
DUR = 4800.0
U = np.array([1.0, 2.0, 1.5, 0.8])
U = U / np.linalg.norm(U)
GROUPS = [[0, 1], [2, 3]]
V = np.array([[1.0, 0.5 * np.exp(1j * np.pi / 4)], [0.5 * np.exp(-1j * np.pi / 4), 1.0]])
SE = [1.0, 1.5]

data = simulate_asynchronous(
    f=1.0, z=0.01, S=1.0, Se_list=SE, u=U, v=V, groups=GROUPS,
    fs=FS, duration=DUR, seed=4,
)

work = os.path.join(tempfile.gettempdir(), "bayoma_async_io")
os.makedirs(work, exist_ok=True)
csv = os.path.join(work, "record.csv")
np.savetxt(csv, data, delimiter=",", fmt="%.10g")

job = os.path.join(work, "job.json")
result = os.path.join(work, "result.json")
spec = {
    "asynchronous": {
        "data": {"path": csv.replace("\\", "/"), "format": "csv", "fs": FS},
        "groups": GROUPS,
        "bands": [[0.9, 1.1]],
        "f0": [1.0],
        "q": 0,
    },
    "output": {"path": result.replace("\\", "/")},
}
with open(job, "w", encoding="utf-8") as fh:
    json.dump(spec, fh, ensure_ascii=False, indent=2)

out = io_json.run_file(job)
print("结果键:", sorted(k for k in out if k not in ("elapsed",)))

ok = True


def check(label, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + label + ("  " + detail if detail else ""))
    if not cond:
        ok = False


check("f 接近真值", abs(out["f"][0] - 1.0) < 0.02, "f=%.6f" % out["f"][0])
check("z 接近真值", abs(out["z"][0] - 0.01) < 0.002, "z=%.6f" % out["z"][0])
check("Se 逐组给出", len(out["Se"]) == 2, str(np.round(out["Se"], 5)))
check("coefv 含 v", "v" in out["coefv"], "coefv keys=" + ",".join(sorted(out["coefv"])))
check("相干矩阵已序列化", "coherence" in out and "re" in out["coherence"], str(out.get("groups")))
check("结果文件已写", os.path.exists(result))
with open(result, encoding="utf-8") as fh:
    back = json.load(fh)
check("结果文件可被读回", abs(back["f"][0] - out["f"][0]) < 1e-12)

print()
print("异步 JSON 接口：" + ("全部通过" if ok else "存在失败项"))
sys.exit(0 if ok else 1)
