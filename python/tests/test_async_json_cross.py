"""跨语言核对异步数据的 JSON 接口：同一份描述、同一份数据，两侧结果应一致。

步骤：生成 CSV 与描述文件 → Python 侧跑一遍 → MATLAB 侧跑一遍 → 逐项比对。
"""
import json
import os
import subprocess
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
V = np.array([[1.0, 0.5 * np.exp(1j * np.pi / 4)],
              [0.5 * np.exp(-1j * np.pi / 4), 1.0]])
SE = [1.0, 1.5]

MATLAB = os.environ.get("BAYOMA_MATLAB", "matlab")
PROJECT = str(ROOT / "matlab")

data = simulate_asynchronous(
    f=1.0, z=0.01, S=1.0, Se_list=SE, u=U, v=V, groups=GROUPS,
    fs=FS, duration=DUR, seed=4,
)

work = os.path.join(tempfile.gettempdir(), "bayoma_async_cross")
os.makedirs(work, exist_ok=True)
csv = os.path.join(work, "record.csv")
np.savetxt(csv, data, delimiter=",", fmt="%.10g")

job = os.path.join(work, "job.json")
py_result = os.path.join(work, "py_result.json")
mb_result = os.path.join(work, "mb_result.json")
spec = {
    "asynchronous": {
        "data": {"path": csv.replace("\\", "/"), "format": "csv", "fs": FS},
        "groups": GROUPS,
        "bands": [[0.9, 1.1]],
        "f0": [1.0],
        "q": 0,
    },
    "output": {"path": py_result.replace("\\", "/")},
}
with open(job, "w", encoding="utf-8") as fh:
    json.dump(spec, fh, ensure_ascii=False, indent=2)

py_out = io_json.run_file(job)
print("PY  f=%.8f z=%.8f S=%.8f Se=%s" % (
    py_out["f"][0], py_out["z"][0], py_out["Sii"][0],
    np.round(py_out["Se"], 5)))

cmd = (
    "addpath(genpath('%s')); "
    "r = bayoma.io_json('%s', '%s'); "
    "fprintf('MB  f=%%.8f z=%%.8f S=%%.8f Se=[%%.5f %%.5f]\\n', "
    "r.f(1), r.z(1), r.Sii(1), r.Se(1), r.Se(2));"
) % (PROJECT, job.replace("\\", "/"), mb_result.replace("\\", "/"))
proc = subprocess.run([MATLAB, "-batch", cmd], capture_output=True, text=True,
                      timeout=1800)
for line in (proc.stdout or "").splitlines():
    if line.strip():
        print("mb:", line.strip())
if proc.returncode != 0:
    print("MATLAB 退出码", proc.returncode)
    print((proc.stderr or "")[-2000:])

ok = True


def check(label, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + label + ("  " + detail if detail else ""))
    if not cond:
        ok = False


if not os.path.exists(mb_result):
    check("MATLAB 结果文件已写", False, mb_result)
    sys.exit(1)

with open(mb_result, encoding="utf-8") as fh:
    mb = json.load(fh)


def as_list(value):
    """长度 1 的数组在 MATLAB 的 jsonencode 里写成标量，读取方统一成列表。"""
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(value)]


mb_f = as_list(mb["f"])[0]
mb_z = as_list(mb["z"])[0]
mb_s = as_list(mb["Sii"])[0]
mb_se = as_list(mb["Se"])

check("f 一致", abs(mb_f - py_out["f"][0]) < 1e-6,
      "py=%.8f mb=%.8f" % (py_out["f"][0], mb_f))
check("z 一致", abs(mb_z - py_out["z"][0]) < 1e-5,
      "py=%.8f mb=%.8f" % (py_out["z"][0], mb_z))
check("S 一致", abs(mb_s / py_out["Sii"][0] - 1.0) < 1e-4,
      "py=%.8f mb=%.8f" % (py_out["Sii"][0], mb_s))
for i in (0, 1):
    check("Se[%d] 一致" % (i + 1),
          abs(mb_se[i] / py_out["Se"][i] - 1.0) < 1e-4,
          "py=%.6f mb=%.6f" % (py_out["Se"][i], mb_se[i]))
check("coefv 含 v", "v" in mb["coefv"], ",".join(sorted(mb["coefv"])))
check("coherence 已序列化", "re" in mb.get("coherence", {}),
      str(mb.get("groups")))
groups_back = mb.get("groups", [])
if isinstance(groups_back, dict):
    groups_back = []
elif groups_back and not isinstance(groups_back[0], (list, tuple)):
    groups_back = [groups_back]
check("groups 一致",
      [[int(x) for x in g] for g in groups_back] == GROUPS,
      str(groups_back))

print()
print("异步 JSON 跨语言核对：" + ("全部通过" if ok else "存在失败项"))
sys.exit(0 if ok else 1)
