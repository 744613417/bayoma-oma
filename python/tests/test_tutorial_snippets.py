"""核对 docs/TUTORIAL.md 中给出的调用片段能直接运行。"""
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import bayoma
from bayoma import SetupData, mpv_multi_setup, scaled_fft
from bayoma import synth

FS = 100.0
PHI = np.array([0.162, 0.286, 0.315, 0.477, 0.546, 0.516], dtype=float)
PHI = PHI / np.linalg.norm(PHI)

ok = True


def check(label, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + label + ("  " + detail if detail else ""))
    if not cond:
        ok = False


# ---- 第 1 步：单模态示例片段 ----
data = synth.simulate_oma_data([1.0], [0.01], np.array([[1.0], [2.0], [2.0]]) / 3.0,
                               [[1.0]], 1.0, FS, 600.0, seed=11)
out = bayoma.bayoma(data, FS, f1f2=[[0.9, 1.1]], f0=[[1.0]], mute=1)
check("bayoma() 返回键齐全",
      all(k in out for k in ("f", "z", "Sii", "Se", "phi", "rms", "S", "coefv", "cmat", "sn", "bands", "nch", "fs")))
check("f 落在频带内", 0.9 < out["f"][0] < 1.1, "f=%.6f" % out["f"][0])

# ---- 第 3.1 节：sn 循环 ----
lines = []
for i in range(len(out["f"])):
    lines.append("f=%.4f z=%.4f sn=%.1f" % (out["f"][i], out["z"][i], out["sn"][i]))
check("sn 逐阶可读", len(lines) == 1, lines[0])

# ---- 第 3.2 节：coefv 字典 ----
cv = out["coefv"]
check("coefv 含 f/z/phi/Sii/Se",
      all(k in cv for k in ("f", "z", "phi", "Sii", "Se")),
      "keys=" + ",".join(sorted(cv)))

# ---- 第 3.3 节：定律对照 ----
law = bayoma.uncertainty_law_cov(
    f=out["f"][0], z=out["z"][0], S=out["Sii"][0], Se=out["Se"][0],
    kappa=10.0, duration=data.shape[0] / FS, nch=data.shape[1], q=0,
)
ratio = float(out["coefv"]["z"][0] / law["z"])
check("定律 z 可读且比值在 0.5-2.0", 0.5 < ratio < 2.0, "ratio=%.4f" % ratio)

# ---- 第 3.3 节的第 16 章片段：相关矩阵 ----
mat = bayoma.posterior_matrix(
    f=out["f"][0], z=out["z"][0], S=out["Sii"][0], Se=out["Se"][0],
    kappa=10.0, duration=data.shape[0] / FS, nch=data.shape[1], q=0,
)
czs = float(mat["corr"][1, 2])
check("posterior_matrix 返回相关矩阵", mat["corr"].shape == (4, 4), "rho(z,S)=%.4f" % czs)
check("zeta 与 S 正相关且绝对值在 0.2-0.9",
      0.2 < czs < 0.9, "rho(z,S)=%.4f" % czs)

# ---- 第 3.4 节：optimal_duration ----
need = bayoma.optimal_duration(f=1.0, z=0.01, S=1.0, Se=1.0, kappa=10.0, nch=3, target=1.0e-3)
check("optimal_duration 返回正有限值", np.isfinite(need) and need > 0.0, "%.0f s" % need)

# ---- 第 4 步：多测组片段（dofs 0 起计） ----
setups = []
specs = (([0, 1, 2], 1.0, 1.0), ([2, 3, 4], 1.2, 0.8), ([3, 4], 0.9, 1.1), ([4, 5], 1.0, 1.0))
for dofs, se, s in specs:
    sub = PHI[dofs]
    arr = synth.simulate_oma_data([3.0], [0.01], sub.reshape(-1, 1), [[s]], se,
                                  FS, 1200.0, seed=100 + len(setups))
    sp = scaled_fft(arr, FS, (2.9, 3.1))
    setups.append(SetupData(freqs=sp.freqs, fft=sp.fft, dofs=dofs))
res = mpv_multi_setup(setups, n_dof=6)
mac = bayoma.mac_matrix(res.phi.ravel(), PHI)
check("多测组全局振型 MAC > 0.99", float(np.real(mac[0, 0])) > 0.99, "MAC=%.6f" % float(np.real(mac[0, 0])))
check("多测组逐组 f 四舍五入到 3.0", all(abs(float(v) - 3.0) < 0.01 for v in np.ravel(res.f)),
      str(np.round(np.ravel(res.f), 4)))

# ---- 第 5 步：JSON 片段 ----
work = os.path.join(tempfile.gettempdir(), "bayoma_tutorial_check")
os.makedirs(work, exist_ok=True)
csv_path = os.path.join(work, "record.csv")
np.savetxt(csv_path, data, delimiter=",", fmt="%.10g")
job = os.path.join(work, "job.json")
result = os.path.join(work, "result.json")
with open(job, "w", encoding="utf-8") as fh:
    fh.write('{\n'
             '  "data": {"path": "%s", "format": "csv", "fs": 100.0},\n'
             '  "bands": [[0.9, 1.1]],\n'
             '  "f0": [[1.0]],\n'
             '  "options": {"q": 0, "mute": 1},\n'
             '  "output": {"path": "%s"}\n'
             '}\n' % (csv_path.replace("\\", "/"), result.replace("\\", "/")))
res_dict = bayoma.io_json.run_file(job)
check("JSON 结果键齐全", all(k in res_dict for k in ("f", "z", "coefv", "bands", "n_modes")))
check("JSON 结果文件已写", os.path.exists(result))
d = (res_dict["bands"][0] if isinstance(res_dict["bands"], list) else [])
check("JSON 频带为二维", isinstance(res_dict["bands"], list) and len(res_dict["bands"]) >= 1,
      str(res_dict.get("bands")))

print()
print("TUTORIAL 片段核对：" + ("全部通过" if ok else "存在失败项"))
sys.exit(0 if ok else 1)
