"""校核 MATLAB 侧不确定度定律与 Python 侧逐项一致。

输出一行一个参数，格式：参数 Python MATLAB 相对差。
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bayoma

MATLAB = os.environ.get("BAYOMA_MATLAB", "matlab")
PROJ = str(ROOT)

CASES = [
    (1.0, 0.01, 1.0, 1.0, 10.0, 2400.0, 3, 0),
    (1.0, 0.01, 1.0, 1.0, 10.0, 600.0, 3, 0),
    (3.0, 0.02, 2.5, 0.5, 12.0, 1800.0, 6, 0),
    (5.0, 0.015, 1.0, 1.0, 8.0, 3600.0, 4, 1),
]

rows = []
for case in CASES:
    py = bayoma.uncertainty_law_cov(*case)
    rows.append(py)

code = (
    "addpath(genpath('%s\\matlab'));"
    "cases = %s;"
    "out = zeros(numel(cases),5);"
    "for i=1:numel(cases)"
    " c=cases{i}; r=bayoma.uncertainty_law(c(1),c(2),c(3),c(4),c(5),c(6),c(7),c(8));"
    " out(i,:)=[r.f r.z r.S r.Se r.phi];"
    "end;"
    "fprintf('%%s\\n', jsonencode(out))"
) % (
    PROJ,
    "{"
    + ";".join(
        "[%s]" % ",".join(repr(float(v)) for v in c) for c in CASES
    )
    + "}",
)

proc = subprocess.run(
    [MATLAB, "-batch", code],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)
lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith("[")]
if not lines:
    print(proc.stdout)
    print(proc.stderr)
    raise SystemExit(1)

mat = json.loads(lines[-1])
names = ["f", "z", "S", "Se", "phi"]
ok = True
for i, (py, mv) in enumerate(zip(rows, mat)):
    print("算例 %d：f=%.1f z=%.3f kappa=%.1f T=%.0f nch=%d" % (
        i + 1, CASES[i][0], CASES[i][1], CASES[i][4], CASES[i][5], CASES[i][6]))
    for j, name in enumerate(names):
        a = float(py[name])
        b = float(mv[j])
        rel = abs(a - b) / max(abs(a), 1e-300)
        flag = "OK " if rel < 1e-10 else "BAD"
        if rel >= 1e-10:
            ok = False
        print("  %s %-4s Python=%.10e MATLAB=%.10e rel=%.2e" % (flag, name, a, b, rel))

print()
print("MATLAB 不确定度定律与 Python %s" % ("逐项一致" if ok else "存在不一致"))
sys.exit(0 if ok else 1)
