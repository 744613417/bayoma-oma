"""生成 Python 侧的多模态合成算例并保存为 .mat，供 MATLAB 侧交叉校核。

同一份缩放 FFT 分别送入两套估计器，若结果一致，则差异来自各自的数据合成环节
而非估计器实现。

运行：

    python python\\tests\\gen_cross_case.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bayoma import data, multi_mode, synth  # noqa: E402

fs = 100.0
duration = 2400.0
f_true = np.array([1.0, 1.02])
z_true = np.array([0.01, 0.01])
phi_true = np.array([[1.0, -2.0], [2.0, 1.0], [2.0, 2.0]]) / 3.0
phi_true = phi_true / np.linalg.norm(phi_true, axis=0, keepdims=True)
coh = 0.5 * np.exp(1j * np.pi / 4.0)
S_true = np.array([[1.0, coh], [np.conj(coh), 1.0]])
Se = 1.0

tdata = synth.simulate_oma_data(f_true, z_true, phi_true, S_true, Se, fs, duration, seed=3)
spec = data.scaled_fft(tdata, fs, (0.9, 1.1))

res = multi_mode.mpv(spec.freqs, spec.fft, 2, q=0, f0=f_true, max_iter=40)
mac = np.abs(np.sum(np.conj(res.phi) * phi_true, axis=0))

print("PYTHON-OWN f   =", np.round(res.f, 6))
print("PYTHON-OWN z   =", np.round(res.z, 6))
print("PYTHON-OWN Se  = %.6f" % res.Se)
print("PYTHON-OWN S12 = %.4f%+.4fj" % (res.S[0, 1].real, res.S[0, 1].imag))
print("PYTHON-OWN MAC =", np.round(mac, 6))
print("PYTHON-OWN nllf= %.4f iters=%d" % (res.nllf, res.n_iter))

out = ROOT.parent / "matlab" / "tests" / "py_multi_case.mat"
savemat(
    out,
    {
        "fft": spec.fft,
        "freqs": spec.freqs.reshape(-1, 1),
        "dTotal": spec.d_total,
        "f": f_true.reshape(-1, 1),
        "z": z_true.reshape(-1, 1),
        "phi": phi_true,
        "S": S_true,
        "Se": Se,
        "fs": fs,
        "nt": tdata.shape[0],
    },
)
print(f"saved {out}")
