"""快速自检：单模态、多模态、多测组、自动选带、机器可读接口、驻点条件。

运行：

    python python\\tests\\test_smoke.py

全部通过时返回码为 0。记录长度取 1200–2400 s，单次运行约 2 分钟。
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bayoma  # noqa: E402
from bayoma import SetupData  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def test_single_mode() -> None:
    phi = (np.array([1.0, 2.0, 2.0]) / 3.0).reshape(-1, 1)
    data = bayoma.simulate_oma_data(
        freqs_hz=[1.0], damping=[0.01], phi=phi, S=np.array([[1.0]]), Se=1.0,
        fs=100.0, duration=2400.0, seed=7,
    )
    out = bayoma.bayoma(data, 100.0, f1f2=[[0.9, 1.1]], f0=[[1.0]], mute=1)
    f = float(np.ravel(out["f"])[0])
    z = float(np.ravel(out["z"])[0])
    mac = float(bayoma.mac_matrix(np.asarray(out["phi"]), phi)[0, 0])
    check("单模态频率", abs(f - 1.0) < 0.005, f"f = {f:.6f}")
    check("单模态阻尼比", abs(z - 0.01) < 0.002, f"z = {z:.6f}")
    check("单模态振型", mac > 0.999, f"MAC = {mac:.6f}")
    cv = out["coefv"]
    check("单模态后验变异系数", 0.0 < float(cv["f"][0]) < 0.01, f"coefv.f = {float(cv['f'][0]):.6f}")


def test_multi_mode() -> None:
    phi = np.column_stack([
        np.array([1.0, 2.0, 2.0]) / 3.0,
        np.array([-2.0, 1.0, 2.0]) / 3.0,
    ])
    coh = 0.5 * np.exp(1j * np.pi / 4.0)
    S = np.array([[1.0, coh], [np.conj(coh), 1.0]])
    data = bayoma.simulate_oma_data(
        freqs_hz=[1.0, 1.02], damping=[0.01, 0.01], phi=phi, S=S, Se=1.0,
        fs=100.0, duration=2400.0, seed=3,
    )
    out = bayoma.bayoma(data, 100.0, f1f2=[[0.9, 1.1]], f0=[[1.0, 1.02]], maxiter=40, mute=1)
    f = np.ravel(out["f"])
    mac = np.asarray(bayoma.mac_matrix(np.asarray(out["phi"]), phi))
    check("多模态频率", np.all(np.abs(f - np.array([1.0, 1.02])) < 0.01), f"f = {np.round(f, 6)}")
    check("多模态振型", np.min(np.diag(mac)) > 0.99, f"MAC = {np.round(np.diag(mac), 6)}")


def test_multi_setup() -> None:
    u = np.array([0.162, 0.286, 0.315, 0.477, 0.546, 0.516])
    u = (u / np.linalg.norm(u)).reshape(-1, 1)
    specs = [
        {"dofs": [0, 1, 2], "Se": 1.0, "S": 1.0},
        {"dofs": [2, 3, 4], "Se": 1.2, "S": 0.8},
        {"dofs": [3, 4], "Se": 0.9, "S": 1.1},
        {"dofs": [4, 5], "Se": 1.0, "S": 1.0},
    ]
    setups = []
    for spec in specs:
        local = u[spec["dofs"], :]
        data = bayoma.simulate_oma_data(
            freqs_hz=[3.0], damping=[0.01], phi=local, S=np.array([[spec["S"]]]),
            Se=spec["Se"], fs=100.0, duration=1200.0, seed=11,
        )
        sp = bayoma.scaled_fft(data, 100.0, (2.9, 3.1))
        setups.append(SetupData(freqs=sp.freqs, fft=sp.fft, dofs=spec["dofs"]))
    res = bayoma.mpv_multi_setup(setups, n_dof=6, q=0)
    mac = float(bayoma.mac_matrix(res.phi, u)[0, 0])
    check("多测组全局振型", mac > 0.999, f"MAC = {mac:.6f}")
    check("多测组收敛", bool(res.converged), f"niter = {res.n_iter}")
    for r, spec in enumerate(specs):
        check(
            f"多测组第 {r + 1} 组噪声谱",
            abs(float(res.Se[r]) - spec["Se"]) < 0.2,
            f"Se = {float(res.Se[r]):.4f}，真值 {spec['Se']}",
        )


def test_select_bands() -> None:
    phi = (np.array([1.0, 2.0, 2.0]) / 3.0).reshape(-1, 1)
    data = bayoma.simulate_oma_data(
        freqs_hz=[1.0, 1.4], damping=[0.01, 0.01], phi=phi,
        S=np.eye(2), Se=1.0, fs=100.0, duration=2400.0, seed=5,
    )
    out = bayoma.bayoma(data, 100.0, mute=1)
    f = np.ravel(out["f"])
    check("自动选带识别出两阶", f.size == 2, f"f = {np.round(f, 6)}")
    check(
        "自动选带频率误差",
        np.all(np.abs(f - np.array([1.0, 1.4])) < 0.02),
        f"f = {np.round(f, 6)}",
    )


def test_json_interface() -> None:
    phi = (np.array([1.0, 2.0, 2.0]) / 3.0).reshape(-1, 1)
    data = bayoma.simulate_oma_data(
        freqs_hz=[1.0], damping=[0.01], phi=phi, S=np.array([[1.0]]), Se=1.0,
        fs=100.0, duration=1200.0, seed=7,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        np.savetxt(tmpdir / "record.csv", data, delimiter=",")
        spec = {
            "data": {"path": str(tmpdir / "record.csv"), "format": "csv", "fs": 100.0},
            "bands": [[0.9, 1.1]],
            "f0": [[1.0]],
            "options": {"mute": 1},
            "output": {"path": str(tmpdir / "result.json")},
        }
        spec_path = tmpdir / "job.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        result = bayoma.io_json.run_file(spec_path)
        ok = (tmpdir / "result.json").exists()
        check("JSON 接口写出结果文件", ok)
        check("JSON 接口频率", abs(float(np.ravel(result["f"])[0]) - 1.0) < 0.005)
        text = (tmpdir / "result.json").read_text(encoding="utf-8")
        check("JSON 结果可被读回", json.loads(text)["n_modes"] == 1)


def test_scale_invariance() -> None:
    """数据的物理单位改变时，无量纲结果不应改变。

    实测记录的幅值以 g 计时，谱密度量级可低至 1e-11；后验协方差的伪逆若不做尺度
    归一，谱参数的曲率会被判成零，对应参数的变异系数退化为零。
    """
    phi = (np.array([1.0, 2.0, 2.0]) / 3.0).reshape(-1, 1)
    base = bayoma.simulate_oma_data(
        freqs_hz=[1.0], damping=[0.01], phi=phi, S=np.array([[1.0]]), Se=1.0,
        fs=100.0, duration=1200.0, seed=11,
    )
    ref = bayoma.bayoma(base, 100.0, f1f2=[[0.9, 1.1]], f0=[[1.0]], mute=1)

    scale = 1.0e-5
    out = bayoma.bayoma(base * scale, 100.0, f1f2=[[0.9, 1.1]], f0=[[1.0]], mute=1)

    # 优化器终止点受浮点舍入影响，最可能值在 1e-7 相对量级上有抖动；这里按 1e-4 判定，
    # 足以捕获「谱参数曲率被判成零、变异系数退化为零」这一失效模式。
    tol = 1.0e-4
    check("单位缩放后频率不变",
          abs(float(out["f"][0]) - float(ref["f"][0])) / float(ref["f"][0]) < tol,
          f"f = {float(out['f'][0]):.9f}，基准 {float(ref['f'][0]):.9f}")
    check("单位缩放后阻尼比不变",
          abs(float(out["z"][0]) - float(ref["z"][0])) / float(ref["z"][0]) < tol,
          f"z = {float(out['z'][0]):.8f}，基准 {float(ref['z'][0]):.8f}")
    for key in ("f", "z", "phi"):
        a = float(np.ravel(ref["coefv"][key])[0])
        b = float(np.ravel(out["coefv"][key])[0])
        check(f"单位缩放后 {key} 的变异系数不变", a > 0.0 and abs(a - b) / a < tol,
              f"{b:.6g}，基准 {a:.6g}")


def test_stationarity() -> None:
    """最可能值处负对数似然的尺度化梯度应为零。

    多模态与多测组都用交替极小化加联合精化求解最可能值。交替极小化在最优点
    附近可能小幅摆动，若过早停止，点估计仍偏离驻点。本项用中心差分逐方向算
    梯度，并乘以参数值得到度量「参数改动一个相对单位时负对数似然的变化」；
    该量相对负对数似然本身应在 1e-5 以下，否则视为未到达驻点。
    """
    from bayoma.multi_mode import _project_fft
    from bayoma.posterior import tangent_basis

    phi = np.array([[1.0, -2.0], [2.0, 1.0], [2.0, 2.0]])
    phi = phi / np.linalg.norm(phi, axis=0)
    data = bayoma.simulate_oma_data(
        np.array([1.0, 1.4]), np.array([0.01, 0.012]), phi, np.eye(2), 1.0, 100.0, 1200.0, seed=11
    )
    spec = bayoma.scaled_fft(data, 100.0, [0.9, 1.5])
    res = bayoma.mpv_multi(spec.freqs, spec.fft, 2, q=0, f0=np.array([1.0, 1.4]), max_iter=40)

    fproj = _project_fft(spec.fft, res.basis)
    d_total = float(np.sum(np.abs(spec.fft) ** 2))
    nch = spec.fft.shape[0]
    f_hat = np.asarray(res.f, dtype=float)
    z_hat = np.asarray(res.z, dtype=float)
    s_hat = np.asarray(res.S, dtype=complex)
    se_hat = float(res.Se)
    alpha = np.asarray(res.alpha, dtype=complex)

    def total(fv, zv, sv, sev, al):
        return float(
            bayoma.multi_mode.nllf_multi(fv, zv, sv, sev, al, spec.freqs, fproj, d_total, 0, nch)
        )

    base = total(f_hat, z_hat, s_hat, se_hat, alpha)
    worst = 0.0
    for i in range(2):
        for name, arr, step in (("f", f_hat, 1e-7), ("z", z_hat, 1e-8)):
            def make(x, arr=arr, name=name):
                out = arr.copy()
                out[i] = x
                return (out, z_hat, s_hat, se_hat, alpha) if name == "f" else (
                    f_hat, out, s_hat, se_hat, alpha
                )

            hi = total(*make(arr[i] + step))
            lo = total(*make(arr[i] - step))
            g = (hi - lo) / (2.0 * step) * max(abs(arr[i]), 1e-12)
            worst = max(worst, abs(g))
        zb = tangent_basis(alpha[:, i])
        for j in range(zb.shape[1]):
            direction = np.zeros_like(alpha)
            direction[:, i] = zb[:, j]
            eps = 1e-7
            hi = total(f_hat, z_hat, s_hat, se_hat, alpha + eps * direction)
            lo = total(f_hat, z_hat, s_hat, se_hat, alpha - eps * direction)
            worst = max(worst, abs((hi - lo) / (2.0 * eps)))

    ratio = worst / max(abs(base), 1.0)
    check("最可能值处负对数似然的尺度化梯度接近零", ratio < 1.0e-5,
          f"最大梯度 {worst:.3e}，占负对数似然 {base:.4f} 的 {ratio:.2e}")


def main() -> int:
    print(f"bayoma {bayoma.__version__}\n")
    for fn in (
        test_single_mode,
        test_multi_mode,
        test_multi_setup,
        test_select_bands,
        test_json_interface,
        test_scale_invariance,
        test_stationarity,
    ):
        print(f"--- {fn.__name__} ---")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {fn.__name__} 抛出异常：{type(exc).__name__}: {exc}")
            failures.append(fn.__name__)
        print()
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
