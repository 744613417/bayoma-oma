"""校验新增结果分组：频带、区分度、联合协方差、振型不确定性、谱、可视化、组装。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import bayoma


GEOMETRY = {
    "channels": [
        {"name": "ch0", "position": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch1", "position": [0.0, 0.0, 3.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch2", "position": [0.0, 0.0, 6.0], "direction": [1.0, 0.0, 0.0]},
    ],
    "connections": [[0, 1], [1, 2]],
}

GEOMETRY4 = {
    "channels": [
        {"name": "ch0", "position": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch1", "position": [0.0, 0.0, 2.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch2", "position": [0.0, 0.0, 4.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch3", "position": [0.0, 0.0, 6.0], "direction": [1.0, 0.0, 0.0]},
    ],
    "connections": [[0, 1], [1, 2], [2, 3]],
}

GEOMETRY5 = {
    "channels": [
        {"name": "ch0", "position": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch1", "position": [0.0, 0.0, 2.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch2", "position": [0.0, 0.0, 4.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch3", "position": [0.0, 0.0, 6.0], "direction": [1.0, 0.0, 0.0]},
        {"name": "ch4", "position": [0.0, 0.0, 8.0], "direction": [1.0, 0.0, 0.0]},
    ],
    "connections": [[0, 1], [1, 2], [2, 3], [3, 4]],
}


def _data(seed=7, duration=1200.0):
    phi = np.array([[1.0], [2.0], [2.0]]) / 3.0
    return bayoma.simulate_oma_data(
        freqs_hz=[1.0],
        damping=[0.01],
        phi=phi,
        S=np.array([[1.0]]),
        Se=1.0,
        fs=100.0,
        duration=duration,
        seed=seed,
    )


def _test_sync() -> list[str]:
    """同步识别：频带、区分度、联合协方差、振型不确定性、谱、可视化。"""
    failures: list[str] = []

    def check(label, ok, detail=""):
        print(("[OK] " if ok else "[FAIL] ") + label + ("  " + str(detail) if detail else ""))
        if not ok:
            failures.append(label)

    tdata = _data()
    with tempfile.TemporaryDirectory() as tmp:
        out = bayoma.bayoma(
            tdata,
            100.0,
            [[0.9, 1.1]],
            [[1.0]],
            mute=1,
            geometry=GEOMETRY,
            figure_dir=tmp,
            spectra=True,
        )
        rec = out["recognition"]

        check("recognition 存在", isinstance(rec, dict))
        check("频带表一项", len(rec["bands"]) == 1, rec["bands"])
        b0 = rec["bands"][0]
        check("频带上下界", abs(b0["lower"] - 0.9) < 1e-12 and abs(b0["upper"] - 1.1) < 1e-12)
        check("频带内模态数", b0["n_modes"] == 1, b0["n_modes"])
        check("频带内频率初值", len(b0["f0"]) == 1 and b0["f0"][0] == 1.0, b0["f0"])

        check("阶次映射", rec["modes"][0]["band_index"] == 0 and rec["modes"][0]["order"] == 0)

        sep = rec["separation"]
        check("区分度单阶无对", sep["pairs"] == [] and sep["has_dense_modes"] is False)

        jc = rec["joint_covariance"]
        nch = out["nch"]
        check("联合协方差维数", jc["size"] == 4 + nch, jc["size"])
        check("联合协方差索引数", len(jc["index"]) == 4 + nch)
        names = [item["name"] for item in jc["index"]]
        check("索引首项为 f[0]", names[0] == "f[0]", names[:4])
        check("索引含 Se", "Se" in names)
        check("索引含 phi 分量", any(n.startswith("phi[0]") for n in names))
        mat = rec["joint_covariance_matrix"]
        check("联合协方差对称", float(np.max(np.abs(mat - mat.T))) < 1e-12)
        check("联合协方差正对角", bool(np.all(np.diag(mat) >= 0.0)))

        su = rec["shape_uncertainty"][0]
        check("振型不确定性阶次", su["order"] == 0)
        check("逐通道标准差长度", len(su["std_per_channel"]) == nch)
        check("振型 c.o.v. 为正", su["coefv"] is not None and su["coefv"] > 0.0, su["coefv"])

        sp = rec["spectra"]
        check("谱存在", sp is not None)
        check("谱频点数不超过上限", sp["n_freq_kept"] <= 1024, sp["n_freq_kept"])
        check("谱与 SV 行数一致", len(sp["sv"]) == len(sp["psd_diag"]))
        check("谱频带标注", sp["bands"][0]["f0"] == [1.0])
        check("谱标注频点数", sp["bands"][0]["n_freq_points"] > 0)

        spa = rec["spatial"]
        check("可视化可用", spa["available"] is True, spa.get("reason"))
        fig = Path(spa["figures"][0]["path"])
        check("图片已落盘", fig.exists() and fig.stat().st_size > 1000, fig.name)

        csv_path = Path(tmp) / "spectra.csv"
        from bayoma import results as R

        sp_full = R.spectra_table(tdata, 100.0, np.array([[0.9, 1.1]]), [[1.0]])
        written = R.save_spectra_csv(sp_full, csv_path)
        check("谱 CSV 已写", written.exists() and written.stat().st_size > 100)

        # 无几何时不应报错，spatial 为 None
        out2 = bayoma.bayoma(tdata, 100.0, [[0.9, 1.1]], [[1.0]], mute=1)
        check("无几何时 spatial 为空", out2["recognition"]["spatial"] is None)
        check("无谱时 spectra 为空", out2["recognition"]["spectra"] is None)

    return failures


def _test_dense_pair() -> list[str]:
    """密集模态：两阶落在同一频带时区分度表应给出判据与相关系数。"""
    failures: list[str] = []

    def check(label, ok, detail=""):
        print(("[OK] " if ok else "[FAIL] ") + label + ("  " + str(detail) if detail else ""))
        if not ok:
            failures.append(label)

    phi = np.array([[1.0, -2.0], [2.0, 1.0], [2.0, 2.0]], dtype=float)
    phi = phi / np.linalg.norm(phi, axis=0, keepdims=True)
    coh = 0.5 * np.exp(1j * np.pi / 4)
    smat = np.array([[1.0, coh], [np.conj(coh), 1.0]])
    tdata = bayoma.simulate_oma_data(
        freqs_hz=[1.0, 1.02],
        damping=[0.01, 0.01],
        phi=phi,
        S=smat,
        Se=1.0,
        fs=100.0,
        duration=1200.0,
        seed=11,
    )
    out = bayoma.bayoma(tdata, 100.0, [[0.9, 1.1]], [[1.0, 1.02]], mute=1)
    rec = out["recognition"]
    sep = rec["separation"]
    check("密集模态有一对", len(sep["pairs"]) == 1, sep["pairs"])
    check("判为密集", sep["has_dense_modes"] is True, sep["dense_pairs"])
    pair = sep["pairs"][0]
    check("给出频率间隔", pair["df"] is not None and pair["df"] > 0.0, pair["df"])
    check("给出半功率带宽", len(pair["half_power_bw"]) == 2)
    check("给出以带宽计的间隔", pair["separation_in_bw"] is not None, pair["separation_in_bw"])
    check("给出频率相关系数", pair.get("f_correlation") is not None, pair.get("f_correlation"))
    check("给出振型 MAC", pair["mac"] is not None, pair["mac"])
    check("联合协方差含两阶", rec["joint_covariance"]["size"] > 4 + out["nch"])
    return failures


def _test_async_assembly() -> list[str]:
    """异步单阶的组装结果：索引与协方差对应、不确定性分解、可视化。"""
    failures: list[str] = []

    def check(label, ok, detail=""):
        print(("[OK] " if ok else "[FAIL] ") + label + ("  " + str(detail) if detail else ""))
        if not ok:
            failures.append(label)

    from bayoma import asynchronous as async_mod
    from bayoma import io_json

    fs = 100.0
    duration = 1200.0
    groups = [[0, 1], [2, 3]]
    u = np.array([1.0, 1.6, 2.1, 1.1])
    u = u / np.linalg.norm(u)
    rho = 0.5 * np.exp(1j * np.pi / 4)
    v12 = np.array([[1.0, rho], [np.conj(rho), 1.0]])
    data = async_mod.simulate_asynchronous(
        1.0, 0.01, 1.0, [1.0, 1.5], u, v12, groups, fs, duration, seed=4, q=0
    )

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "async.csv"
        np.savetxt(csv_path, data, delimiter=",")
        out_path = Path(tmp) / "async.json"
        spec = {
            "fs": fs,
            "asynchronous": {
                "data": {"path": str(csv_path), "format": "csv", "fs": fs},
                "groups": groups,
                "bands": [[0.9, 1.1]],
                "f0": [1.0],
                "n_modes": 1,
            },
            "geometry": GEOMETRY4,
            "figures": {"dir": str(Path(tmp) / "fig")},
            "output": {"path": str(out_path)},
        }
        res = io_json.run_from_dict(spec)

        asm = res.get("assembly")
        check("异步单阶有组装分组", isinstance(asm, dict), type(asm).__name__)
        n = len(asm["posterior_covariance"])
        check("索引与协方差维数一致", len(asm["index"]) == n, f"{len(asm['index'])} vs {n}")
        names = [item["name"] for item in asm["index"]]
        check("索引含 f", "f[0]" in names, names[:5])
        check("索引含各测组 Se", "Se[0]" in names and "Se[1]" in names)
        check("索引含相干实虚部", "Rev[0][1]" in names and "Imv[0][1]" in names)
        check("索引含振型切空间", any(x.startswith("phi[0].theta") for x in names))

        unc = asm["uncertainty"]
        total = unc["trace_total"]
        parts = unc["trace_assembly"] + unc["trace_recognition"]
        check(
            "不确定性分解可加",
            abs(total - parts) <= 1e-12 * max(abs(total), 1.0),
            f"{total:.6e} vs {parts:.6e}",
        )
        check("总不确定性为正", unc["total_coefv"] > 0.0, unc["total_coefv"])

        su = asm["shape_uncertainty"]
        check("组装振型逐通道标准差", len(su["std_per_channel"]) == 4)

        spa = asm["spatial"]
        check("组装可视化可用", spa["available"] is True, spa.get("reason"))
        fig = Path(spa["figures"][0]["path"])
        check("组装图片已落盘", fig.exists() and fig.stat().st_size > 1000, fig.name)

        rec = res.get("recognition") or {}
        check("异步识别频带表", len(rec.get("bands", [])) == 1, rec.get("bands"))
        check("异步识别阶次映射", len(rec.get("modes", [])) == 1)

        saved = json.loads(out_path.read_text(encoding="utf-8"))
        check("组装分组已写入结果文件", "assembly" in saved)
        check("结果文件可读回索引", len(saved["assembly"]["index"]) == n)

    return failures


def _test_async_multi_assembly() -> list[str]:
    """异步多阶的组装结果：多阶索引、逐阶振型不确定性、可视化。"""
    failures: list[str] = []

    def check(label, ok, detail=""):
        print(("[OK] " if ok else "[FAIL] ") + label + ("  " + str(detail) if detail else ""))
        if not ok:
            failures.append(label)

    from bayoma import asynchronous_multi as am
    from bayoma import io_json

    fs = 100.0
    duration = 1200.0
    f_true = [1.0, 1.4]
    z_true = [0.01, 0.012]
    s_true = np.array([[1.0, 0.2], [0.2, 0.9]])
    se_true = [1.0, 1.5]
    groups = [[0, 1, 2], [3, 4]]
    phi = np.array(
        [[1.0, 0.4], [1.8, -0.5], [2.2, 0.6], [1.2, 1.0], [0.7, -1.1]], dtype=float
    )
    phi = phi / np.linalg.norm(phi, axis=0, keepdims=True)
    rho = 0.5 * np.exp(1j * np.pi / 4)
    vij = {(0, 1): np.array([[1.0, rho], [np.conj(rho), 1.0]])}
    data = am.simulate_asynchronous_multi(
        f_true, z_true, s_true, se_true, phi, vij, groups, fs, duration, seed=33, q=0
    )

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "am.csv"
        np.savetxt(csv_path, data, delimiter=",")
        out_path = Path(tmp) / "am.json"
        spec = {
            "fs": fs,
            "asynchronous": {
                "data": {"path": str(csv_path), "format": "csv", "fs": fs},
                "groups": groups,
                "bands": [[0.9, 1.5]],
                "f0": f_true,
                "n_modes": 2,
            },
            "geometry": GEOMETRY5,
            "figures": {"dir": str(Path(tmp) / "fig")},
            "output": {"path": str(out_path)},
        }
        res = io_json.run_from_dict(spec)
        asm = res.get("assembly")
        check("异步多阶有组装分组", isinstance(asm, dict))
        n = len(asm["posterior_covariance"])
        check("索引与协方差维数一致", len(asm["index"]) == n, f"{len(asm['index'])} vs {n}")
        names = [item["name"] for item in asm["index"]]
        check("索引含两阶频率", "f[0]" in names and "f[1]" in names)
        check("索引含逐阶振型", "phi[0].theta[0]" in names and "phi[1].theta[0]" in names)
        check(
            "索引含相干模与相位",
            any(x.startswith("|v|") for x in names) and any(x.startswith("phase v") for x in names),
        )
        per_mode = asm.get("shape_uncertainty_per_mode")
        check("逐阶振型不确定性", isinstance(per_mode, list) and len(per_mode) == 2, per_mode)
        check("逐阶不确定性为正", all(item["coefv"] > 0.0 for item in per_mode))
        unc = asm["uncertainty"]
        check(
            "不确定性分解可加",
            abs(unc["trace_total"] - (unc["trace_assembly"] + unc["trace_recognition"]))
            <= 1e-12 * max(abs(unc["trace_total"]), 1.0),
        )
        spa = asm["spatial"]
        check("组装可视化可用", spa["available"] is True, spa.get("reason"))
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        check("结果文件含组装分组", "assembly" in saved)

    return failures


def main() -> int:
    print("=== 同步识别的结果分组 ===")
    failures = _test_sync()
    print()
    print("=== 密集模态的区分度 ===")
    failures += _test_dense_pair()
    print()
    print("=== 异步单阶的组装分组 ===")
    failures += _test_async_assembly()
    print()
    print("=== 异步多阶的组装分组 ===")
    failures += _test_async_multi_assembly()

    print()
    if failures:
        print(f"失败 {len(failures)} 项：" + "；".join(failures))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
