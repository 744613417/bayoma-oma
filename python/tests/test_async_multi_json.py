"""异步数据多阶识别的 JSON 接口校核：复模态形状与复模态力谱的导出口径。

单测组（``ng == 1``）且多阶（``m > 1``）时，模型取 Hermitian 复模态力谱矩阵与复模态
形状（第 10.4 节与第 13 章的模型类一致）。此时 ``io_json`` 必须按 ``{"re": …, "im": …}``
序列化复数数组；用 ``dtype=float`` 强转会静默丢弃虚部，使调用方拿到错误的模态形状。

运行：``E:\\anaconda3\\python.exe python\\tests\\test_async_multi_json.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bayoma  # noqa: E402
from bayoma import asynchronous_multi as am  # noqa: E402
from bayoma import io_json  # noqa: E402

F_TRUE = [1.0, 1.4]
Z_TRUE = [0.01, 0.012]
PHI_TRUE = np.array(
    [[1.0, 0.5], [1.8, -0.4], [2.2, 0.6], [1.2, 1.0], [0.7, -1.1]], dtype=float
)
PHI_TRUE = PHI_TRUE / np.linalg.norm(PHI_TRUE, axis=0, keepdims=True)
S_TRUE = np.array([[1.0, 0.2], [0.2, 0.9]], dtype=float)
SE_TRUE = [1.0]
FS = 100.0
DURATION = 600.0
SEED = 33
BAND = (0.9, 1.5)
GROUPS = [[0, 1, 2, 3, 4]]

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(("[OK ] " if ok else "[FAIL] ") + label + (("  " + detail) if detail else ""))
    if not ok:
        failures.append(label)


def _write_job(work: Path, data: np.ndarray) -> Path:
    csv_path = work / "data.csv"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10e")
    spec = {
        "fs": FS,
        "asynchronous": {
            "data": {"path": str(csv_path), "format": "csv", "fs": FS},
            "groups": GROUPS,
            "bands": [list(BAND)],
            "f0": F_TRUE,
            "n_modes": 2,
        },
        "options": {"q": 0},
        "output": {"path": str(work / "result.json")},
    }
    spec_path = work / "job.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return spec_path


def test_complex_shape_export() -> None:
    """单测组多阶识别的 JSON 结果保留复模态形状的虚部。"""
    t0 = time.perf_counter()
    data = am.simulate_asynchronous_multi(
        F_TRUE, Z_TRUE, S_TRUE, SE_TRUE, PHI_TRUE, {}, GROUPS, FS, DURATION, seed=SEED
    )
    work = Path(tempfile.mkdtemp(prefix="bayoma_async_multi_json_"))
    _write_job(work, data)

    result = io_json.run_file(str(work / "job.json"))
    payload = result if isinstance(result, dict) else json.loads(json.dumps(result))

    phi = payload["phi"]
    check("模态形状以复数形式序列化",
          isinstance(phi, dict) and "re" in phi and "im" in phi,
          f"type={type(phi).__name__}")
    if isinstance(phi, dict):
        re_part = np.asarray(phi["re"], dtype=float)
        im_part = np.asarray(phi["im"], dtype=float)
        check("模态形状形状正确", re_part.shape == (len(PHI_TRUE), 2),
              f"shape={re_part.shape}")
        check("模态形状确有非零虚部", bool(np.any(np.abs(im_part) > 1.0e-9)),
              f"max|Im| = {np.max(np.abs(im_part)):.3e}")
        est = re_part + 1j * im_part
        mac = np.abs(est.conj().T @ (PHI_TRUE * np.exp(1j * np.angle(
            np.sum(est.conj() * PHI_TRUE, axis=0))))) ** 2 / (
            np.sum(np.abs(est) ** 2, axis=0) * np.sum(PHI_TRUE ** 2, axis=0)
        )
        check("复模态形状与真值 MAC 接近 1",
              bool(np.all(np.real(np.diag(mac)) > 0.99)),
              f"MAC = {np.round(np.real(np.diag(mac)), 6)}")

    s_off = payload["coefv"].get("S_off")
    check("coefv.S_off 保留复数", isinstance(s_off, dict) and "im" in s_off,
          f"type={type(s_off).__name__}")

    check("顶层 f 与真值接近",
          bool(np.all(np.abs(np.asarray(payload["f"]) / np.asarray(F_TRUE) - 1.0) < 2.0e-2)),
          f"f = {np.round(np.asarray(payload['f']), 6)}")

    text = json.dumps(payload, ensure_ascii=False)
    check("结果可被 json.dumps 序列化", len(text) > 0, f"{len(text)} 字符")

    written = json.loads((work / "result.json").read_text(encoding="utf-8"))
    written_phi = written["phi"]
    ok = isinstance(written_phi, dict) and "im" in written_phi
    max_im = (float(np.max(np.abs(np.asarray(written_phi["im"]))))
              if ok else float("nan"))
    check("写盘文件中的虚部保留", ok and max_im > 1.0e-9, f"max|Im| = {max_im:.3e}")
    check("结果带 n_modes = 2", int(payload.get("n_modes", 0)) == 2,
          f"n_modes = {payload.get('n_modes')}")
    print(f"  （本项用时 {time.perf_counter() - t0:.1f} s）")


def test_multi_group_real_shape_stays_real() -> None:
    """多测组时模态力谱取实对称、模态形状取实向量，导出不应出现虚部。"""
    t0 = time.perf_counter()
    groups = [[0, 1, 2], [3, 4]]
    vij = {(0, 1): np.array([[1.0, 0.5 * np.exp(1j * np.pi / 4)],
                             [np.conj(0.5 * np.exp(1j * np.pi / 4)), 1.0]])}
    data = am.simulate_asynchronous_multi(
        F_TRUE, Z_TRUE, S_TRUE, [1.0, 1.5], PHI_TRUE, vij, groups, FS, 600.0, seed=SEED
    )
    work = Path(tempfile.mkdtemp(prefix="bayoma_async_multi_json_real_"))
    csv_path = work / "data.csv"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.10e")
    spec = {
        "fs": FS,
        "asynchronous": {
            "data": {"path": str(csv_path), "format": "csv", "fs": FS},
            "groups": groups,
            "bands": [list(BAND)],
            "f0": F_TRUE,
            "n_modes": 2,
        },
        "options": {"q": 0},
    }
    payload = io_json.run_from_dict(spec)
    phi = payload["phi"]
    ok_list = isinstance(phi, list)
    check("多测组时模态形状按实数列表给出", ok_list, f"type={type(phi).__name__}")
    if ok_list:
        arr = np.asarray(phi, dtype=float)
        check("多测组时模态形状形状正确", arr.shape == (len(PHI_TRUE), 2),
              f"shape={arr.shape}")
    s_mat = payload["S"][0]
    max_im = 0.0
    if isinstance(s_mat, dict):
        max_im = float(np.max(np.abs(np.asarray(s_mat.get("im", 0.0)))))
    check("多测组时模态力谱虚部为零", max_im < 1.0e-12, f"max|Im S| = {max_im:.3e}")
    print(f"  （本项用时 {time.perf_counter() - t0:.1f} s）")


def main() -> int:
    print("异步数据多阶识别的 JSON 导出校核\n")
    test_complex_shape_export()
    print()
    test_multi_group_real_shape_stays_real()
    print()
    if failures:
        print(f"失败 {len(failures)} 项")
        for item in failures:
            print("  -", item)
        return 1
    print("异步多阶 JSON 接口：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
