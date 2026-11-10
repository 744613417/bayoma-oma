"""JSON 接口契约的机读校验。

用 ``schema/job.schema.json`` 与 ``schema/result.schema.json`` 校验任务描述与识别结果，
并实际跑一遍四类识别（同步单阶、同步多阶、多测组、异步单阶），确认两套实现写出的
字段与 schema 一致。

运行::

    E:\\anaconda3\\python.exe python\\tests\\test_schema.py

依赖 ``jsonschema``（缺失时跳过校验并给出提示，不算失败，因为它是可选依赖）。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import bayoma  # noqa: E402
import bayoma.asynchronous as async_mod  # noqa: E402
from bayoma import io_json  # noqa: E402

SCHEMA_DIR = ROOT / "schema"
JOB_SCHEMA = SCHEMA_DIR / "job.schema.json"
RESULT_SCHEMA = SCHEMA_DIR / "result.schema.json"

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    """记录一项检查结果。"""
    global failures
    if ok:
        print(f"[OK] {name}" + (f"  {detail}" if detail else ""))
    else:
        failures += 1
        print(f"[FAIL] {name}" + (f"  {detail}" if detail else ""))


def load_schemas() -> tuple:
    """读取两份 schema，缺 ``jsonschema`` 时返回 ``(None, None, None)``。"""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        print("未安装 jsonschema，跳过 schema 校验（pip install jsonschema 后可启用）")
        return None, None, None
    return (
        json.loads(JOB_SCHEMA.read_text(encoding="utf-8")),
        json.loads(RESULT_SCHEMA.read_text(encoding="utf-8")),
        jsonschema,
    )


def validate(js, schema: dict, payload: dict, label: str) -> None:
    """用给定 schema 校验 payload，把错误逐条打印。"""
    validator = js.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    if not errors:
        check(label, True, "")
        return
    detail = "；".join(
        f"{'/'.join(str(p) for p in e.absolute_path) or '<根>'}: {e.message}"
        for e in errors[:4]
    )
    check(label, False, detail)


def write_csv(path: Path, data: np.ndarray) -> None:
    """把数据矩阵写成无表头 CSV。"""
    np.savetxt(path, data, delimiter=",")


def build_cases(tmp: Path) -> list:
    """构造四类识别的任务描述与数据，返回 ``[(名称, 描述)]``。"""
    cases = []

    # 一、同步单阶
    sync1 = bayoma.simulate_oma_data(
        [1.0], [0.01], np.array([[1.0], [2.0], [2.0]]) / 3.0,
        np.array([[1.0]]), 1.0, 100.0, 600.0, seed=7,
    )
    p = tmp / "sync1.csv"
    write_csv(p, sync1)
    cases.append((
        "同步单阶",
        {
            "data": {"path": str(p), "format": "csv", "fs": 100.0},
            "bands": [[0.9, 1.1]],
            "f0": [[1.0]],
            "options": {"q": 0, "mute": 1},
        },
    ))

    # 二、同步多阶（密集模态）
    phi2 = np.column_stack([
        np.array([1.0, 2.0, 2.0]) / 3.0,
        np.array([-2.0, 1.0, 2.0]) / 3.0,
    ])
    coh = 0.5 * np.exp(1j * np.pi / 4.0)
    s2 = np.array([[1.0, coh], [np.conj(coh), 1.0]])
    sync2 = bayoma.simulate_oma_data(
        [1.0, 1.02], [0.01, 0.01], phi2, s2, 1.0, 100.0, 1200.0, seed=3,
    )
    p = tmp / "sync2.csv"
    write_csv(p, sync2)
    cases.append((
        "同步多阶",
        {
            "data": {"path": str(p), "format": "csv", "fs": 100.0},
            "bands": [[0.9, 1.1]],
            "f0": [[1.0, 1.02]],
            "options": {"q": 0, "mute": 1},
        },
    ))

    # 三、多测组（四测组覆盖六个自由度）
    n_dof = 6
    glob = np.array([0.162, 0.286, 0.315, 0.477, 0.546, 0.516])
    glob = glob / np.linalg.norm(glob)
    setup_specs = [
        {"dofs": [0, 1, 2]},
        {"dofs": [2, 3, 4]},
        {"dofs": [3, 4]},
        {"dofs": [4, 5]},
    ]
    for item in setup_specs:
        item.update({
            "fs": 100.0,
            "duration": 600.0,
            "Se": 1.0,
            "S": np.array([[1.0]]),
            "f": [3.0],
            "z": [0.01],
        })
    datas = bayoma.simulate_multi_setup_data(setup_specs, glob, n_dof, seed=21)
    entries = []
    for i, (data, item) in enumerate(zip(datas, setup_specs)):
        p = tmp / f"setup{i}.csv"
        write_csv(p, data)
        entries.append({
            "data": {"path": str(p), "format": "csv", "fs": 100.0},
            "dofs": item["dofs"],
            "bands": [[2.9, 3.1]],
        })
    cases.append((
        "多测组",
        {"fs": 100.0, "multi_setup": {"n_dof": n_dof, "setups": entries},
         "options": {"q": 0}},
    ))

    # 四、异步单阶（两组各两通道）
    groups = [[0, 1], [2, 3]]
    shape = np.array([1.0, 2.0, 2.0, 1.0])
    shape = shape / np.linalg.norm(shape)
    data_a = async_mod.simulate_asynchronous(
        1.0, 0.01, 1.0, [1.0, 1.2], shape,
        np.array([[1.0, 0.4], [0.4, 1.0]]), groups, 100.0, 600.0, seed=5,
    )
    p = tmp / "async1.csv"
    write_csv(p, data_a)
    cases.append((
        "异步单阶",
        {
            "asynchronous": {
                "data": {"path": str(p), "format": "csv", "fs": 100.0},
                "groups": groups,
                "bands": [[0.9, 1.1]],
            },
            "options": {"q": 0},
        },
    ))
    return cases


def main() -> int:
    """执行全部检查。"""
    job_schema, result_schema, js = load_schemas()
    if job_schema is not None:
        check("两份 schema 可解析为 draft 2020-12 文档",
              job_schema.get("$schema", "").endswith("2020-12/schema")
              and result_schema.get("$schema", "").endswith("2020-12/schema"))

        # 反例：缺 data/multi_setup/asynchronous 三者应被拒
        validator = js.Draft202012Validator(job_schema)
        bad = list(validator.iter_errors({"fs": 100.0}))
        check("描述缺三类测量组织中任一类时被拒", bool(bad),
              "；".join(e.message for e in bad[:2]))

        # 反例：asynchronous 的 groups 每项只给一个通道应被拒
        bad = list(validator.iter_errors({
            "asynchronous": {
                "data": {"path": "x.csv"},
                "groups": [[0]],
                "bands": [[0.9, 1.1]],
            }
        }))
        check("异步测组少于两个通道时被拒", bool(bad),
              "；".join(e.message for e in bad[:2]))

        # 反例：结果缺必需字段应被拒
        rvalidator = js.Draft202012Validator(result_schema)
        bad = list(rvalidator.iter_errors({"f": [1.0]}))
        check("结果缺必需字段时被拒", bool(bad),
              "；".join(e.message for e in bad[:2]))

    with tempfile.TemporaryDirectory(prefix="bayoma_schema_") as tmpdir:
        tmp = Path(tmpdir)
        for name, spec in build_cases(tmp):
            if job_schema is not None:
                validate(js, job_schema, spec, f"{name}：描述通过 schema 校验")
            output = tmp / f"{name}.json"
            spec.setdefault("output", {})["path"] = str(output)
            result = io_json.run_from_dict(spec)

            # 结果同样要能通过 schema
            if result_schema is not None:
                validate(js, result_schema, result, f"{name}：结果通过 schema 校验")

            check(f"{name}：结果已写盘且可读回",
                  output.is_file() and json.loads(output.read_text(encoding="utf-8"))["n_modes"] >= 1,
                  f"n_modes = {result['n_modes']}")

            # 复数字段按 {"re","im"} 给出时，两个键长度应一致
            phi = result["phi"]
            if isinstance(phi, dict):
                check(f"{name}：复振型的 re/im 形状一致",
                      np.shape(phi["re"]) == np.shape(phi["im"]),
                      f"{np.shape(phi['re'])}")

    print()
    if failures == 0:
        print("全部通过")
        return 0
    print(f"{failures} 项检查未通过")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
