"""用 ``schema/result.schema.json`` 校验 MATLAB 侧写出的结果文件。

MATLAB 侧的两类产物（同步识别、异步组装）与 Python 侧共用同一份 JSON 契约，
本脚本按该契约校验，确认两侧字段形状一致。数据集来自
``matlab/tests/test_results.m`` 写入 ``tempdir/bayoma_test_results`` 的文件。

用法：
    E:/anaconda3/python.exe python/tests/check_matlab_schema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schema" / "result.schema.json"

try:
    import jsonschema
except ImportError:  # pragma: no cover - 仅在缺少依赖时走到
    jsonschema = None


CANDIDATES = [
    ("异步单阶组装结果", "async_single_result.json"),
    ("异步多阶组装结果", "async_multi_result.json"),
]


def work_dir() -> Path:
    """MATLAB 的 tempdir 在本机解析为 D:\\SystemTemp。"""
    for base in (Path("D:/SystemTemp"), Path.home() / "AppData/Local/Temp"):
        path = base / "bayoma_test_results"
        if path.is_dir():
            return path
    raise SystemExit("找不到 MATLAB 测试产物目录 bayoma_test_results")


def describe(err: "jsonschema.ValidationError") -> list[str]:
    """把 oneOf/anyOf 的失败展开到最内层，否则只看到一句笼统提示。"""
    if not err.context:
        where = "/".join(str(x) for x in err.path)
        return [f"{where} :: {err.message}"]
    lines: list[str] = []
    for sub in err.context:
        lines.extend(describe(sub))
    return lines


def main() -> int:
    if jsonschema is None:
        raise SystemExit("需要 jsonschema：pip install jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    directory = work_dir()
    failed = 0
    for label, name in CANDIDATES:
        path = directory / name
        if not path.is_file():
            print(f"[SKIP] {label}：{path} 不存在")
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        if not errors:
            groups = [k for k in ("recognition", "assembly") if doc.get(k)]
            print(f"[OK] {label}：schema 校验通过，含分组 {groups}")
            continue
        failed += 1
        print(f"[FAIL] {label}：{len(errors)} 处顶层不符")
        for err in errors:
            for line in describe(err)[:8]:
                print(f"        {line}")

    if failed:
        print(f"\n失败 {failed} 项")
        return 1
    print("\nMATLAB 侧结果文件与 JSON 契约一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
