"""通用工具：复数序列化、计时、日志、参数校验。

本模块只提供基础设施，不包含任何模态识别算法。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable

import numpy as np

LOG = logging.getLogger("bayoma")


def complex_to_json(value: Any) -> Any:
    """把含复数的嵌套结构转成可 JSON 序列化的形式。

    约定：复数写成 ``{"re": 实部, "im": 虚部}``；``numpy`` 数组写成嵌套列表。
    """
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, np.complexfloating):
        return {"re": float(np.real(value)), "im": float(np.imag(value))}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [
                [complex_to_json(x) for x in row] if value.ndim > 1 else complex_to_json(x)
                for row in value
            ]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): complex_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [complex_to_json(v) for v in value]
    return value


def dumps(payload: Any, indent: int | None = 2) -> str:
    """按统一约定把结果字典序列化为 JSON 文本。"""
    return json.dumps(complex_to_json(payload), ensure_ascii=False, indent=indent)


class Timer:
    """秒表，用于填充 ``time_req`` 输出字段。"""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0


def as_float_array(value: Any, name: str) -> np.ndarray:
    """把输入转成 ``float64`` 数组；无法转换时报出字段名。"""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 依赖外部输入
        raise ValueError(f"输入字段 {name} 无法转换为数值数组：{exc}") from exc
    if arr.size == 0:
        raise ValueError(f"输入字段 {name} 为空")
    return arr


def as_index_list(value: Any, name: str) -> list[int] | None:
    """把 0/1 基索引输入统一成 Python 整数列表；``None`` 表示全部。"""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.ndim != 1:
        raise ValueError(f"输入字段 {name} 必须是一维索引数组")
    if not np.all(np.equal(np.mod(arr, 1), 0)):
        raise ValueError(f"输入字段 {name} 必须为整数索引")
    return [int(x) for x in arr]


def positive(value: Any, name: str) -> float:
    """读取必须为正的标量。"""
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"输入字段 {name} 必须为正的有限数，收到 {value!r}")
    return out


def band_pairs(value: Any, name: str) -> np.ndarray:
    """把频带输入规整成 ``(nb, 2)`` 数组，并检查上下界顺序。"""
    arr = as_float_array(value, name)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"输入字段 {name} 若是向量必须恰好两个元素")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"输入字段 {name} 必须是 (nb, 2) 数组")
    if np.any(arr[:, 0] >= arr[:, 1]):
        raise ValueError(f"输入字段 {name} 每一行需满足 下界 < 上界")
    return arr


def ensure_finite(matrix: np.ndarray, name: str) -> np.ndarray:
    """检查矩阵元素有限，返回原矩阵便于链式调用。"""
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 含非有限数值，请检查输入数据是否存在缺测或异常")
    return matrix


def summarize_list(values: Iterable[float], digits: int = 6) -> str:
    """把一列数格式化为便于日志阅读的字符串。"""
    return ", ".join(f"{v:.{digits}g}" for v in values)
